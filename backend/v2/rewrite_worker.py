"""Background worker for durable rewrite jobs.

The worker intentionally reuses the existing /v2/rewrite implementation for
the actual generation path, then adds job persistence, cancellation checks and
long-chapter segmentation around it.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from collections.abc import Callable, Iterable
from typing import Any

from flask import Flask

from . import api, storage


SEGMENT_REWRITE_SOURCE_THRESHOLD = 2200
SEGMENT_REWRITE_TARGET_CHARS = 2200
SEGMENT_REWRITE_MIN_CHARS = 900
CLAIM_LEASE_SECONDS = 1800
TRUNCATED_OUTPUT_ERROR = "模型输出达到本次最大生成长度"

_internal_app: Flask | None = None


def _max_job_auto_retries() -> int:
    try:
        value = int(os.environ.get("REWRITE_JOB_MAX_AUTO_RETRIES", "4"))
    except (TypeError, ValueError):
        value = 4
    return max(1, min(8, value))


MAX_JOB_AUTO_RETRIES = _max_job_auto_retries()


class JobOwnershipLost(RuntimeError):
    """Raised when a canceled or reclaimed job should stop without side effects."""


def _get_internal_app() -> Flask:
    global _internal_app
    if _internal_app is None:
        app = Flask("rewrite_worker_internal")
        app.register_blueprint(api.v2_bp)
        _internal_app = app
    return _internal_app


def _parse_sse_events(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in (body or "").splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[len("data:"):].strip()
        if not raw:
            continue
        try:
            events.append(json.loads(raw))
        except Exception:
            continue
    return events


def _iter_sse_events(chunks: Iterable[bytes | str]) -> Iterable[dict[str, Any]]:
    buffer = ""
    for raw in chunks:
        if not raw:
            continue
        part = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
        buffer += part.replace("\r\n", "\n")
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            yield from _parse_sse_events(block)
    if buffer.strip():
        yield from _parse_sse_events(buffer)


def _event_progress(
    event: dict[str, Any],
    *,
    source_len: int,
    floor: int,
    ceiling: int,
) -> int:
    if event.get("done"):
        return ceiling
    phase = str(event.get("phase") or "")
    if phase in {"format_retry", "quality_retry", "quality_review"}:
        fraction = 0.82
    elif event.get("rewritten"):
        fraction = min(0.78, max(0.15, len(str(event.get("rewritten") or "")) / max(source_len, 1)))
    else:
        fraction = 0.12
    return max(floor, min(ceiling, floor + int((ceiling - floor) * fraction)))


def _progress_callback(
    job: dict[str, Any],
    *,
    source_len: int,
    floor: int,
    ceiling: int,
    default_phase: str,
) -> Callable[[dict[str, Any]], None]:
    last = {"time": 0.0, "phase": "", "progress": -1}

    def callback(event: dict[str, Any]) -> None:
        if event.get("done"):
            return
        phase = str(event.get("phase") or default_phase)
        progress = _event_progress(event, source_len=source_len, floor=floor, ceiling=ceiling)
        now = time.time()
        if (
            progress == last["progress"]
            and phase == last["phase"]
            and now - float(last["time"]) < 8
        ):
            return
        _ensure_owned_update(job, phase=phase, progress=progress)
        last.update({"time": now, "phase": phase, "progress": progress})

    return callback


def run_rewrite_payload(
    payload: dict[str, Any],
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the existing streaming rewrite endpoint in-process and return its
    final SSE event. This keeps old compatibility behavior and tests intact.
    """
    app = _get_internal_app()
    response = app.test_client().post("/v2/rewrite", json=payload, buffered=False)
    if response.status_code >= 400:
        try:
            detail = response.get_json() or {}
            message = detail.get("error") or json.dumps(detail, ensure_ascii=False)
        except Exception:
            message = response.get_data(as_text=True)
        raise RuntimeError(message or f"rewrite failed with HTTP {response.status_code}")
    events: list[dict[str, Any]] = []
    try:
        for event in _iter_sse_events(response.response):
            events.append(event)
            if progress_cb:
                progress_cb(event)
    finally:
        response.close()
    if not events:
        raise RuntimeError("rewrite returned no events")
    final = events[-1]
    if final.get("error"):
        raise RuntimeError(str(final.get("error")))
    return final


def _job_payload(job: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(job.get("payload_json") or "{}")
    except Exception as exc:
        raise ValueError(f"invalid job payload: {exc}") from exc


def _job_is_canceled(job_id: str) -> bool:
    current = storage.get_rewrite_job(job_id)
    return bool(current and current.get("status") == "canceled")


def _update(job_or_id: dict[str, Any] | str, **fields: Any) -> dict[str, Any] | None:
    fields.setdefault("locked_at", time.time())
    if isinstance(job_or_id, dict):
        expected_locked_at = job_or_id.get("locked_at")
        if expected_locked_at is not None:
            fields.setdefault("expected_locked_at", expected_locked_at)
            fields.setdefault("require_status", "running")
        updated = storage.update_rewrite_job(job_or_id["id"], **fields)
        if updated:
            job_or_id.update(updated)
        return updated
    return storage.update_rewrite_job(job_or_id, **fields)


def _ensure_owned_update(job: dict[str, Any], **fields: Any) -> dict[str, Any]:
    updated = _update(job, **fields)
    if not updated:
        raise JobOwnershipLost("任务已取消或已被其他 worker 接管")
    return updated


def _chapter_update_for_rewrite(rewritten: str, quality: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rewritten": rewritten,
        "status": "done",
    }
    if quality:
        if quality.get("overlap4") is not None:
            payload["overlap"] = quality.get("overlap4")
        payload["quality_score"] = quality.get("score")
        payload["quality_grade"] = (
            quality.get("delivery_status")
            or quality.get("delivery_label")
            or quality.get("grade")
            or ""
        )
        payload["quality_issues"] = json.dumps(quality.get("issues") or [], ensure_ascii=False)
    return payload


def _finish_with_chapter_result(
    job: dict[str, Any],
    payload: dict[str, Any],
    result: dict[str, Any],
    rewritten: str,
    quality: dict[str, Any],
) -> dict[str, Any]:
    updated = storage.finish_rewrite_job_with_chapter(
        job["id"],
        expected_locked_at=job.get("locked_at"),
        result=result,
        status="done",
        phase="done",
        chapter_update=_chapter_update_for_rewrite(rewritten, quality),
        novel_id=payload.get("novel_id"),
        chapter_id=payload.get("chapter_id"),
    )
    if not updated:
        raise JobOwnershipLost("任务已取消或已被其他 worker 接管")
    job.update(updated)
    return result


def _should_segment_payload(payload: dict[str, Any]) -> bool:
    if payload.get("internal_segment"):
        return False
    if payload.get("task_type") and payload.get("task_type") != "rewrite":
        return False
    if payload.get("force_internal_segment"):
        return len((payload.get("text") or "").strip()) > SEGMENT_REWRITE_MIN_CHARS * 2
    return len((payload.get("text") or "").strip()) > SEGMENT_REWRITE_SOURCE_THRESHOLD


def _clean_segment_payload(payload: dict[str, Any], text: str) -> dict[str, Any]:
    segment_payload = dict(payload)
    segment_payload["text"] = text
    segment_payload["internal_segment"] = True
    segment_payload.pop("chapter_id", None)
    return segment_payload


def _score_func_for_payload(payload: dict[str, Any]):
    protected_terms = api._resolve_quality_protected_terms(
        payload.get("novel_id"),
        payload.get("chapter_id"),
    )

    def score(rewritten: str, source: str) -> dict[str, Any]:
        if not protected_terms:
            return api.score_rewrite_quality(rewritten, source)
        return api.score_rewrite_quality(
            rewritten,
            source,
            protected_terms=protected_terms,
        )

    return score


def _is_truncated_output_error(exc: Exception) -> bool:
    return TRUNCATED_OUTPUT_ERROR in str(exc)


def _run_segment_piece(
    job: dict[str, Any],
    payload: dict[str, Any],
    chunk: str,
    *,
    floor: int,
    ceiling: int,
    depth: int = 0,
) -> list[tuple[str, dict[str, Any]]]:
    if _job_is_canceled(job["id"]):
        raise RuntimeError("任务已取消")
    try:
        segment_result = run_rewrite_payload(
            _clean_segment_payload(payload, chunk),
            progress_cb=_progress_callback(
                job,
                source_len=len(chunk),
                floor=floor,
                ceiling=ceiling,
                default_phase="segment_rewrite",
            ),
        )
    except RuntimeError as exc:
        can_resplit = len(chunk) >= SEGMENT_REWRITE_MIN_CHARS * 2 and depth < 3
        if not (_is_truncated_output_error(exc) and can_resplit):
            raise
        target = max(SEGMENT_REWRITE_MIN_CHARS, len(chunk) // 2)
        subchunks = api._chunk_text(chunk, target)
        if len(subchunks) <= 1:
            raise
        results: list[tuple[str, dict[str, Any]]] = []
        _ensure_owned_update(job, phase="segment_resplit", progress=floor)
        for sub_index, subchunk in enumerate(subchunks):
            if _job_is_canceled(job["id"]):
                raise RuntimeError("任务已取消")
            sub_floor = floor + int((sub_index / max(1, len(subchunks))) * (ceiling - floor))
            sub_ceiling = floor + int(((sub_index + 1) / max(1, len(subchunks))) * (ceiling - floor))
            results.extend(
                _run_segment_piece(
                    job,
                    payload,
                    subchunk,
                    floor=sub_floor,
                    ceiling=max(sub_floor + 1, sub_ceiling),
                    depth=depth + 1,
                )
            )
        return results
    if _job_is_canceled(job["id"]):
        raise RuntimeError("任务已取消")
    rewritten = (segment_result.get("rewritten") or "").strip()
    quality = segment_result.get("quality") or _score_func_for_payload(payload)(rewritten, chunk)
    if not rewritten:
        raise RuntimeError("长章分段返回空正文")
    return [(rewritten, quality)]


def _finish_with_result(
    job: dict[str, Any],
    result: dict[str, Any],
    *,
    status: str = "done",
    phase: str = "done",
) -> None:
    updated = _ensure_owned_update(
        job,
        status=status,
        phase=phase,
        progress=100,
        result_json=json.dumps(result, ensure_ascii=False),
        error="" if status == "done" else result.get("error", ""),
    )
    if updated and status in {"error", "canceled"} and job.get("chapter_id"):
        storage.update_chapter(job["chapter_id"], status=status)


def _persist_direct_rewrite_result(
    job: dict[str, Any],
    payload: dict[str, Any],
    result: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    source = (payload.get("text") or "").strip()
    rewritten = (result.get("rewritten") or "").strip()
    if not rewritten:
        raise RuntimeError("模型返回空正文")

    protected_terms = api._resolve_quality_protected_terms(
        payload.get("novel_id"),
        payload.get("chapter_id"),
    )
    score_func = _score_func_for_payload(payload)
    existing_rewritten, existing_quality = api._existing_rewrite_quality(
        payload.get("chapter_id"),
        source,
        score_func,
    )
    kept_previous = False
    if existing_rewritten and existing_quality and not api._candidate_quality_is_better(
        quality,
        existing_quality,
        rewritten,
        existing_rewritten,
        source,
        protected_terms,
    ):
        rewritten = existing_rewritten
        quality = existing_quality
        kept_previous = True

    if _job_is_canceled(job["id"]):
        raise RuntimeError("任务已取消")

    persisted = dict(result)
    persisted.update(
        {
            "rewritten": rewritten,
            "raw": result.get("raw") or rewritten,
            "quality": quality,
            "saved": True,
        }
    )
    if kept_previous:
        persisted["kept_previous"] = True
    _ensure_owned_update(job, phase="saving", progress=94)
    return _finish_with_chapter_result(job, payload, persisted, rewritten, quality)


def _finish_with_existing_rewrite_if_usable(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    reason: str,
) -> bool:
    chapter_id = payload.get("chapter_id")
    if not chapter_id or _job_is_canceled(job["id"]):
        return False
    source = (payload.get("text") or "").strip()
    existing_rewritten, existing_quality = api._existing_rewrite_quality(
        chapter_id,
        source,
        _score_func_for_payload(payload),
    )
    if not existing_rewritten or not existing_quality or existing_quality.get("issues"):
        return False
    result = {
        "done": True,
        "rewritten": existing_rewritten,
        "raw": existing_rewritten,
        "quality": existing_quality,
        "saved": True,
        "kept_previous": True,
        "fallback_reason": reason,
    }
    _finish_with_chapter_result(
        job,
        payload,
        result,
        existing_rewritten,
        existing_quality,
    )
    return True


def _requeue_job_for_retry(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    error: str,
) -> bool:
    if _job_is_canceled(job["id"]):
        return False
    retry_count = int(job.get("retry_count") or 0)
    force_segment = (
        TRUNCATED_OUTPUT_ERROR in error
        and not payload.get("force_internal_segment")
        and len((payload.get("text") or "").strip()) >= SEGMENT_REWRITE_MIN_CHARS * 2
    )
    if retry_count >= MAX_JOB_AUTO_RETRIES and not force_segment:
        return False
    next_retry = retry_count + (0 if retry_count >= MAX_JOB_AUTO_RETRIES else 1)
    next_payload = dict(payload)
    if force_segment:
        next_payload["force_internal_segment"] = True
    updated = _update(
        job,
        status="queued",
        phase="retry_wait",
        progress=0,
        retry_count=next_retry,
        locked_at=None,
        payload_json=json.dumps(next_payload, ensure_ascii=False),
        error=f"自动重试中（{next_retry}/{MAX_JOB_AUTO_RETRIES}）：{error[:900]}",
    )
    if not updated or updated.get("status") != "queued":
        return False
    chapter_id = payload.get("chapter_id") or job.get("chapter_id")
    if chapter_id:
        storage.update_chapter(chapter_id, status="queued")
    return True


def _run_segmented_job(job: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    source = (payload.get("text") or "").strip()
    chunks = api._chunk_text(source, SEGMENT_REWRITE_TARGET_CHARS)
    if len(chunks) <= 1:
        return run_rewrite_payload(payload)

    _ensure_owned_update(
        job,
        phase="segmenting",
        progress=8,
    )
    rewritten_parts: list[str] = []
    segment_qualities: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if _job_is_canceled(job["id"]):
            raise RuntimeError("任务已取消")
        progress = 10 + int((index / max(1, len(chunks))) * 70)
        _ensure_owned_update(
            job,
            phase="segment_rewrite",
            progress=progress,
        )
        floor = 10 + int((index / max(1, len(chunks))) * 70)
        ceiling = 10 + int(((index + 1) / max(1, len(chunks))) * 70)
        segment_results = _run_segment_piece(
            job,
            payload,
            chunk,
            floor=floor,
            ceiling=ceiling,
        )
        _ensure_owned_update(job, phase="segment_rewrite", progress=ceiling)
        for rewritten, quality in segment_results:
            rewritten_parts.append(rewritten)
            segment_qualities.append(quality)

    if _job_is_canceled(job["id"]):
        raise RuntimeError("任务已取消")

    _ensure_owned_update(job, phase="merging", progress=84)
    merged = "\n\n".join(part.strip() for part in rewritten_parts if part.strip()).strip()
    score_func = _score_func_for_payload(payload)
    protected_terms = api._resolve_quality_protected_terms(
        payload.get("novel_id"),
        payload.get("chapter_id"),
    )
    final_quality = score_func(merged, source)
    if not merged:
        raise RuntimeError("长章合并后返回空正文")

    existing_rewritten, existing_quality = api._existing_rewrite_quality(
        payload.get("chapter_id"),
        source,
        score_func,
    )
    kept_previous = False
    if existing_rewritten and existing_quality and not api._candidate_quality_is_better(
        final_quality,
        existing_quality,
        merged,
        existing_rewritten,
        source,
        protected_terms,
    ):
        merged = existing_rewritten
        final_quality = existing_quality
        kept_previous = True

    result = {
        "done": True,
        "rewritten": merged,
        "raw": merged,
        "quality": final_quality,
        "saved": True,
        "segmented": True,
        "segment_count": len(rewritten_parts),
        "segment_qualities": segment_qualities,
        "kept_previous": kept_previous,
    }
    _ensure_owned_update(job, phase="saving", progress=92)
    return _finish_with_chapter_result(job, payload, result, merged, final_quality)


def process_job(job: dict[str, Any]) -> bool:
    if not job:
        return False
    if job.get("status") not in {"queued", "running"}:
        return False
    payload: dict[str, Any] = {}
    payload_loaded = False
    try:
        payload = _job_payload(job)
        payload_loaded = True
        if _job_is_canceled(job["id"]):
            return False
        _ensure_owned_update(job, status="running", phase="initial", progress=5)
        if payload.get("chapter_id"):
            storage.update_chapter(payload["chapter_id"], status="running")

        did_segment = _should_segment_payload(payload)
        if did_segment:
            _run_segmented_job(job, payload)
            return True
        else:
            direct_payload = dict(payload)
            direct_payload.pop("chapter_id", None)
            result = run_rewrite_payload(
                direct_payload,
                progress_cb=_progress_callback(
                    job,
                    source_len=len((payload.get("text") or "").strip()),
                    floor=8,
                    ceiling=92,
                    default_phase="generating",
                ),
            )

        if _job_is_canceled(job["id"]):
            return False
        _ensure_owned_update(job, phase="quality_review", progress=92)
        quality = result.get("quality") or {}
        if (
            not did_segment
            and payload.get("chapter_id")
            and (payload.get("task_type") in (None, "rewrite"))
        ):
            result = _persist_direct_rewrite_result(job, payload, result, quality)
            return True
        _finish_with_result(job, result)
        return True
    except Exception as exc:
        if isinstance(exc, JobOwnershipLost):
            return False
        if _job_is_canceled(job["id"]):
            return False
        error = str(exc) or exc.__class__.__name__
        if not payload_loaded:
            _finish_with_result(
                job,
                {"error": error},
                status="error",
                phase="error",
            )
            return False
        if _finish_with_existing_rewrite_if_usable(job, payload, reason=error):
            return True
        current_job = storage.get_rewrite_job(job["id"]) or job
        if _requeue_job_for_retry(current_job, payload, error=error):
            return False
        traceback.print_exc()
        _finish_with_result(
            job,
            {"error": error},
            status="error",
            phase="error",
        )
        return False


def run_once(worker_id: str) -> bool:
    job = storage.claim_rewrite_job(worker_id, CLAIM_LEASE_SECONDS)
    if not job:
        return False
    return process_job(job)


def run_forever() -> None:
    worker_id = os.environ.get("REWRITE_WORKER_ID") or f"worker-{os.getpid()}"
    idle_sleep = float(os.environ.get("REWRITE_WORKER_IDLE_SLEEP", "1.5"))
    while True:
        try:
            did_work = run_once(worker_id)
        except Exception:
            traceback.print_exc()
            did_work = False
        if not did_work:
            time.sleep(idle_sleep)


if __name__ == "__main__":
    run_forever()
