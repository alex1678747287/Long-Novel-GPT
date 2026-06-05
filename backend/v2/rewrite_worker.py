"""Background worker for durable rewrite jobs.

The worker intentionally reuses the existing /v2/rewrite implementation for
the actual generation path, then adds job persistence, cancellation checks and
long-chapter segmentation around it.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from collections.abc import Callable, Iterable
from typing import Any

from flask import Flask

from . import api, storage


SEGMENT_REWRITE_SOURCE_THRESHOLD = 2200
SEGMENT_REWRITE_TARGET_CHARS = 2200
SEGMENT_REWRITE_MIN_CHARS = 900
# /v2/rewrite 对每次调用有"单段上限"，且依模型而定：DeepSeek 被 codex 压到 1600 字以提质量。
# 若 worker 仍按固定 2200 分段，切出的段会超过该上限被 /v2/rewrite 直接 413 拒绝（连质量门都到不了），
# 长章因此每段必失败→重试耗尽→回退旧稿。这里按模型实际上限分段；留一点安全余量避免边界越界。
SEGMENT_REWRITE_SAFETY_MARGIN = 200
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


def _job_wall_clock_seconds() -> float:
    """任务级总时限：超过它就停止继续重排，接受当前最佳候选。

    长开篇/难章在"分段×段内重试×requeue"相乘下可能跑很久，过去只有单次模型调用
    的 360s 上限、没有任务级总时限，导致一直磨到 requeue 耗尽才 error。这里给一个
    保守的兜底总时限（默认 15 分钟，可用环境变量覆盖），到点就落最佳候选而非继续磨。
    """
    try:
        value = float(os.environ.get("REWRITE_JOB_WALL_CLOCK_SECONDS", "900"))
    except (TypeError, ValueError):
        value = 900.0
    return max(120.0, min(3600.0, value))


REWRITE_JOB_WALL_CLOCK_SECONDS = _job_wall_clock_seconds()


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


def _quality_requires_model_retry(quality: dict[str, Any] | None) -> bool:
    if not quality:
        return False
    issues = quality.get("issues") or []
    if issues:
        return True
    score = quality.get("score")
    if score is not None:
        try:
            if float(score) < 75:
                return True
        except (TypeError, ValueError):
            pass
    status = str(
        quality.get("delivery_status")
        or quality.get("delivery_label")
        or quality.get("grade")
        or ""
    ).strip().lower()
    return status in {"review", "risk", "需复查", "高风险", "有风险"}


def _job_over_budget(
    job: dict[str, Any],
    job_started: float | None,
    payload: dict[str, Any] | None = None,
) -> bool:
    """是否应停止继续重排、接受当前最佳候选。

    满足任一条件即为 True：
      ① 已耗尽自动重试次数（retry_count >= MAX_JOB_AUTO_RETRIES）；
      ② 本轮 process_job 单次耗时超过 REWRITE_JOB_WALL_CLOCK_SECONDS；
      ③ 跨 requeue 的**总耗时**(从 payload.first_started_at 起算)超过该时限——
         每次 requeue 都是新的 process_job 调用、job_started 会重置，靠持久化在 payload
         里的首次开始时间(epoch)来给"反复重试"封一个总时间上界，避免长开篇磨很久。
    过去丢弃整份候选→回退旧稿/置空(quality_score=null)，现在改为落"当前最佳候选"。
    """
    if int(job.get("retry_count") or 0) >= MAX_JOB_AUTO_RETRIES:
        return True
    if job_started is not None and (time.monotonic() - job_started) > REWRITE_JOB_WALL_CLOCK_SECONDS:
        return True
    first_started = (payload or {}).get("first_started_at")
    if first_started is not None:
        try:
            if (time.time() - float(first_started)) > REWRITE_JOB_WALL_CLOCK_SECONDS:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _scene_fidelity_merged_quality(
    payload: dict[str, Any],
    rewritten: str,
    source: str,
    quality: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """对"将被接受或兜底落库"的候选做一次 LLM 忠实度审核，命中换戏跑题则把'跑题换戏'并入 issues
    (从而触发一次重试或在 best_effort 时如实标注)。换皮会改物件名，纯规则区分不了换皮与换戏，
    故只能用语义判断；为控成本只在候选"否则就会落库"时跑一次。任何异常都原样返回，不阻塞。"""
    if not quality:
        return quality
    # 特性开关：默认关闭(单测/无网环境零副作用)，线上 worker 用 -e REWRITE_SCENE_FIDELITY=1 开启。
    if os.environ.get("REWRITE_SCENE_FIDELITY", "0") != "1":
        return quality
    if any('跑题换戏' in str(item) for item in (quality.get('issues') or [])):
        return quality
    try:
        model_cfg = None
        model_id = payload.get('model_id')
        if model_id:
            model_cfg = api.registry.get_model(model_id)
        if model_cfg is None:
            model_cfg = api.registry.get_active_model()
        issue = api._scene_fidelity_issue(rewritten, source, model_cfg)
    except Exception:
        return quality
    if not issue:
        return quality
    merged = dict(quality)
    merged['issues'] = list(quality.get('issues') or []) + [issue]
    try:
        merged['score'] = max(0, int(merged.get('score') or 0) - 22)
    except (TypeError, ValueError):
        pass
    merged['delivery_status'] = 'risk'
    return merged


def _dialogue_boosted(
    payload: dict[str, Any],
    rewritten: str,
    source: str,
    quality: dict[str, Any] | None,
    score_func,
) -> tuple[str, dict[str, Any] | None]:
    """对"对话占比偏低"的候选做一次聚焦"对话化二次pass"(把叙述改成对白),让成品对白≥60%——
    客户要成品、不能让人工补对话。整章重写时模型不肯把回忆/铺垫对话化,单任务聚焦改写更易照做。
    特性开关 REWRITE_DIALOGUE_BOOST(默认关,单测/无网零副作用),线上 worker -e 开启。任何异常原样返回。"""
    if os.environ.get("REWRITE_DIALOGUE_BOOST", "0") != "1":
        return rewritten, quality
    if not quality or not any('对话占比偏低' in str(item) for item in (quality.get('issues') or [])):
        return rewritten, quality
    try:
        model_cfg = None
        model_id = payload.get('model_id')
        if model_id:
            model_cfg = api.registry.get_model(model_id)
        if model_cfg is None:
            model_cfg = api.registry.get_active_model()
        boosted = api._dialogue_boost(rewritten, source, model_cfg)
    except Exception:
        return rewritten, quality
    if not boosted or boosted == rewritten:
        return rewritten, quality
    try:
        return boosted, score_func(boosted, source)
    except Exception:
        return rewritten, quality


def _quality_retry_failure_message(quality: dict[str, Any] | None) -> str:
    if not quality:
        return "质量复查未通过：缺少质量评分"
    issues = [str(item).strip() for item in (quality.get("issues") or []) if str(item).strip()]
    if issues:
        return "质量复查未通过：" + "；".join(issues[:3])
    label = (
        quality.get("delivery_label")
        or quality.get("delivery_status")
        or quality.get("grade")
        or "未达到交付线"
    )
    score = quality.get("score")
    if score is None:
        return f"质量复查未通过：{label}"
    return f"质量复查未通过：{label}，评分 {score}"


def _quality_failure_hint_for_retry(error: str) -> str:
    """Preserve internal quality failure detail for the next model attempt."""
    message = re.sub(r"\s+", " ", (error or "").strip())
    if not message or "质量复查未通过" not in message:
        return ""
    return message[:900]


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


def _payload_rewrite_limit(payload: dict[str, Any]) -> int:
    """该 payload 所用模型在 /v2/rewrite 的单段上限（DeepSeek=1600，其它=split 目标）。"""
    text = payload.get("text") or ""
    model_cfg = None
    try:
        model_id = payload.get("model_id")
        if model_id:
            model_cfg = api.registry.get_model(model_id)
        if model_cfg is None:
            model_cfg = api.registry.get_active_model()
    except Exception:
        model_cfg = None
    try:
        return int(api._resolve_rewrite_target(text, None, model_cfg))
    except Exception:
        return SEGMENT_REWRITE_TARGET_CHARS


def _segment_threshold_for(payload: dict[str, Any]) -> int:
    """超过模型单段上限就必须分段（否则直连也会被 413）。对默认 2200 上限的模型保持原阈值。"""
    return min(SEGMENT_REWRITE_SOURCE_THRESHOLD, _payload_rewrite_limit(payload))


def _segment_target_for(payload: dict[str, Any]) -> int:
    """分段目标：上限更小（如 DeepSeek 1600）时落到 上限-余量；否则维持原 2200（不影响其它模型）。"""
    limit = _payload_rewrite_limit(payload)
    if limit < SEGMENT_REWRITE_TARGET_CHARS:
        return max(SEGMENT_REWRITE_MIN_CHARS, limit - SEGMENT_REWRITE_SAFETY_MARGIN)
    return SEGMENT_REWRITE_TARGET_CHARS


def _should_segment_payload(payload: dict[str, Any]) -> bool:
    if payload.get("internal_segment"):
        return False
    if payload.get("task_type") and payload.get("task_type") != "rewrite":
        return False
    if payload.get("force_internal_segment"):
        return len((payload.get("text") or "").strip()) > SEGMENT_REWRITE_MIN_CHARS * 2
    return len((payload.get("text") or "").strip()) > _segment_threshold_for(payload)


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
    analysis_data = api._resolve_analysis_data(
        payload.get("novel_id"),
        payload.get("chapter_id"),
    )
    name_map = api._analysis_name_map(analysis_data)
    rename_ledger = api._analysis_replacement_map(analysis_data)

    def score(rewritten: str, source: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if protected_terms:
            kwargs["protected_terms"] = protected_terms
        if name_map:
            kwargs["name_map"] = name_map
        if rename_ledger:
            kwargs["rename_ledger"] = rename_ledger
        if not kwargs:
            return api.score_rewrite_quality(rewritten, source)
        return api.score_rewrite_quality(rewritten, source, **kwargs)

    return score


def _is_truncated_output_error(exc: Exception) -> bool:
    return TRUNCATED_OUTPUT_ERROR in str(exc)


def _segment_payload_for_budget(payload: dict[str, Any], job_started: float | None) -> dict[str, Any]:
    """已用掉 60% 总时限后，把后续分段的 quality_mode 降到 balanced，避免每段还在 auto/deep
    跑满重试把总时长继续拉长（fast 会关掉格式自愈，故降到 balanced 而非 fast）。"""
    if job_started is None:
        return payload
    if (time.monotonic() - job_started) <= REWRITE_JOB_WALL_CLOCK_SECONDS * 0.6:
        return payload
    if str(payload.get("quality_mode") or "").strip().lower() not in {"", "auto", "deep"}:
        return payload
    downgraded = dict(payload)
    downgraded["quality_mode"] = "balanced"
    return downgraded


def _run_segment_piece(
    job: dict[str, Any],
    payload: dict[str, Any],
    chunk: str,
    *,
    floor: int,
    ceiling: int,
    depth: int = 0,
    job_started: float | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    if _job_is_canceled(job["id"]):
        raise RuntimeError("任务已取消")
    try:
        segment_result = run_rewrite_payload(
            _clean_segment_payload(_segment_payload_for_budget(payload, job_started), chunk),
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
                    job_started=job_started,
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
    job_started: float | None = None,
) -> dict[str, Any]:
    source = (payload.get("text") or "").strip()
    rewritten = (result.get("rewritten") or "").strip()
    if not rewritten:
        raise RuntimeError("模型返回空正文")
    analysis_data = api._resolve_analysis_data(
        payload.get("novel_id"),
        payload.get("chapter_id"),
    )
    name_map = api._analysis_name_map(analysis_data)
    # 确定性替换用"地名+术语+人名"合并表(跨章一致性):把地点/势力/标志术语也像人名一样锁死,
    # 不再只靠 prompt 软提示。质量门仍只用人名表(避免抬高重洗率)。
    replacement_map = api._analysis_replacement_map(analysis_data)
    repaired = api._repair_name_map_residue(
        api._repair_non_core_detail_residue(rewritten, source),
        replacement_map,
    )
    if repaired != rewritten:
        rewritten = repaired
        quality = _score_func_for_payload(payload)(rewritten, source)

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
    # 旧稿可能是修复前留下的、带替换脏数据(重复人名等)的版本；保留旧稿前先清一次，
    # 避免"新候选打不过旧脏稿→保留旧脏稿"把脏数据永久留在章节里。
    if existing_rewritten:
        cleaned_existing = api._repair_name_map_residue(existing_rewritten, replacement_map)
        if cleaned_existing != existing_rewritten:
            existing_rewritten = cleaned_existing
            existing_quality = score_func(existing_rewritten, source)
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
    # 对话不足的候选先做一次聚焦"对话化二次pass"(成品需对白≥60%,不让人工补)
    rewritten, quality = _dialogue_boosted(payload, rewritten, source, quality, score_func)
    # 候选若将被接受(无需重试)或已到预算上限(将兜底落库)，先做一次忠实度审核抓换戏跑题。
    if (not _quality_requires_model_retry(quality)) or _job_over_budget(job, job_started, payload):
        quality = _scene_fidelity_merged_quality(payload, rewritten, source, quality)
    best_effort = False
    if _quality_requires_model_retry(quality):
        if not _job_over_budget(job, job_started, payload):
            raise RuntimeError(_quality_retry_failure_message(quality))
        # 重试次数/总时限已耗尽：不再回退旧稿/置空，落"当前最佳候选"为成稿。
        best_effort = True

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
    if best_effort:
        persisted["best_effort"] = True
        persisted["quality_issues_remaining"] = quality.get("issues") or []
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
    if (
        not existing_rewritten
        or not existing_quality
        or _quality_requires_model_retry(existing_quality)
    ):
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
    quality_hint = _quality_failure_hint_for_retry(error)
    if quality_hint:
        next_payload["quality_failure_hint"] = quality_hint
        # 注意：不再在 retry>=2 时自动升 deep。deep 会把每段重试次数拉满，叠加分段后
        # 让"越重试越慢"，而 best_effort 兜底已能保证最终落稿，无需靠 deep 防 null。
        # 用户仍可显式选择 deep 模式。
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


def _run_segmented_job(
    job: dict[str, Any],
    payload: dict[str, Any],
    job_started: float | None = None,
) -> dict[str, Any]:
    source = (payload.get("text") or "").strip()
    chunks = api._chunk_text(source, _segment_target_for(payload))
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
            job_started=job_started,
        )
        _ensure_owned_update(job, phase="segment_rewrite", progress=ceiling)
        for rewritten, quality in segment_results:
            rewritten_parts.append(rewritten)
            segment_qualities.append(quality)

    if _job_is_canceled(job["id"]):
        raise RuntimeError("任务已取消")

    _ensure_owned_update(job, phase="merging", progress=84)
    merged = "\n\n".join(part.strip() for part in rewritten_parts if part.strip()).strip()
    # 长章合并后同样做"非核心细节 + 人名残留(含连续重复/首字粘连脏数据)"确定性修复，
    # 与直连路径对齐——此前分段路径完全跳过修复，长章易残留替换脏数据。
    analysis_data = api._resolve_analysis_data(
        payload.get("novel_id"),
        payload.get("chapter_id"),
    )
    name_map = api._analysis_name_map(analysis_data)
    replacement_map = api._analysis_replacement_map(analysis_data)
    merged = api._repair_name_map_residue(
        api._repair_non_core_detail_residue(merged, source),
        replacement_map,
    )
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
    if existing_rewritten:
        cleaned_existing = api._repair_name_map_residue(existing_rewritten, replacement_map)
        if cleaned_existing != existing_rewritten:
            existing_rewritten = cleaned_existing
            existing_quality = score_func(existing_rewritten, source)
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

    merged, final_quality = _dialogue_boosted(payload, merged, source, final_quality, score_func)
    if (not _quality_requires_model_retry(final_quality)) or _job_over_budget(job, job_started, payload):
        final_quality = _scene_fidelity_merged_quality(payload, merged, source, final_quality)
    best_effort = False
    if _quality_requires_model_retry(final_quality):
        if not _job_over_budget(job, job_started, payload):
            raise RuntimeError(_quality_retry_failure_message(final_quality))
        # 重试次数/总时限已耗尽：落已合并的"当前最佳候选"，不回退旧稿/置空。
        best_effort = True

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
    if best_effort:
        result["best_effort"] = True
        result["quality_issues_remaining"] = final_quality.get("issues") or []
    _ensure_owned_update(job, phase="saving", progress=92)
    return _finish_with_chapter_result(job, payload, result, merged, final_quality)


def process_job(job: dict[str, Any]) -> bool:
    if not job:
        return False
    if job.get("status") not in {"queued", "running"}:
        return False
    payload: dict[str, Any] = {}
    payload_loaded = False
    job_started = time.monotonic()
    try:
        payload = _job_payload(job)
        payload_loaded = True
        # 记录首次开始时间(epoch)，供跨 requeue 的总时限判定；requeue 通过 dict(payload) 自动延续。
        payload.setdefault("first_started_at", time.time())
        if _job_is_canceled(job["id"]):
            return False
        _ensure_owned_update(job, status="running", phase="initial", progress=5)
        if payload.get("chapter_id"):
            storage.update_chapter(payload["chapter_id"], status="running")

        did_segment = _should_segment_payload(payload)
        if did_segment:
            _run_segmented_job(job, payload, job_started=job_started)
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
            result = _persist_direct_rewrite_result(
                job, payload, result, quality, job_started=job_started
            )
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
