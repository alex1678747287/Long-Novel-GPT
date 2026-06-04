#!/usr/bin/env python3
"""只读抽检云端小说洗稿结果，并用本地质量规则重新评分。"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.v2 import api


DEFAULT_BASE_URL = "http://124.174.16.151:8233"


def _fetch_json(base_url: str, path: str) -> Any:
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)


def _load_novel(base_url: str, novel_id: str | None, title_contains: str | None) -> dict[str, Any]:
    if novel_id:
        return _fetch_json(base_url, f"/api/v2/novels/{novel_id}")
    novels = _fetch_json(base_url, "/api/v2/novels")
    matches = [
        item for item in novels
        if title_contains and title_contains in str(item.get("title") or "")
    ]
    if not matches:
        raise SystemExit(f"未找到标题包含“{title_contains}”的云端小说")
    if len(matches) > 1:
        ids = ", ".join(f"{item.get('id')}:{item.get('title')}" for item in matches)
        raise SystemExit(f"匹配到多本小说，请指定 --novel-id：{ids}")
    return _fetch_json(base_url, f"/api/v2/novels/{matches[0]['id']}")


def _issue_hit(issues: list[str], marker: str) -> bool:
    return any(marker in issue for issue in issues)


def audit_novel(novel: dict[str, Any]) -> dict[str, Any]:
    analysis = json.loads(novel.get("analysis") or "{}")
    protected_terms = api._analysis_protected_terms(analysis)
    name_map = api._analysis_name_map(analysis)
    chapters: list[dict[str, Any]] = []
    generated = 0
    risk_count = 0
    complaint_hits = {
        "non_core_detail": 0,
        "opening_beat": 0,
        "name_map": 0,
        "structure": 0,
    }

    for chapter in novel.get("chapters") or []:
        rewritten = chapter.get("rewritten") or ""
        if not rewritten.strip():
            continue
        generated += 1
        quality = api.score_rewrite_quality(
            rewritten,
            chapter.get("content") or "",
            protected_terms=protected_terms,
            name_map=name_map,
        )
        issues = list(quality.get("issues") or [])
        if quality.get("delivery_status") == "risk":
            risk_count += 1
        if _issue_hit(issues, "非核心细节照搬"):
            complaint_hits["non_core_detail"] += 1
        if _issue_hit(issues, "叙述骨架照搬"):
            complaint_hits["opening_beat"] += 1
        if _issue_hit(issues, "人名未按对照表"):
            complaint_hits["name_map"] += 1
        if _issue_hit(issues, "结构相似"):
            complaint_hits["structure"] += 1
        chapters.append({
            "idx": chapter.get("idx"),
            "title": chapter.get("title"),
            "cloud_quality_grade": chapter.get("quality_grade"),
            "cloud_quality_score": chapter.get("quality_score"),
            "local_delivery_status": quality.get("delivery_status"),
            "local_quality_score": quality.get("score"),
            "overlap4": quality.get("overlap4"),
            "structure_similarity": quality.get("structure_similarity"),
            "opening_beat_similarity": quality.get("opening_beat_similarity"),
            "issues": issues,
        })

    return {
        "novel": {
            "id": novel.get("id"),
            "title": novel.get("title"),
            "chapter_count": len(novel.get("chapters") or []),
            "generated_count": generated,
            "analysis_status": novel.get("analysis_status"),
        },
        "summary": {
            "risk_count": risk_count,
            "review_or_risk_count": sum(
                1 for item in chapters
                if item.get("local_delivery_status") in {"review", "risk"}
            ),
            "complaint_hits": complaint_hits,
        },
        "chapters": chapters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="只读抽检云端小说洗稿质量")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--novel-id")
    parser.add_argument("--title-contains", default="前夫再婚")
    parser.add_argument("--output", help="保存 JSON 报告到指定路径")
    parser.add_argument("--fail-on-risk", action="store_true", help="存在 risk 章节时返回非零")
    args = parser.parse_args()

    novel = _load_novel(args.base_url, args.novel_id, args.title_contains)
    report = audit_novel(novel)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    if args.fail_on_risk and report["summary"]["risk_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
