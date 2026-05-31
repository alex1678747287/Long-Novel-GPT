"""Whole-novel pre-analysis.

Before any chapter gets rewritten, we run the full novel through the model
once to produce a global mapping (original name -> new name + things to
keep verbatim). That mapping then gets injected into every per-chapter
rewrite, so the same character is renamed consistently across all 60
chapters of a short-drama script.

The analyzer is intentionally a separate concern from the rewrite prompt —
this is structural information extraction, not generation.
"""
from __future__ import annotations

import json
import re
from typing import Any

from . import registry
from .llm_client import one_shot


# Prompt is engineered for strict JSON output. We rely on the SDK fix in
# llm_client.one_shot (stream-then-stitch) so even SSE-only providers like
# APIMart work fine.
ANALYZE_PROMPT = """你是一位专业网文洗稿编辑助手。我会给你一份小说（可能不完整，只是前面若干章/采样片段），请你**扫描全文**，输出一份用于"洗稿全本一致性"的元数据。

═════════ 任务 ═════════

为这本小说生成一张"洗稿对照表"，让后续逐章洗稿时所有章节都遵守，避免人物/地点/术语漂移。

═════════ 改名规则 ═════════

1. **所有原创主要人物名都要换**（包括男主、女主、反派、配角、孩子小名等）。
2. 换名策略：换姓+换名，保留三字结构和性格质感（如"林轩→陆延"、"苏婉儿→沈青柠"、"赵元成→周元成"）。
3. **反派家族整族换姓**（如"赵家"→"周家"）。
4. **原文中无名角色（"工长""老公""女儿"等）必须补一个三字真名**。
5. **保留**：
   - 真实公众人物（鲁迅、武则天等）
   - 强设定的角色/招式（如"系统"、"道一刀"）
   - 路人小角色（"小张""王老师""那大爷"）

═════════ 输出格式 ═════════

**严格输出 JSON 对象**（不要 markdown、不要解释、不要思考过程），结构如下：

```json
{
  "name_map": {
    "原名1": "新名1",
    "原名2": "新名2"
  },
  "place_map": {
    "原地名1": "新地名1"
  },
  "keep_terms": ["羊脂玉佩", "青瓷药壶"],
  "character_profiles": [
    {
      "original": "林轩",
      "rewrite": "陆延",
      "role": "男主",
      "traits": "沉稳、有保护欲",
      "must_keep": "护住女主和孩子的动机不能改"
    }
  ],
  "worldview": "架空古代侯府与王府权谋",
  "plot_lines": ["女主带嫁妆周旋侯府", "男主暗中护她"],
  "do_not_change": ["嫁妆是核心筹码", "男女主关系从互疑到互信"],
  "relationship_rules": ["女主与继子女称谓保持稳定"],
  "term_rules": ["嫁妆账册、玉佩等关键道具规则不能改"],
  "style_note": "古风武侠第三人称",
  "notes": "一两句话说明本书背景题材，方便后续洗稿对齐基调"
}
```

要求：
- name_map 要**尽量覆盖你在本片段里看到的每一个有名有姓的角色**（主角、配角、反派、孩子小名都算），不要只挑主角；哪怕某角色只出现一两次、或在片段靠后才登场，只要有名字就要给它一个新名映射。这是为了防止逐章洗稿时后段才出场的角色被各章随意改成不同的名字。
- 本次给你的可能只是全书的一部分章节或采样片段；请只对你实际看到的人名做映射，不要臆造没出现过的人物，但凡看到的都不要遗漏。
- name_map 的 key 用原文中出现的全名形式（"林轩"而不是"林"）
- place_map 只列**可换的虚构地名**，真实地名（上京、江市）保留不放进表
- keep_terms 列出**关键道具/招式名/标志台词关键词**——这些洗稿时必须保留
- character_profiles 用于人物表；每个角色写清原名、新名、身份、性格、禁改动机
- worldview 写清时代背景/世界观/权力结构
- plot_lines 写主线情节线，只保留不能乱改的主线
- do_not_change 写禁改点：身份、关系、关键道具、关键因果、爽点反转
- relationship_rules 写称谓、亲缘、敌友、上下级等必须稳定的关系规则
- term_rules 写关键道具/术语/资产规则
- style_note 不超过 12 字

═════════ 原稿样本 ═════════

{novel_text}
"""


# Hard cap on chars sent to the model. We sample chapter beginnings to fit
# within reasonable context windows (and cost) without losing major
# characters. Tuned for ~30k Chinese chars which fits comfortably even in
# 32k-token models.
MAX_ANALYSIS_CHARS = 30000
LARGE_CONTEXT_ANALYSIS_CHARS = 100000
# Upper bound on analysis model calls per novel. Non-large-context models pack
# chapters into budget-sized windows; this caps how many windows we run so the
# one-time pre-rewrite analysis bill stays bounded even for very long books.
MAX_ANALYSIS_PASSES = 5


def _sample_text(
    chapters: list[dict],
    max_chars: int = MAX_ANALYSIS_CHARS,
    full_body: bool = False,
) -> str:
    """Stitch chapters into one analysis-friendly blob.

    For each chapter we keep its title plus the first ~800 chars (where new
    characters typically debut) and the last ~200 chars (which often hint
    at upcoming arcs). If the total chapter count is small enough that we
    can fit everything, we do.
    """
    parts: list[str] = []
    budget = max_chars
    for c in chapters:
        title = c.get('title') or ''
        body = (c.get('content') or '').strip()
        if not body:
            continue
        if full_body:
            piece = body
        elif len(body) <= 1600:
            piece = body
        else:
            piece = body[:1200] + '\n……（中段省略）……\n' + body[-300:]
        block = f'【{title}】\n{piece}\n'
        if budget - len(block) < 0:
            # Truncate the last block so we end cleanly.
            parts.append(block[:max(0, budget)])
            break
        parts.append(block)
        budget -= len(block)
    # 每个 block 自带结尾换行，用 '' 拼接，避免额外空行并让总长精确不超 max_chars。
    return ''.join(parts)


def _sample_text_for_model(model_cfg: dict, chapters: list[dict]) -> str:
    if registry.is_large_context_model(model_cfg):
        return _sample_text(chapters, max_chars=LARGE_CONTEXT_ANALYSIS_CHARS, full_body=True)
    return _sample_text(chapters)


_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            cleaned = {
                str(k): (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
                for k, v in item.items()
                if str(k).strip()
            }
            if cleaned:
                out.append(cleaned)
    return out


def _merge_unique_strings(*values: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in value:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                merged.append(text)
    return merged


def _merge_unique_dicts(*values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        for item in value:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                merged.append(item)
    return merged


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Pull the JSON object out of a model response. Tolerates leading
    'thinking' text and markdown fences."""
    # First try the whole thing.
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Strip ```json fences.
    body = raw.strip()
    if body.startswith('```'):
        body = re.sub(r'^```[a-zA-Z]*\n', '', body)
        body = re.sub(r'\n```\s*$', '', body)
        try:
            return json.loads(body)
        except Exception:
            pass
    # Fall back to grabbing the first top-level object.
    m = _JSON_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _merge_results(acc: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Merge a new partial analysis into the accumulator. Keys already mapped
    win — we don't let later chapters reassign main characters."""
    name_map = dict(acc.get('name_map') or {})
    for k, v in (new.get('name_map') or {}).items():
        if k not in name_map:
            name_map[k] = v
    place_map = dict(acc.get('place_map') or {})
    for k, v in (new.get('place_map') or {}).items():
        if k not in place_map:
            place_map[k] = v
    keep_terms = _merge_unique_strings(
        _string_list(acc.get('keep_terms')),
        _string_list(new.get('keep_terms')),
    )
    return {
        'name_map': name_map,
        'place_map': place_map,
        'keep_terms': keep_terms,
        'character_profiles': _merge_unique_dicts(
            _dict_list(acc.get('character_profiles')),
            _dict_list(new.get('character_profiles')),
        ),
        'worldview': acc.get('worldview') or new.get('worldview') or '',
        'plot_lines': _merge_unique_strings(
            _string_list(acc.get('plot_lines')),
            _string_list(new.get('plot_lines')),
        ),
        'do_not_change': _merge_unique_strings(
            _string_list(acc.get('do_not_change')),
            _string_list(new.get('do_not_change')),
        ),
        'relationship_rules': _merge_unique_strings(
            _string_list(acc.get('relationship_rules')),
            _string_list(new.get('relationship_rules')),
        ),
        'term_rules': _merge_unique_strings(
            _string_list(acc.get('term_rules')),
            _string_list(new.get('term_rules')),
        ),
        'style_note': acc.get('style_note') or new.get('style_note') or '',
        'notes': acc.get('notes') or new.get('notes') or '',
    }


def _run_one_pass(model_cfg: dict, sample: str) -> dict[str, Any]:
    messages = [
        {'role': 'system', 'content': '你是一个文本结构化工具。严格输出 JSON。'},
        {'role': 'user', 'content': ANALYZE_PROMPT.replace('{novel_text}', sample)},
    ]
    raw = one_shot(model_cfg, messages, temperature=0.2)
    parsed = _extract_json(raw)
    if parsed is None:
        raise ValueError(f'analyzer returned unparseable JSON. Head: {raw[:300]}')
    return {
        'name_map': dict(parsed.get('name_map') or {}),
        'place_map': dict(parsed.get('place_map') or {}),
        'keep_terms': _string_list(parsed.get('keep_terms')),
        'character_profiles': _dict_list(parsed.get('character_profiles')),
        'worldview': str(parsed.get('worldview') or ''),
        'plot_lines': _string_list(parsed.get('plot_lines')),
        'do_not_change': _string_list(parsed.get('do_not_change')),
        'relationship_rules': _string_list(parsed.get('relationship_rules')),
        'term_rules': _string_list(parsed.get('term_rules')),
        'style_note': str(parsed.get('style_note') or ''),
        'notes': str(parsed.get('notes') or ''),
    }


def _estimate_chapter_sample_chars(chapter: dict) -> int:
    """How many chars _sample_text will keep for this chapter (head+tail)."""
    body = (chapter.get('content') or '').strip()
    if not body:
        return 0
    title = chapter.get('title') or ''
    sampled = len(body) if len(body) <= 1600 else 1500
    return sampled + len(title) + 4


def _plan_analysis_windows(model_cfg: dict, chapters: list[dict]) -> list[list[dict]]:
    """Pick chapter windows so EVERY chapter contributes its debuting names.

    Large-context models read the whole book in one pass. Other models get
    multiple budget-sized windows — this is the key fix for cross-chapter name
    drift: previously non-large-context models ran a single 30k-truncated pass,
    so characters who debut in later chapters never entered name_map and each
    chapter then renamed them independently. Now windows cover the whole book
    and _merge_results (first-wins) accumulates names across passes. Capped at
    MAX_ANALYSIS_PASSES to bound the one-time analysis bill."""
    usable = [c for c in chapters if (c.get('content') or '').strip()]
    if not usable:
        return [chapters] if chapters else []
    if registry.is_large_context_model(model_cfg):
        return [usable]

    windows: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for c in usable:
        size = _estimate_chapter_sample_chars(c)
        if current and current_chars + size > MAX_ANALYSIS_CHARS:
            windows.append(current)
            current = []
            current_chars = 0
        current.append(c)
        current_chars += size
    if current:
        windows.append(current)
    if not windows:
        return [usable]
    if len(windows) <= MAX_ANALYSIS_PASSES:
        return windows
    # Too many budget windows for the pass cap: regroup into MAX_ANALYSIS_PASSES
    # contiguous slices so no chapter is skipped (a slice may still be truncated
    # by _sample_text's per-call cap, but every chapter appears in some pass).
    n = len(usable)
    step = (n + MAX_ANALYSIS_PASSES - 1) // MAX_ANALYSIS_PASSES
    return [usable[i:i + step] for i in range(0, n, step)]


def analyze_novel(
    model_cfg: dict,
    chapters: list[dict],
) -> dict[str, Any]:
    """Run the analyzer. We run one or more passes over chapter windows so
    late-arriving characters still get into the name_map (see
    _plan_analysis_windows). Each pass costs one model call; capped at
    MAX_ANALYSIS_PASSES to keep token bills reasonable."""
    if not chapters:
        return {
            'name_map': {},
            'place_map': {},
            'keep_terms': [],
            'character_profiles': [],
            'worldview': '',
            'plot_lines': [],
            'do_not_change': [],
            'relationship_rules': [],
            'term_rules': [],
            'style_note': '',
            'notes': '',
        }

    windows = _plan_analysis_windows(model_cfg, chapters)

    acc: dict[str, Any] = {}
    for w in windows:
        sample = _sample_text_for_model(model_cfg, w)
        if not sample.strip():
            continue
        try:
            partial = _run_one_pass(model_cfg, sample)
        except Exception:
            # If a single window fails we still keep what other windows
            # produced rather than nuking the whole analysis.
            continue
        acc = _merge_results(acc, partial) if acc else partial

    if not acc:
        raise ValueError('all analyzer passes failed to produce JSON')
    return acc


def format_for_rewrite_prompt(analysis: dict[str, Any]) -> str:
    """Render the analysis into a short, model-friendly block that gets
    prepended to every per-chapter rewrite. Returns empty string if there
    is nothing useful to inject."""
    if not analysis:
        return ''
    name_map = analysis.get('name_map') or {}
    place_map = analysis.get('place_map') or {}
    keep_terms = _string_list(analysis.get('keep_terms'))
    character_profiles = _dict_list(analysis.get('character_profiles'))
    worldview = str(analysis.get('worldview') or '').strip()
    plot_lines = _string_list(analysis.get('plot_lines'))
    do_not_change = _string_list(analysis.get('do_not_change'))
    relationship_rules = _string_list(analysis.get('relationship_rules'))
    term_rules = _string_list(analysis.get('term_rules'))
    style = analysis.get('style_note') or ''
    notes = analysis.get('notes') or ''
    if not (
        name_map
        or place_map
        or keep_terms
        or character_profiles
        or worldview
        or plot_lines
        or do_not_change
        or relationship_rules
        or term_rules
        or style
        or notes
    ):
        return ''

    lines: list[str] = ['【本书洗稿对照（必须严格遵守，全书所有章节一致）】']
    if style:
        lines.append(f'· 题材/视角：{style}')
    if notes:
        lines.append(f'· 背景：{notes}')
    if character_profiles:
        lines.append('· 人物表：')
        for item in character_profiles[:24]:
            original = str(item.get('original') or '').strip()
            rewrite = str(item.get('rewrite') or '').strip()
            role = str(item.get('role') or '').strip()
            traits = str(item.get('traits') or '').strip()
            must_keep = str(item.get('must_keep') or '').strip()
            head = ' → '.join(part for part in [original, rewrite] if part)
            detail = '；'.join(part for part in [role, traits, must_keep] if part)
            if head and detail:
                lines.append(f'    {head}：{detail}')
            elif head:
                lines.append(f'    {head}')
            elif detail:
                lines.append(f'    {detail}')
    if worldview:
        lines.append(f'· 世界观/时代背景：{worldview}')
    if plot_lines:
        lines.append('· 主线情节线：' + '；'.join(plot_lines[:12]))
    if do_not_change:
        lines.append('· 禁改点：' + '；'.join(do_not_change[:16]))
    if relationship_rules:
        lines.append('· 关系/称谓规则：' + '；'.join(relationship_rules[:16]))
    if term_rules:
        lines.append('· 术语/道具规则：' + '；'.join(term_rules[:16]))
    if name_map:
        lines.append('· 人名映射（出现任何一处原名都要替换为新名）：')
        for orig, new in name_map.items():
            lines.append(f'    {orig} → {new}')
    if place_map:
        lines.append('· 地名映射：')
        for orig, new in place_map.items():
            lines.append(f'    {orig} → {new}')
    if keep_terms:
        lines.append('· 必须原样保留的关键词：' + '、'.join(keep_terms))
    lines.append('')
    lines.append(
        '上表是全本统一的对照——本章洗稿必须严格使用其中的新名，不要自创新名。'
        '如果本章出现了对照表里没列出的新角色，按同样的改名规则给它命名，并在全书后续章节保持这个名字一致，绝不要让同一个角色在不同章里出现不同的名字。'
    )
    return '\n'.join(lines)
