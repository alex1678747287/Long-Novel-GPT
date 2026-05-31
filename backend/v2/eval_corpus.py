"""Internal evaluation corpus helpers.

The public workbench stays simple; this module is for importing customer/test
ZIPs that contain 原稿/精修 pairs and turning them into repeatable quality
baselines.
"""
from __future__ import annotations

import json
import re
import statistics
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from .registry import DATA_DIR


CORPUS_PATH = DATA_DIR / 'eval_corpus.json'


def _corpus_path() -> Path:
    return DATA_DIR / 'eval_corpus.json'


def _decode_text(data: bytes) -> tuple[str, str]:
    for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'gbk', 'big5'):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            pass
    return data.decode('utf-8', errors='replace'), 'utf-8-replace'


def _compact(text: str) -> str:
    return re.sub(r'\s+', '', text or '')


def _overlap_4gram(a: str, b: str) -> float:
    left = _compact(a)
    right = _compact(b)
    if len(left) < 4 or len(right) < 4:
        return 0.0
    a_grams = {left[i:i + 4] for i in range(len(left) - 3)}
    b_grams = {right[i:i + 4] for i in range(len(right) - 3)}
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))


def _safe_basename(name: str) -> str:
    return re.sub(r'[\x00-\x1f/\\:]', '_', Path(name).name)


def _file_key(name: str) -> tuple[str, int | None, int | None]:
    m = re.match(r'^(原稿|精修|改稿)(\d+)(?:[-_](\d+))?.*?\.txt$', name, re.I)
    if not m:
        return '其他', None, None
    return m.group(1), int(m.group(2)), int(m.group(3) or 0)


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 4) if values else None


def import_zip_bytes(data: bytes, persist: bool = True) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    with zipfile.ZipFile(BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = _safe_basename(info.filename)
            if not name.lower().endswith('.txt'):
                continue
            text, encoding = _decode_text(z.read(info))
            prefix, num, part = _file_key(name)
            files.append({
                'name': name,
                'prefix': prefix,
                'num': num,
                'part': part,
                'encoding': encoding,
                'chars': len(text),
                'nonspace_chars': len(_compact(text)),
                'text': text,
            })

    originals = {f['num']: f for f in files if f['prefix'] == '原稿' and f['num'] is not None}
    refined_by_num: dict[int, list[dict[str, Any]]] = {}
    for f in files:
        if f['prefix'] in {'精修', '改稿'} and f['num'] is not None:
            refined_by_num.setdefault(f['num'], []).append(f)

    pairs: list[dict[str, Any]] = []
    for num, original in sorted(originals.items()):
        refined_files = refined_by_num.get(num) or []
        if not refined_files:
            continue
        refined_files = sorted(refined_files, key=lambda f: (f['part'] or 0, f['name']))
        refined_text = '\n'.join(f['text'] for f in refined_files)
        source_len = len(_compact(original['text']))
        refined_len = len(_compact(refined_text))
        pairs.append({
            'num': num,
            'original_name': original['name'],
            'refined_names': [f['name'] for f in refined_files],
            'original_chars': source_len,
            'refined_chars': refined_len,
            'length_ratio': round(refined_len / source_len, 4) if source_len else 0,
            'overlap4': round(_overlap_4gram(refined_text, original['text']), 4),
        })

    ratios = [p['length_ratio'] for p in pairs if p['length_ratio']]
    overlaps = [p['overlap4'] for p in pairs]
    original_lengths = [f['nonspace_chars'] for f in originals.values()]
    refined_count = len({f['num'] for f in files if f['prefix'] in {'精修', '改稿'} and f['num'] is not None})
    summary: dict[str, Any] = {
        'version': 1,
        'imported_at': time.time(),
        'total_files': len(files),
        'original_count': len(originals),
        'refined_count': refined_count,
        'pair_count': len(pairs),
        'other_files': [f['name'] for f in files if f['prefix'] == '其他'],
        'original_chars': {
            'min': min(original_lengths) if original_lengths else 0,
            'max': max(original_lengths) if original_lengths else 0,
            'median': int(statistics.median(original_lengths)) if original_lengths else 0,
            'over_100k_count': sum(1 for n in original_lengths if n > 100_000),
        },
        'reference_quality': {
            'length_ratio_median': _median(ratios),
            'length_ratio_mean': round(statistics.mean(ratios), 4) if ratios else None,
            'overlap_median': _median(overlaps),
            'overlap_mean': round(statistics.mean(overlaps), 4) if overlaps else None,
            'overlap_under_15pct_count': sum(1 for n in overlaps if n <= 0.15),
            'overlap_under_30pct_count': sum(1 for n in overlaps if n <= 0.30),
        },
        'sample_pairs': pairs[:10],
    }
    if persist:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with _corpus_path().open('w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def load_summary() -> dict[str, Any]:
    path = _corpus_path()
    if not path.exists():
        return {'version': 1, 'pair_count': 0, 'total_files': 0}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)
