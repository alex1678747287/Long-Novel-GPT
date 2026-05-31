"""End-to-end test for the 洗稿 prompt against a live Doubao model.

Usage (inside container):
    docker exec -it long-novel-gpt python tests/test_xigao.py
Or locally with .env loaded:
    cd /app && python tests/test_xigao.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import API_SETTINGS, reload_user_config
reload_user_config()

from backend.v2 import registry
from backend.v2.api import _build_rewrite_messages, _extract_rewritten
from backend.v2.llm_client import stream_chat

# Add backend/ to path so we can reuse get_model_config_from_provider_model.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from backend_utils import get_model_config_from_provider_model  # noqa: E402,F811

SAMPLE = (
    "林轩缓缓睁开双眼，只觉得头脑中一阵剧痛，他强忍着不适，环顾四周，"
    "发现自己竟身处一座古朴的木屋中。屋内陈设简陋，只有一张破旧的木床"
    "和一张油渍斑斑的方桌。窗外传来阵阵蝉鸣，阳光透过破旧的窗纸，在地上"
    "投下斑驳的光影。他试图回想自己为何会出现在这里，但脑海中却一片混乱，"
    "只有些零碎的画面在闪现。"
)

PLOT = "主角林轩昏迷后醒来，发现身处陌生木屋，对自己为何在此感到困惑。"

SAMPLE2 = (
    "苏婉儿拎着青瓷药壶推门而入，见林轩已经醒来，眉头微微一蹙：\n"
    "\"你倒是命大，要不是我赶得及时，你这条小命怕是今日就交代在山涧里了。\"\n"
    "她将药壶放在桌上，倒出一碗黝黑的汤药，递到他面前。\n"
    "\"喝了，三日内不许下床。\"\n"
    "林轩接过药碗，看着碗里浮着几片不知名的草叶，皱了皱眉：\"姑娘，"
    "在下连你的名字都不知道，怎敢就这么喝下去？\"\n"
    "苏婉儿冷哼一声：\"信不过我？那你大可以倒掉。反正死的是你，不是我。\""
)

PLOT2 = (
    "苏婉儿端药给昏迷醒来的林轩，催他喝下并卧床三日；"
    "林轩对陌生人的药持有戒心；苏婉儿不耐烦，让他自己选择。"
)


def run_one(prompt_name: str, sample: str, plot: str, model: str = 'doubao/doubao-seed-2.0-pro'):
    print(f"\n{'='*70}\n>>> {prompt_name}\n{'='*70}")
    prompt = registry.get_prompt(f'builtin:{prompt_name}', reveal_builtin=True)
    if not prompt:
        raise RuntimeError(f'prompt not found: {prompt_name}')

    model_config = get_model_config_from_provider_model(model)
    messages = _build_rewrite_messages(prompt['content'], sample, plot)

    raw = ''
    for chunk in stream_chat(model_config, messages, temperature=0.9):
        raw = chunk['text']

    if not raw:
        print('[no response]')
        return None
    text = _extract_rewritten(raw)
    print(text)
    return text


def overlap_ratio(a: str, b: str, n: int = 4) -> float:
    """Fraction of length-n character windows from a that appear verbatim in b."""
    if len(a) < n:
        return 0.0
    hits = sum(1 for i in range(len(a) - n + 1) if a[i:i + n] in b)
    return hits / max(1, len(a) - n + 1)


def list_overlaps(a: str, b: str, n: int = 4):
    """Return unique n-grams from a that appear verbatim in b."""
    seen = set()
    for i in range(len(a) - n + 1):
        chunk = a[i:i + n]
        if chunk in b:
            seen.add(chunk)
    return sorted(seen)


def filtered_overlap(a: str, b: str, allow: set, n: int = 4):
    """Overlap rate excluding n-grams that are fully covered by allow-listed
    proper nouns (chars from 人名/地名/招式名/数字)."""
    allow_chars = set(''.join(allow))
    total = 0
    hits = 0
    legit_hits = 0
    for i in range(len(a) - n + 1):
        chunk = a[i:i + n]
        total += 1
        if chunk in b:
            hits += 1
            # treat as a "legit" hit (excluded from rate) if all chars are allow-listed
            if all(c in allow_chars or c in '，。：；！？、 "”“' for c in chunk):
                legit_hits += 1
    raw = hits / max(1, total)
    real = (hits - legit_hits) / max(1, total)
    return raw, real, hits, legit_hits


if __name__ == '__main__':
    # 允许保留的专有名词集合（每个样本一份）
    allows = {
        '场景描写': {'林轩'},
        '对话场景': {'苏婉儿', '林轩', '山涧', '青瓷药壶', '三日'},
    }
    for label, sample, plot in [
        ('场景描写', SAMPLE, PLOT),
        ('对话场景', SAMPLE2, PLOT2),
    ]:
        print(f"\n##### {label} #####")
        print('【原文】')
        print(sample)
        out = run_one('洗稿', sample, plot)
        if out:
            raw = overlap_ratio(out, sample, n=4)
            r_raw, r_real, hits, legit = filtered_overlap(out, sample, allows[label], n=4)
            shared = list_overlaps(out, sample, n=4)
            print(f"\n[原始 4-gram 重合率: {raw:.1%}]")
            print(f"[去除专有名词后真实重合率: {r_real:.1%}]  (命中 {hits}, 其中专有名词 {legit})")
            print(f"[具体重合片段]: {' | '.join(shared)}")
