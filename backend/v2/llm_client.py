"""Single OpenAI-compatible client used for every model. Streams chunks back."""
from __future__ import annotations

from typing import Generator

from openai import OpenAI


def _model_omits_temperature(model_name: str) -> bool:
    name = (model_name or '').strip().lower()
    return name.startswith('claude-') or '/claude-' in name


def _disable_thinking_extra_body(model_name: str) -> dict:
    """本应用的所有调用（洗稿、人物分析、总结、转剧本）都是"文本转换/抽取"任务，不是推理
    任务。推理(thinking)在这类任务上是有害的：思考 token 计入 completion 抢 max_tokens 额度→
    正文被截断，且更慢更贵、还会让混合模型无视篇幅/人名/不增删情节等硬约束（reasoning rigidity）。
    因此对"能显式关思考"的混合模型一律关掉思考。不同厂商参数不同（依据官方文档）：
      - DeepSeek V4(deepseek-v4-pro/flash、deepseek-chat) / 智谱 GLM-4.x：{"thinking":{"type":"disabled"}}
      - 通义千问 Qwen3 混合档（DashScope OpenAI 兼容）：{"enable_thinking": false}
    未识别的模型(豆包等)不加任何参数，保持原行为，避免误传未知字段被拒。"""
    name = (model_name or '').strip().lower()
    if 'deepseek' in name or 'glm' in name or 'chatglm' in name or 'zhipu' in name:
        return {'thinking': {'type': 'disabled'}}
    if 'qwen' in name or 'qwq' in name or 'tongyi' in name:
        return {'enable_thinking': False}
    return {}


def _chat_completion_kwargs(
    model_cfg: dict,
    messages: list[dict],
    temperature: float | None = None,
    disable_thinking: bool = True,
) -> dict:
    max_tokens = int(model_cfg.get('max_tokens') or 16384)
    kwargs = {
        'model': model_cfg['model'],
        'messages': messages,
        'stream': True,
        'stream_options': {'include_usage': True},
        'max_tokens': max_tokens,
    }
    if not _model_omits_temperature(model_cfg.get('model', '')):
        kwargs['temperature'] = temperature if temperature is not None else model_cfg.get('temperature', 0.7)
    # 仅对“转换类”调用(洗稿/转剧本)关思考；理解抽取类(人物分析)保留思考——抽取全本角色更全。
    if disable_thinking:
        extra_body = _disable_thinking_extra_body(model_cfg.get('model', ''))
        if extra_body:
            kwargs['extra_body'] = extra_body
    return kwargs


def stream_chat(model_cfg: dict, messages: list[dict], temperature: float | None = None, disable_thinking: bool = True) -> Generator[dict, None, dict]:
    """Yield {'delta': str, 'text': str} dicts. Final yield includes
    {'done': True, 'text': str, 'usage': {...}, 'finish_reason': str}."""
    client = OpenAI(api_key=model_cfg['api_key'], base_url=model_cfg['base_url'])
    # Default to 8192 because the legacy 4096 default was truncating long
    # rewrites. Most providers accept 8192+ for output; if a model rejects
    # the value the user can override it in the model card.
    # Default to 16384 so single-chapter rewrites of 4000-6000 字 originals
    # (which often expand 1.5x to 7000-9000 字) don't get truncated mid-paragraph.
    # Users can override per-model in the settings page.
    response = client.chat.completions.create(**_chat_completion_kwargs(model_cfg, messages, temperature, disable_thinking))

    text = ''
    usage = None
    finish_reason = None
    for chunk in response:
        if chunk.usage:
            usage = {
                'prompt_tokens': chunk.usage.prompt_tokens,
                'completion_tokens': chunk.usage.completion_tokens,
                'total_tokens': chunk.usage.total_tokens,
            }
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if getattr(choice, 'finish_reason', None):
            finish_reason = choice.finish_reason
        delta = choice.delta.content or ''
        if delta:
            text += delta
            yield {'delta': delta, 'text': text, 'done': False}

    final = {'done': True, 'text': text, 'usage': usage, 'finish_reason': finish_reason}
    yield final
    return final


def one_shot(model_cfg: dict, messages: list[dict], temperature: float | None = None,
             disable_thinking: bool = False) -> str:
    """Non-streaming convenience wrapper used by 人物分析 / /api/split / /api/test.

    Internally we *always* use streaming and stitch the chunks back together.
    The reason: some OpenAI-compatible aggregators (e.g. APIMart) only support
    streaming and return SSE text even when stream=False is requested. The
    OpenAI SDK then can't parse the response and ends up handing back the raw
    string. Streaming mode works on every provider we've tested, so we use
    that path uniformly.

    disable_thinking defaults to False here: 人物分析是"从长文里抽全角色"的理解任务，
    保留思考能显著提升角色召回(且输出是短 JSON，不会被推理挤截断)。洗稿走 stream_chat
    直连，默认关思考。
    """
    text = ''
    for chunk in stream_chat(model_cfg, messages, temperature=temperature, disable_thinking=disable_thinking):
        if chunk.get('done'):
            text = chunk.get('text', text)
            break
        text = chunk.get('text', text)
    return text
