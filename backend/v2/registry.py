"""Unified model + prompt registry.

All models are treated as OpenAI-compatible endpoints. This lets us drop the
provider-specific routing in legacy llm_api/ and rely on a single client.

Persisted to data/v2_config.json so changes survive restarts. The file is
written under the same data/ volume that already gets mounted into the
container, so no Docker changes are needed.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

# Locate the project root by walking up until we find a 'prompts/' dir. This
# handles both the dev layout (repo/backend/v2/) and the container layout
# (/app/v2/) without hard-coding parent counts.
def _find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / 'prompts').exists():
            return parent
    return here.parents[1]


ROOT = _find_root()
DATA_DIR = ROOT / 'data'
CONFIG_PATH = DATA_DIR / 'v2_config.json'
PROMPTS_DIR = Path(__file__).resolve().parent / 'builtin_prompts'
CUSTOM_PROMPTS_DIR = DATA_DIR / 'prompts'

_lock = threading.Lock()
DEFAULT_CHAPTER_SIZE = 2200
ONE_MILLION_CONTEXT_CHAPTER_SIZE = 3000
MID_CONTEXT_CHAPTER_SIZE = 2200

# Provider presets the UI surfaces as "one-click add". User still has to paste
# their API key, but base_url + the typical default model are pre-filled.
PROVIDER_PRESETS: list[dict[str, str]] = [
    {
        'id': 'doubao',
        'label': '豆包 (火山方舟)',
        'base_url': 'https://ark.cn-beijing.volces.com/api/v3',
        'default_model': 'doubao-seed-2-0-pro-260215',
        'docs': 'https://www.volcengine.com/docs/82379',
    },
    {
        'id': 'zhipuai',
        'label': '智谱 GLM',
        'base_url': 'https://open.bigmodel.cn/api/paas/v4',
        'default_model': 'glm-4-air',
        'docs': 'https://open.bigmodel.cn/dev/api',
    },
    {
        'id': 'deepseek',
        'label': 'DeepSeek',
        'base_url': 'https://api.deepseek.com/v1',
        'default_model': 'deepseek-v4-pro',
        'docs': 'https://platform.deepseek.com/api-docs/',
        'models': [
            {'id': 'deepseek-v4-pro', 'label': 'DeepSeek V4 Pro（1M 上下文）'},
            {'id': 'deepseek-chat', 'label': 'DeepSeek Chat'},
        ],
    },
    {
        'id': 'moonshot',
        'label': 'Kimi (Moonshot)',
        'base_url': 'https://api.moonshot.cn/v1',
        'default_model': 'moonshot-v1-32k',
        'docs': 'https://platform.moonshot.cn/docs',
    },
    {
        'id': 'qwen',
        'label': '通义千问',
        'base_url': 'https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
        'default_model': 'qwen3.7-max',
        'docs': 'https://help.aliyun.com/zh/model-studio/codex-token-plan',
        'models': [
            {'id': 'qwen3.7-max', 'label': 'qwen3.7-max（长文稳定，推荐）'},
            {'id': 'qwen-plus', 'label': 'qwen-plus'},
        ],
    },
    {
        'id': 'openai',
        'label': 'OpenAI',
        'base_url': 'https://api.openai.com/v1',
        'default_model': 'gpt-4o-mini',
        'docs': 'https://platform.openai.com/docs',
    },
    {
        'id': 'openrouter',
        'label': 'OpenRouter (聚合)',
        'base_url': 'https://openrouter.ai/api/v1',
        'default_model': 'anthropic/claude-sonnet-4',
        'docs': 'https://openrouter.ai/docs',
    },
    {
        'id': 'apimart',
        'label': 'APIMart (聚合)',
        'base_url': 'https://api.apimart.ai/v1',
        'default_model': 'claude-opus-4-7',
        'docs': 'https://docs.apimart.ai/cn',
        # 候选模型列表 — 前端把 '模型 ID' 输入框改成下拉，但仍允许手填覆盖
        # （当 APIMart 上线新模型时只在这里追加，不必改动 UI）。
        'models': [
            {'id': 'claude-opus-4-7', 'label': 'Claude 4.7 Opus（写作旗舰，推荐）'},
            {'id': 'claude-opus-4-6', 'label': 'Claude 4.6 Opus'},
            {'id': 'claude-sonnet-4-7', 'label': 'Claude 4.7 Sonnet（性价比）'},
            {'id': 'claude-sonnet-4-6', 'label': 'Claude 4.6 Sonnet'},
            {'id': 'claude-haiku-4-5', 'label': 'Claude 4.5 Haiku（极速）'},
            {'id': 'deepseek-v4-pro', 'label': 'DeepSeek V4 Pro（1M 上下文）'},
            {'id': 'gpt-5.5', 'label': 'GPT-5.5'},
            {'id': 'gpt-5', 'label': 'GPT-5'},
            {'id': 'gpt-5-mini', 'label': 'GPT-5 Mini'},
            {'id': 'gpt-4o', 'label': 'GPT-4o'},
            {'id': 'gpt-4o-mini', 'label': 'GPT-4o Mini'},
            {'id': 'o3', 'label': 'OpenAI o3'},
            {'id': 'o3-mini', 'label': 'OpenAI o3 Mini'},
            {'id': 'gemini-2.5-pro', 'label': 'Gemini 2.5 Pro'},
            {'id': 'gemini-2.5-flash', 'label': 'Gemini 2.5 Flash'},
        ],
    },
    {
        'id': 'custom',
        'label': '自定义 (OpenAI 兼容)',
        'base_url': '',
        'default_model': '',
        'docs': '',
    },
]


import base64

# Light obfuscation of api_key on disk. Plain base64 with a marker prefix —
# this is NOT real encryption; the goal is just to keep API keys out of
# casual config-file inspection (screenshots / paste in chat / etc.). A
# determined attacker with file access can still recover them.
_OBFS_PREFIX = 'oxg1:'


def _obfuscate(s: str) -> str:
    if not s or s.startswith(_OBFS_PREFIX):
        return s
    return _OBFS_PREFIX + base64.urlsafe_b64encode(s.encode('utf-8')).decode('ascii')


def _deobfuscate(s: str) -> str:
    if not s or not s.startswith(_OBFS_PREFIX):
        return s  # legacy plaintext records continue to work
    try:
        return base64.urlsafe_b64decode(s[len(_OBFS_PREFIX):].encode('ascii')).decode('utf-8')
    except Exception:
        return s


def _load() -> dict:
    if not CONFIG_PATH.exists():
        return {'models': [], 'active_model_id': None, 'max_concurrency': 5}
    try:
        with CONFIG_PATH.open('r', encoding='utf-8') as f:
            data = json.load(f) or {}
    except Exception:
        return {'models': [], 'active_model_id': None, 'max_concurrency': 5}
    # In-memory we always work with plaintext keys.
    for m in data.get('models', []) or []:
        if 'api_key' in m:
            m['api_key'] = _deobfuscate(m['api_key'])
    return data


def _save(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # On disk we always store obfuscated keys.
    persisted = {**data, 'models': [
        {**m, 'api_key': _obfuscate(m.get('api_key', ''))}
        for m in data.get('models', []) or []
    ]}
    with CONFIG_PATH.open('w', encoding='utf-8') as f:
        json.dump(persisted, f, ensure_ascii=False, indent=2)
    try:
        CONFIG_PATH.chmod(0o600)
    except OSError:
        pass


# ---- Models ----

def list_models() -> list[dict[str, Any]]:
    data = _load()
    return data.get('models', [])


def get_active_model() -> dict[str, Any] | None:
    data = _load()
    aid = data.get('active_model_id')
    for m in data.get('models', []):
        if m['id'] == aid:
            return m
    return data['models'][0] if data.get('models') else None


def get_model(model_id: str) -> dict[str, Any] | None:
    for m in list_models():
        if m['id'] == model_id:
            return m
    return None


def upsert_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Create or update a model.

    On edits, an empty api_key means "keep the existing secret" so users can
    adjust model IDs, temperature, or notes without re-pasting credentials.
    """
    required = ['name', 'base_url', 'model']
    for k in required:
        if not payload.get(k):
            raise ValueError(f'missing field: {k}')

    with _lock:
        data = _load()
        models = data.setdefault('models', [])
        model_id = payload.get('id') or str(uuid.uuid4())
        existing = next((m for m in models if m['id'] == model_id), None)
        api_key = (payload.get('api_key') or '').strip()
        if not api_key and existing:
            api_key = existing.get('api_key', '')
        if not api_key:
            raise ValueError('missing field: api_key')
        record = {
            'id': model_id,
            'name': payload['name'].strip(),
            'preset_id': payload.get('preset_id', 'custom'),
            'base_url': payload['base_url'].strip().rstrip('/'),
            'api_key': api_key,
            'model': payload['model'].strip(),
            'temperature': payload.get('temperature', 0.7),
            'max_tokens': payload.get('max_tokens', 16384),
            'note': payload.get('note', ''),
        }

        for i, m in enumerate(models):
            if m['id'] == model_id:
                models[i] = record
                break
        else:
            models.append(record)

        if not data.get('active_model_id'):
            data['active_model_id'] = model_id

        _save(data)
        return record


def delete_model(model_id: str) -> None:
    with _lock:
        data = _load()
        data['models'] = [m for m in data.get('models', []) if m['id'] != model_id]
        if data.get('active_model_id') == model_id:
            data['active_model_id'] = data['models'][0]['id'] if data['models'] else None
        _save(data)


def set_active_model(model_id: str) -> None:
    with _lock:
        data = _load()
        if not any(m['id'] == model_id for m in data.get('models', [])):
            raise ValueError(f'model not found: {model_id}')
        data['active_model_id'] = model_id
        _save(data)


# ---- Model capability helpers ----

def _intish(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def is_large_context_model(model: dict[str, Any] | None) -> bool:
    """Best-effort 1M-context detection.

    User-provided routers do not expose a normalized context-window field, so
    we support explicit config first and fall back to common 1M model-name
    patterns used in this workbench.
    """
    if not model:
        return False

    explicit_context = max(
        _intish(model.get('max_context_tokens')),
        _intish(model.get('context_window')),
        _intish(model.get('context_window_tokens')),
        _intish(model.get('context_length')),
    )
    if explicit_context >= 1_000_000:
        return True

    name = (model.get('model') or model.get('name') or '').strip().lower().replace('_', '-')
    compact = name.replace('-', '').replace(' ', '')
    large_patterns = [
        '1m',
        '1-million',
        '1000k',
        'claude-opus-4-7',
        'claude-opus-4-6',
        'claude-sonnet-4-7',
        'claude-sonnet-4-6',
        'gpt-5.5',
        'deepseek-v4',
        'deepseek-v4-pro',
        'deepseek-v4pro',
    ]
    if any(pattern in name for pattern in large_patterns):
        return True
    return 'deepseekv4pro' in compact or 'gpt55' in compact


def is_mid_context_model(model: dict[str, Any] | None) -> bool:
    if not model:
        return False
    explicit_context = max(
        _intish(model.get('max_context_tokens')),
        _intish(model.get('context_window')),
        _intish(model.get('context_window_tokens')),
        _intish(model.get('context_length')),
    )
    if 256_000 <= explicit_context < 1_000_000:
        return True

    name = (model.get('model') or model.get('name') or '').strip().lower().replace('_', '-')
    return '256k' in name or '512k' in name


def recommended_chapter_size(
    model: dict[str, Any] | None = None,
    configured: int | None = None,
) -> int:
    base = configured or get_system_params().get('max_chapter_size') or DEFAULT_CHAPTER_SIZE
    try:
        base = int(base)
    except (TypeError, ValueError):
        base = DEFAULT_CHAPTER_SIZE
    if is_large_context_model(model):
        return ONE_MILLION_CONTEXT_CHAPTER_SIZE
    if is_mid_context_model(model):
        return MID_CONTEXT_CHAPTER_SIZE
    return min(base, DEFAULT_CHAPTER_SIZE)


# ---- System parameters ----

def get_system_params() -> dict[str, Any]:
    data = _load()
    return {
        'max_concurrency': data.get('max_concurrency', 5),
        'max_chapter_size': data.get('max_chapter_size', DEFAULT_CHAPTER_SIZE),
    }


def set_system_params(payload: dict[str, Any]) -> None:
    with _lock:
        data = _load()
        if 'max_concurrency' in payload:
            data['max_concurrency'] = int(payload['max_concurrency'])
        if 'max_chapter_size' in payload:
            data['max_chapter_size'] = int(payload['max_chapter_size'])
        _save(data)


# ---- Prompts ----

# Built-in prompts live with the v2 backend. The files keep their historical
# names, while the API exposes customer-facing task names.
BUILTIN_PROMPTS = [
    {'id': '洗稿', 'name': '洗稿', 'file': '洗稿.txt', 'task': 'rewrite'},
    {'id': '转剧本', 'name': '转剧本', 'file': '转剧本.txt', 'task': 'script'},
]
BUILTIN_PROMPT_IDS = [p['id'] for p in BUILTIN_PROMPTS]
BUILTIN_PROMPT_ALIASES = {
    '精修': '洗稿',
    '洗稿剧本版': '转剧本',
    '精修剧本版': '转剧本',
}


def canonical_prompt_name(name: str) -> str:
    return BUILTIN_PROMPT_ALIASES.get(name, name)


def canonical_prompt_id(prompt_id: str | None) -> str | None:
    if not prompt_id or not prompt_id.startswith('builtin:'):
        return prompt_id
    name = prompt_id.split(':', 1)[1]
    return f'builtin:{canonical_prompt_name(name)}'


def _read_prompt_file(path: Path) -> str:
    """Strip // comments and the leading 'user:' marker."""
    if not path.exists():
        return ''
    raw = path.read_text(encoding='utf-8')
    lines = [ln for ln in raw.split('\n') if not ln.startswith('//')]
    content = '\n'.join(lines).strip()
    if content.startswith('user:\n'):
        content = content[len('user:\n'):]
    return content.strip()


def list_prompts(reveal_builtin: bool = False) -> list[dict[str, Any]]:
    """Return the prompt catalog.

    By default the *content* of built-in prompts is redacted (returned as an
    empty string) so the engineering team's curated洗稿 prompts are not
    leaked through the UI. Pass reveal_builtin=True for internal call sites
    (e.g. /api/rewrite) that genuinely need the text to send to the model.
    """
    out = []
    for spec in BUILTIN_PROMPTS:
        content = _read_prompt_file(PROMPTS_DIR / spec['file'])
        if content:
            out.append({
                'id': f"builtin:{spec['id']}",
                'name': spec['name'],
                'content': content if reveal_builtin else '',
                'is_builtin': True,
                'task': spec['task'],
            })

    if CUSTOM_PROMPTS_DIR.exists():
        for p in sorted(CUSTOM_PROMPTS_DIR.glob('*.json')):
            try:
                meta = json.loads(p.read_text(encoding='utf-8'))
                out.append({
                    'id': f'custom:{p.stem}',
                    'name': meta.get('name', p.stem),
                    'content': meta.get('content', ''),
                    'is_builtin': False,
                    'task': meta.get('task') or 'rewrite',
                })
            except Exception:
                continue
    return out


def get_prompt(prompt_id: str, reveal_builtin: bool = False) -> dict[str, Any] | None:
    prompt_id = canonical_prompt_id(prompt_id)
    for p in list_prompts(reveal_builtin=reveal_builtin):
        if p['id'] == prompt_id:
            return p
    return None


def _safe_custom_prompt_stem(stem: str) -> str:
    stem = (stem or '').strip()
    if not stem or not all(c.isalnum() or c in '_-' for c in stem):
        raise ValueError('invalid custom prompt id')
    return stem


def _custom_prompt_path(stem: str) -> Path:
    safe_stem = _safe_custom_prompt_stem(stem)
    base = CUSTOM_PROMPTS_DIR.resolve()
    path = (CUSTOM_PROMPTS_DIR / f'{safe_stem}.json').resolve()
    if base not in path.parents:
        raise ValueError('invalid custom prompt id')
    return path


def upsert_prompt(payload: dict[str, Any]) -> dict[str, Any]:
    name = (payload.get('name') or '').strip()
    content = payload.get('content') or ''
    task = payload.get('task') or 'rewrite'
    if not name:
        raise ValueError('name is required')
    if not content.strip():
        raise ValueError('content is required')
    if task not in {'rewrite', 'script'}:
        raise ValueError('task must be rewrite or script')

    CUSTOM_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    # Use the name (sanitized) as the file stem so it's easy to inspect.
    prompt_id = payload.get('id') or ''
    existing_stem = prompt_id.split(':', 1)[1] if prompt_id.startswith('custom:') else ''
    stem = _safe_custom_prompt_stem(existing_stem) if existing_stem else (
        ''.join(c for c in name if c.isalnum() or c in '_-') or str(uuid.uuid4())[:8]
    )
    path = _custom_prompt_path(stem)
    with _lock:
        if existing_stem and not path.exists():
            raise ValueError(f'prompt not found: {prompt_id}')
        path.write_text(
            json.dumps({'name': name, 'content': content, 'task': task}, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
    return {'id': f'custom:{stem}', 'name': name, 'content': content, 'is_builtin': False, 'task': task}


def delete_prompt(prompt_id: str) -> None:
    if prompt_id.startswith('builtin:'):
        raise ValueError('cannot delete builtin prompt')
    if not prompt_id.startswith('custom:'):
        raise ValueError('invalid custom prompt id')
    stem = prompt_id.split(':', 1)[1]
    path = _custom_prompt_path(stem)
    if path.exists():
        path.unlink()
