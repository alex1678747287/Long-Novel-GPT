import os
import json
import threading
from dotenv import dotenv_values


def _safe_env_log_value(value):
    return '<empty>' if value in ('', None) else '<set>'

print("Loading .env file...")
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    env_dict = dotenv_values(env_path)

    print("Environment variables to be loaded:")
    for key, value in env_dict.items():
        print(f"{key}={_safe_env_log_value(value)}")
    print("-" * 50)

    os.environ.update(env_dict)
    print(f"Loaded environment variables from: {env_path}")
else:
    print("Warning: .env file not found")


# User-overridable config persisted from UI. Lives under data/ which is
# mounted as a docker volume so settings survive container restarts.
USER_CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'data')
USER_CONFIG_PATH = os.path.join(USER_CONFIG_DIR, 'user_config.json')
_user_config_lock = threading.Lock()


def _load_user_config_file():
    if not os.path.exists(USER_CONFIG_PATH):
        return {}
    try:
        with open(USER_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f) or {}
    except Exception as e:
        print(f"Warning: failed to read {USER_CONFIG_PATH}: {e}")
        return {}


def _write_user_config_file(data):
    os.makedirs(USER_CONFIG_DIR, exist_ok=True)
    with open(USER_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(USER_CONFIG_PATH, 0o600)
    except OSError:
        pass


# Thread Configuration
MAX_THREAD_NUM = int(os.getenv('MAX_THREAD_NUM', 5))


MAX_NOVEL_SUMMARY_LENGTH = int(os.getenv('MAX_NOVEL_SUMMARY_LENGTH', 20000))

# MongoDB Configuration
ENABLE_MONOGODB = os.getenv('ENABLE_MONGODB', 'false').lower() == 'true'
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://127.0.0.1:27017/')
MONOGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'llm_api')
ENABLE_MONOGODB_CACHE = os.getenv('ENABLE_MONGODB_CACHE', 'false').lower() == 'true'
CACHE_REPLAY_SPEED = float(os.getenv('CACHE_REPLAY_SPEED', 2))
CACHE_REPLAY_MAX_DELAY = float(os.getenv('CACHE_REPLAY_MAX_DELAY', 5))

# API Cost Limits
API_COST_LIMITS = {
    'HOURLY_LIMIT_RMB': float(os.getenv('API_HOURLY_LIMIT_RMB', 100)),
    'DAILY_LIMIT_RMB': float(os.getenv('API_DAILY_LIMIT_RMB', 500)),
    'USD_TO_RMB_RATE': float(os.getenv('API_USD_TO_RMB_RATE', 7))
}

# API Settings
API_SETTINGS = {
    'wenxin': {
        'ak': os.getenv('WENXIN_AK', ''),
        'sk': os.getenv('WENXIN_SK', ''),
        'available_models': os.getenv('WENXIN_AVAILABLE_MODELS', '').split(','),
        'max_tokens': 4096,
    },
    'doubao': {
        'api_key': os.getenv('DOUBAO_API_KEY', ''),
        'endpoint_ids': os.getenv('DOUBAO_ENDPOINT_IDS', '').split(','),
        'available_models': os.getenv('DOUBAO_AVAILABLE_MODELS', '').split(','),
        'max_tokens': 4096,
    },
    'gpt': {
        'base_url': os.getenv('GPT_BASE_URL', ''),
        'api_key': os.getenv('GPT_API_KEY', ''),
        'proxies': os.getenv('GPT_PROXIES', ''),
        'available_models': os.getenv('GPT_AVAILABLE_MODELS', '').split(','),
        'max_tokens': 4096,
    },
    'zhipuai': {
        'api_key': os.getenv('ZHIPUAI_API_KEY', ''),
        'available_models': os.getenv('ZHIPUAI_AVAILABLE_MODELS', '').split(','),
        'max_tokens': 4096,
    },
    'local': {
        'base_url': os.getenv('LOCAL_BASE_URL', ''),
        'api_key': os.getenv('LOCAL_API_KEY', ''),
        'available_models': os.getenv('LOCAL_AVAILABLE_MODELS', '').split(','),
        'max_tokens': 4096,
    }
}

for model in API_SETTINGS.values():
    model['available_models'] = [e.strip() for e in model['available_models']]

DEFAULT_MAIN_MODEL = os.getenv('DEFAULT_MAIN_MODEL', 'wenxin/ERNIE-Novel-8K')
DEFAULT_SUB_MODEL = os.getenv('DEFAULT_SUB_MODEL', 'wenxin/ERNIE-3.5-8K')

ENABLE_ONLINE_DEMO = os.getenv('ENABLE_ONLINE_DEMO', 'false').lower() == 'true'


# Fields each provider exposes to the UI. Order matters for rendering.
PROVIDER_UI_FIELDS = {
    'doubao': ['api_key', 'endpoint_ids', 'available_models'],
    'zhipuai': ['api_key', 'available_models'],
    'gpt': ['base_url', 'api_key', 'available_models'],
    'wenxin': ['ak', 'sk', 'available_models'],
    'local': ['base_url', 'api_key', 'available_models'],
}
_LIST_FIELDS = {'endpoint_ids', 'available_models'}


def _apply_provider_config(provider, cfg):
    """Merge a provider config dict into API_SETTINGS in place."""
    if provider not in API_SETTINGS:
        return
    target = API_SETTINGS[provider]
    for key, value in cfg.items():
        if key in _LIST_FIELDS:
            if isinstance(value, str):
                value = [e.strip() for e in value.split(',') if e.strip()]
            elif isinstance(value, list):
                value = [str(e).strip() for e in value if str(e).strip()]
            else:
                continue
        target[key] = value


def reload_user_config():
    """Re-read user_config.json and overlay it on API_SETTINGS + defaults."""
    global DEFAULT_MAIN_MODEL, DEFAULT_SUB_MODEL
    data = _load_user_config_file()
    for provider, cfg in (data.get('providers') or {}).items():
        _apply_provider_config(provider, cfg)
    if data.get('DEFAULT_MAIN_MODEL'):
        DEFAULT_MAIN_MODEL = data['DEFAULT_MAIN_MODEL']
    if data.get('DEFAULT_SUB_MODEL'):
        DEFAULT_SUB_MODEL = data['DEFAULT_SUB_MODEL']


def save_provider_config(provider, cfg):
    """Persist a single provider's config and apply to memory."""
    if provider not in API_SETTINGS:
        raise ValueError(f"unknown provider: {provider}")
    with _user_config_lock:
        data = _load_user_config_file()
        providers = data.setdefault('providers', {})
        existing = providers.get(provider, {})
        existing.update(cfg)
        providers[provider] = existing
        _write_user_config_file(data)
        _apply_provider_config(provider, cfg)


def save_default_models(main_model=None, sub_model=None):
    """Persist DEFAULT_MAIN_MODEL / DEFAULT_SUB_MODEL choices."""
    global DEFAULT_MAIN_MODEL, DEFAULT_SUB_MODEL
    with _user_config_lock:
        data = _load_user_config_file()
        if main_model:
            data['DEFAULT_MAIN_MODEL'] = main_model
            DEFAULT_MAIN_MODEL = main_model
        if sub_model:
            data['DEFAULT_SUB_MODEL'] = sub_model
            DEFAULT_SUB_MODEL = sub_model
        _write_user_config_file(data)


# Apply persisted user overrides on top of .env defaults.
reload_user_config()
