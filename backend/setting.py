from flask import Blueprint, jsonify, request

setting_bp = Blueprint('setting', __name__)


_SECRET_FIELDS = {'api_key', 'sk', 'ak'}
_MASK_TOKEN = '****'


def _mask_secret(value):
    value = str(value or '')
    if not value:
        return ''
    if len(value) <= 4:
        return _MASK_TOKEN
    prefix = value[:3] if len(value) > 8 else value[:1]
    suffix = value[-4:] if len(value) > 8 else value[-1:]
    return f'{prefix}{_MASK_TOKEN}{suffix}'


def _is_masked_secret(value):
    return isinstance(value, str) and _MASK_TOKEN in value


def _sanitize_provider_config(cfg):
    clean = {}
    for field, value in (cfg or {}).items():
        if field in _SECRET_FIELDS:
            if value in ('', None) or _is_masked_secret(value):
                continue
        clean[field] = value
    return clean


def _serialize_api_settings():
    """Return API_SETTINGS in a shape the UI can render."""
    from config import API_SETTINGS, PROVIDER_UI_FIELDS

    out = {}
    for provider, fields in PROVIDER_UI_FIELDS.items():
        cfg = API_SETTINGS.get(provider, {})
        provider_payload = {'fields': []}
        for field in fields:
            value = cfg.get(field, '')
            if isinstance(value, list):
                # Drop empty placeholders that .env split('') leaves behind.
                value = ','.join([v for v in value if v])
            if field in _SECRET_FIELDS:
                value = _mask_secret(value)
            provider_payload['fields'].append({
                'name': field,
                'value': value,
                'is_secret': field in _SECRET_FIELDS,
            })
        out[provider] = provider_payload
    return out


@setting_bp.route('/setting', methods=['GET'])
def get_settings():
    """Get current settings and models"""
    from config import API_SETTINGS, DEFAULT_MAIN_MODEL, DEFAULT_SUB_MODEL, MAX_THREAD_NUM, MAX_NOVEL_SUMMARY_LENGTH, reload_user_config
    reload_user_config()

    # Get models grouped by provider; drop empty entries so the dropdown
    # doesn't show "provider/" placeholders for unconfigured providers.
    models = {
        provider: [m for m in cfg.get('available_models', []) if m]
        for provider, cfg in API_SETTINGS.items()
    }

    settings = {
        'models': models,
        'MAIN_MODEL': DEFAULT_MAIN_MODEL,
        'SUB_MODEL': DEFAULT_SUB_MODEL,
        'MAX_THREAD_NUM': MAX_THREAD_NUM,
        'MAX_NOVEL_SUMMARY_LENGTH': MAX_NOVEL_SUMMARY_LENGTH,
        'api_settings': _serialize_api_settings(),
    }
    return jsonify(settings)


@setting_bp.route('/setting/api', methods=['POST'])
def save_api_settings():
    """Persist API credentials for a provider and apply them in-memory."""
    try:
        data = request.get_json(force=True) or {}
        provider = data.get('provider')
        cfg = _sanitize_provider_config(data.get('config') or {})
        if not provider:
            return jsonify({'success': False, 'error': 'provider is required'}), 400

        from config import save_provider_config
        save_provider_config(provider, cfg)
        return jsonify({'success': True, 'api_settings': _serialize_api_settings()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@setting_bp.route('/setting/defaults', methods=['POST'])
def save_default_models_endpoint():
    """Persist the default main/sub model selection."""
    try:
        data = request.get_json(force=True) or {}
        from config import save_default_models
        save_default_models(
            main_model=data.get('MAIN_MODEL'),
            sub_model=data.get('SUB_MODEL'),
        )
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@setting_bp.route('/test_model', methods=['POST'])
def test_model():
    """Test if a model configuration works"""
    try:
        data = request.get_json()
        provider_model = data.get('provider_model')

        from backend_utils import get_model_config_from_provider_model
        model_config = get_model_config_from_provider_model(provider_model)

        from llm_api import test_stream_chat
        response = None
        for msg in test_stream_chat(model_config):
            response = msg

        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
