import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask

from backend import setting


class LegacySettingSecurityTest(unittest.TestCase):
    def _fake_config(self, save_provider_config=None):
        fake = types.ModuleType("config")
        fake.API_SETTINGS = {
            "gpt": {
                "base_url": "https://api.example.test/v1",
                "api_key": "sk-live-secret",
                "available_models": ["demo-model"],
            }
        }
        fake.PROVIDER_UI_FIELDS = {
            "gpt": ["base_url", "api_key", "available_models"],
        }
        fake.DEFAULT_MAIN_MODEL = "gpt/demo-model"
        fake.DEFAULT_SUB_MODEL = "gpt/demo-model"
        fake.MAX_THREAD_NUM = 5
        fake.MAX_NOVEL_SUMMARY_LENGTH = 20000
        fake.reload_user_config = lambda: None
        fake.save_default_models = lambda main_model=None, sub_model=None: None
        fake.save_provider_config = save_provider_config or (lambda provider, cfg: None)
        return fake

    def test_legacy_setting_serializer_masks_secret_values(self):
        with patch.dict(sys.modules, {"config": self._fake_config()}):
            payload = setting._serialize_api_settings()

        raw = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("sk-live-secret", raw)
        api_key = payload["gpt"]["fields"][1]
        self.assertTrue(api_key["is_secret"])
        self.assertEqual(api_key["value"], "sk-****cret")

    def test_legacy_setting_save_ignores_masked_secret_placeholders(self):
        saved = {}

        def save_provider_config(provider, cfg):
            saved["provider"] = provider
            saved["cfg"] = cfg

        app = Flask(__name__)
        app.register_blueprint(setting.setting_bp)
        fake_config = self._fake_config(save_provider_config=save_provider_config)

        with patch.dict(sys.modules, {"config": fake_config}):
            res = app.test_client().post(
                "/setting/api",
                json={
                    "provider": "gpt",
                    "config": {
                        "base_url": "https://api.example.test/v2",
                        "api_key": "sk-****cret",
                        "available_models": "demo-model",
                    },
                },
            )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(saved["provider"], "gpt")
        self.assertNotIn("api_key", saved["cfg"])
        self.assertEqual(saved["cfg"]["base_url"], "https://api.example.test/v2")

    def test_legacy_setting_save_ignores_blank_secret_placeholders(self):
        saved = {}

        def save_provider_config(provider, cfg):
            saved["cfg"] = cfg

        app = Flask(__name__)
        app.register_blueprint(setting.setting_bp)

        with patch.dict(sys.modules, {"config": self._fake_config(save_provider_config=save_provider_config)}):
            res = app.test_client().post(
                "/setting/api",
                json={
                    "provider": "gpt",
                    "config": {
                        "base_url": "https://api.example.test/v2",
                        "api_key": "",
                    },
                },
            )

        self.assertEqual(res.status_code, 200)
        self.assertNotIn("api_key", saved["cfg"])

    def test_config_source_does_not_print_raw_env_values(self):
        source = Path("config.py").read_text(encoding="utf-8")
        self.assertNotIn('print(f"{key}={value}")', source)

    def test_sparkai_source_does_not_hardcode_credentials(self):
        source = Path("llm_api/sparkai_api.py").read_text(encoding="utf-8")

        self.assertNotIn("01793781", source)
        self.assertNotIn("YzJkNTI5N2Q5NDY4N2RlNWI5YjA5ZDM4", source)
        self.assertNotIn("5dd33ea830aff0c9dff18e2561a5e6c7", source)
        self.assertIn("os.getenv('SPARKAI_APP_ID'", source)

    def test_nginx_api_location_allows_any_origin_for_public_workbench(self):
        source = Path("frontend/nginx.conf").read_text(encoding="utf-8")
        api_location = source.split("location /api/ {", 1)[1].split("\n    }", 1)[0]

        self.assertIn('add_header Access-Control-Allow-Origin "*" always;', api_location)
        self.assertIn("proxy_hide_header Access-Control-Allow-Origin;", api_location)
        self.assertIn("proxy_buffering off;", api_location)

    def test_backend_public_api_allows_cross_origin_without_origin_guard(self):
        app_source = Path("backend/app.py").read_text(encoding="utf-8")

        self.assertNotIn("origin not allowed", app_source)
        self.assertNotIn("def _guard_api_origin", app_source)
        self.assertIn("Access-Control-Allow-Origin'] = '*'", app_source)

    def test_runtime_timeouts_cover_long_model_streams(self):
        start = Path("start.sh").read_text(encoding="utf-8")
        nginx = Path("frontend/nginx.conf").read_text(encoding="utf-8")

        self.assertIn("TIMEOUT=${TIMEOUT:-900}", start)
        self.assertIn("proxy_send_timeout 900s;", nginx)
        self.assertIn("proxy_read_timeout 900s;", nginx)

    def test_runtime_starts_bounded_parallel_rewrite_workers(self):
        start = Path("start.sh").read_text(encoding="utf-8")

        self.assertIn("REWRITE_WORKER_CONCURRENCY=${REWRITE_WORKER_CONCURRENCY:-3}", start)
        self.assertIn("REWRITE_JOB_MAX_AUTO_RETRIES=${REWRITE_JOB_MAX_AUTO_RETRIES:-4}", start)
        self.assertIn("recover_running_rewrite_jobs(\"container startup\")", start)
        self.assertIn("seq 1 \"$REWRITE_WORKER_CONCURRENCY\"", start)
        self.assertIn("REWRITE_WORKER_ID=\"worker-$i\"", start)

    def test_asset_cache_header_is_not_declared_twice(self):
        source = Path("frontend/nginx.conf").read_text(encoding="utf-8")
        asset_location = source.split("location /assets/ {", 1)[1].split("\n    }", 1)[0]

        self.assertNotIn("expires 30d;", asset_location)
        self.assertIn(
            'add_header Cache-Control "public, max-age=2592000, immutable" always;',
            asset_location,
        )

    def test_dockerfile_declares_container_healthcheck(self):
        source = Path("Dockerfile").read_text(encoding="utf-8")

        self.assertIn("npm ci --no-audit --no-fund", source)
        self.assertIn("pip install --no-cache-dir -r requirements.txt", source)
        self.assertIn("HEALTHCHECK", source)
        self.assertIn("python healthcheck.py", source)

    def test_mongodb_cache_defaults_away_from_plaintext_llm_payloads(self):
        config_source = Path("config.py").read_text(encoding="utf-8")
        cache_source = Path("llm_api/mongodb_cache.py").read_text(encoding="utf-8")

        self.assertIn("ENABLE_MONGODB_CACHE', 'false'", config_source)
        self.assertIn("ALLOW_PLAINTEXT_LLM_CACHE", cache_source)
