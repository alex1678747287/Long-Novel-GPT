#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-long-novel-gpt:codex-quality}"
PLATFORM="${PLATFORM:-linux/amd64}"
OUTPUT_DIR="${OUTPUT_DIR:-tmp/release}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-$OUTPUT_DIR/long-novel-gpt-codex-quality.tar.gz}"
SMOKE_CONTAINER="${SMOKE_CONTAINER:-long-novel-gpt-codex-smoke}"
SMOKE_PORT="${SMOKE_PORT:-18233}"
SKIP_BUILD="${SKIP_BUILD:-0}"

mkdir -p "$OUTPUT_DIR"

cleanup_smoke() {
    docker rm -f "$SMOKE_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup_smoke EXIT

if [ "$SKIP_BUILD" = "1" ]; then
    echo "[1/4] Skipping build, using existing image: $IMAGE_TAG"
    docker image inspect "$IMAGE_TAG" >/dev/null
else
    echo "[1/4] Building production image: $IMAGE_TAG ($PLATFORM)"
    docker build --platform "$PLATFORM" -t "$IMAGE_TAG" .
fi

echo "[2/4] Running image quality self-check"
docker run --rm -i "$IMAGE_TAG" python - <<'PY'
from v2 import api

source = '系统显示6月12号离婚。再睁眼，我倒在捡垃圾的路上。请唐楠到2号诊室体检。'
rewritten = '系统记录6月12号离婚。我捡了三年垃圾，最后倒在雨夜路边。请沈青柠到2号诊室体检。'
quality = api.score_rewrite_quality(rewritten, source, name_map={'唐楠': '沈青柠'})
assert quality['delivery_status'] == 'risk'
assert any('非核心细节照搬' in item for item in quality['issues'])
assert any('叙述骨架照搬' in item for item in quality['issues'])
print(quality['delivery_status'])
for issue in quality['issues']:
    print(issue)
PY

echo "[3/4] Running container HTTP smoke test on port $SMOKE_PORT"
cleanup_smoke
docker run -d \
  --name "$SMOKE_CONTAINER" \
  -e REWRITE_WORKER_ENABLED=0 \
  -p "$SMOKE_PORT:80" \
  "$IMAGE_TAG" >/dev/null

for _ in $(seq 1 20); do
    if curl -fsS "http://127.0.0.1:$SMOKE_PORT/api/health" >/dev/null; then
        break
    fi
    sleep 1
done

curl -fsS "http://127.0.0.1:$SMOKE_PORT/api/health" >/dev/null
QUALITY_RESPONSE="$OUTPUT_DIR/quality-smoke-response.json"
curl -fsS "http://127.0.0.1:$SMOKE_PORT/api/v2/quality/score" \
  -H 'Content-Type: application/json' \
  -d '{"source":"系统显示6月12号离婚。再睁眼，我倒在捡垃圾的路上。请唐楠到2号诊室体检。","rewritten":"系统记录6月12号离婚。我捡了三年垃圾，最后倒在雨夜路边。请沈青柠到2号诊室体检。","name_map":{"唐楠":"沈青柠"}}' \
  > "$QUALITY_RESPONSE"
python3 - "$QUALITY_RESPONSE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as fh:
    quality = json.load(fh)
assert quality['delivery_status'] == 'risk'
assert any('非核心细节照搬' in item for item in quality['issues'])
assert any('叙述骨架照搬' in item for item in quality['issues'])
print(quality['delivery_status'])
PY
cleanup_smoke

echo "[4/4] Exporting image archive: $OUTPUT_IMAGE"
docker save "$IMAGE_TAG" | gzip > "$OUTPUT_IMAGE"
python3 - "$OUTPUT_IMAGE" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open('rb') as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b''):
        digest.update(chunk)
checksum_path = path.with_suffix(path.suffix + '.sha256')
checksum_path.write_text(f"{digest.hexdigest()}  {path.name}\n", encoding='utf-8')
print(f"SHA256: {checksum_path}")
PY

cat <<EOF
Release archive is ready:
  $OUTPUT_IMAGE
  $OUTPUT_IMAGE.sha256

Upload it to the server, then run:
  sha256sum -c long-novel-gpt-codex-quality.tar.gz.sha256
  gunzip -c long-novel-gpt-codex-quality.tar.gz | docker load
  docker tag $IMAGE_TAG long-novel-gpt:amd64
  docker rm -f long-novel-gpt >/dev/null 2>&1 || true
  docker run -d --name long-novel-gpt --restart unless-stopped -p 8233:80 -v /opt/lng/data:/app/data long-novel-gpt:amd64
EOF
