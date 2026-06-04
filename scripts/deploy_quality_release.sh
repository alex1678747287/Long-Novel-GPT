#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET="${SSH_TARGET:?Set SSH_TARGET, for example root@124.174.16.151}"
RELEASE_IMAGE="${RELEASE_IMAGE:-tmp/release/long-novel-gpt-codex-quality.tar.gz}"
CHECKSUM_FILE="${CHECKSUM_FILE:-$RELEASE_IMAGE.sha256}"
REMOTE_DIR="${REMOTE_DIR:-/opt/lng/releases}"
REMOTE_PORT="${REMOTE_PORT:-8233}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-http://124.174.16.151:$REMOTE_PORT}"
POST_DEPLOY_AUDIT="${POST_DEPLOY_AUDIT:-1}"
AUDIT_NOVEL_ID="${AUDIT_NOVEL_ID:-ac424df848d5}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:-tmp/cloud_quality_audit_${AUDIT_NOVEL_ID}.after-deploy.json}"
CONTAINER_NAME="${CONTAINER_NAME:-long-novel-gpt}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-long-novel-gpt:amd64}"
LOADED_IMAGE="${LOADED_IMAGE:-long-novel-gpt:codex-quality}"
DATA_DIR="${DATA_DIR:-/opt/lng/data}"

if [ ! -f "$RELEASE_IMAGE" ]; then
    echo "release image not found: $RELEASE_IMAGE" >&2
    exit 1
fi
if [ ! -f "$CHECKSUM_FILE" ]; then
    echo "checksum file not found: $CHECKSUM_FILE" >&2
    exit 1
fi

release_name="$(basename "$RELEASE_IMAGE")"
checksum_name="$(basename "$CHECKSUM_FILE")"

echo "[1/4] Uploading release files to $SSH_TARGET:$REMOTE_DIR"
ssh "$SSH_TARGET" "mkdir -p '$REMOTE_DIR'"
scp "$RELEASE_IMAGE" "$CHECKSUM_FILE" "$SSH_TARGET:$REMOTE_DIR/"

echo "[2/4] Loading image and restarting container"
ssh "$SSH_TARGET" \
  "REMOTE_DIR='$REMOTE_DIR' RELEASE_NAME='$release_name' CHECKSUM_NAME='$checksum_name' RUNTIME_IMAGE='$RUNTIME_IMAGE' LOADED_IMAGE='$LOADED_IMAGE' CONTAINER_NAME='$CONTAINER_NAME' DATA_DIR='$DATA_DIR' REMOTE_PORT='$REMOTE_PORT' bash -s" <<'REMOTE'
set -euo pipefail

cd "$REMOTE_DIR"
sha256sum -c "$CHECKSUM_NAME"

stamp="$(date +%Y%m%d%H%M%S)"
rollback_image="${RUNTIME_IMAGE%:*}:rollback-$stamp"
data_parent="$(dirname "$DATA_DIR")"
data_base="$(basename "$DATA_DIR")"

docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1 && docker image tag "$RUNTIME_IMAGE" "$rollback_image" || true
if [ -d "$DATA_DIR" ]; then
    tar --warning=no-file-changed --ignore-failed-read \
      -C "$data_parent" \
      -czf "$data_parent/data-backup-$stamp.tar.gz" \
      "$data_base"
fi

gunzip -c "$RELEASE_NAME" | docker load
docker tag "$LOADED_IMAGE" "$RUNTIME_IMAGE"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  -p "$REMOTE_PORT:80" \
  -v "$DATA_DIR:/app/data" \
  "$RUNTIME_IMAGE" >/dev/null

for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:$REMOTE_PORT/api/health" >/dev/null; then
        exit 0
    fi
    sleep 2
done

echo "health check failed; attempting rollback to $rollback_image" >&2
docker logs --tail 120 "$CONTAINER_NAME" >&2 || true
if docker image inspect "$rollback_image" >/dev/null 2>&1; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker tag "$rollback_image" "$RUNTIME_IMAGE"
    docker run -d \
      --name "$CONTAINER_NAME" \
      --restart unless-stopped \
      -p "$REMOTE_PORT:80" \
      -v "$DATA_DIR:/app/data" \
      "$RUNTIME_IMAGE" >/dev/null
fi
exit 1
REMOTE

echo "[3/4] Verifying public endpoints"
curl -fsS "$PUBLIC_BASE_URL/api/health" >/dev/null
curl -fsS "$PUBLIC_BASE_URL/api/v2/models" >/dev/null

if [ "$POST_DEPLOY_AUDIT" = "1" ]; then
    echo "[4/4] Running read-only cloud quality audit"
    if python3 - <<'PY' >/dev/null 2>&1
import flask  # noqa: F401
PY
    then
        python3 scripts/cloud_quality_audit.py \
          --base-url "$PUBLIC_BASE_URL" \
          --novel-id "$AUDIT_NOVEL_ID" \
          --output "$AUDIT_OUTPUT"
    else
        docker run --rm \
          -v "$PWD:/work" \
          -w /work \
          long-novel-gpt:local \
          python scripts/cloud_quality_audit.py \
            --base-url "$PUBLIC_BASE_URL" \
            --novel-id "$AUDIT_NOVEL_ID" \
            --output "$AUDIT_OUTPUT"
    fi
else
    echo "[4/4] Skipping read-only cloud quality audit"
fi

echo "Deployment finished. Run a real rewrite acceptance test next."
