#!/bin/bash

# 设置默认值
FRONTEND_PORT=${FRONTEND_PORT:-80}
BACKEND_PORT=${BACKEND_PORT:-7869}
BACKEND_HOST=${BACKEND_HOST:-0.0.0.0}
WORKERS=${WORKERS:-4}
THREADS=${THREADS:-2}
TIMEOUT=${TIMEOUT:-900}
REWRITE_WORKER_ENABLED=${REWRITE_WORKER_ENABLED:-1}
REWRITE_WORKER_CONCURRENCY=${REWRITE_WORKER_CONCURRENCY:-3}
REWRITE_JOB_MAX_AUTO_RETRIES=${REWRITE_JOB_MAX_AUTO_RETRIES:-4}

# 在Linux环境下添加host.docker.internal解析
# if ! grep -q "host.docker.internal" /etc/hosts; then
#     DOCKER_INTERNAL_HOST="$(ip route | grep default | awk '{print $3}')"
#     echo "$DOCKER_INTERNAL_HOST host.docker.internal" >> /etc/hosts
# fi

# 替换nginx配置中的端口
sed -i "s/listen 9999/listen $FRONTEND_PORT/g" /etc/nginx/conf.d/default.conf
sed -i "s/host.docker.internal:7869/localhost:$BACKEND_PORT/g" /etc/nginx/conf.d/default.conf

# 启动nginx
nginx

if [ "$REWRITE_WORKER_ENABLED" = "1" ]; then
    python - <<'PY'
from v2 import storage

count = storage.recover_running_rewrite_jobs("container startup")
if count:
    print(f"recovered {count} running rewrite jobs", flush=True)
PY
    for i in $(seq 1 "$REWRITE_WORKER_CONCURRENCY"); do
        (
            export REWRITE_WORKER_ID="worker-$i"
            while true; do
                python -m v2.rewrite_worker
                echo "rewrite worker $REWRITE_WORKER_ID exited; restarting in 2s" >&2
                sleep 2
            done
        ) &
    done
fi

# 启动gunicorn
gunicorn --bind $BACKEND_HOST:$BACKEND_PORT \
    --workers $WORKERS \
    --threads $THREADS \
    --worker-class gthread \
    --timeout $TIMEOUT \
    --access-logfile - \
    --error-logfile - \
    app:app
