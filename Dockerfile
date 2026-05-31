# ===== Stage 1: build the Vue3 frontend =====
FROM node:20-alpine AS frontend-build

WORKDIR /build

# Use a faster mirror for npm in CN — falls back to npmjs.org if unreachable.
RUN npm config set registry https://registry.npmmirror.com

COPY frontend2/package.json frontend2/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend2/ ./
RUN npm run build


# ===== Stage 2: backend + nginx runtime =====
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx iproute2 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default \
    && rm -f /etc/nginx/sites-available/default

# Python deps
COPY backend/requirements.txt .
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --no-cache-dir -r requirements.txt

# Backend code
COPY config.py .
COPY backend/app.py .
COPY backend/setting.py .
COPY backend/summary.py .
COPY backend/backend_utils.py .
COPY backend/healthcheck.py .
COPY backend/v2/ ./v2/
COPY core/ ./core/
COPY llm_api/ ./llm_api/
COPY prompts/ ./prompts/
COPY custom/ ./custom/

# Frontend dist from build stage
COPY --from=frontend-build /build/dist /usr/share/nginx/html

# Nginx config (kept in legacy frontend/ dir; same /api proxy still applies)
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

COPY start.sh .
RUN chmod +x start.sh

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python healthcheck.py

CMD ["./start.sh"]
