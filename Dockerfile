FROM python:3.10-slim

# System deps (needed for asyncpg + common crypto/ssl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN mkdir -p /var/log/wan-kenobi/gunicorn \
    && touch /var/log/wan-kenobi/gunicorn/access.log \
    && touch /var/log/wan-kenobi/gunicorn/error.log

# Copy application code
COPY . /app

EXPOSE 3000

# ---- Production server ----
# Tune workers via env var WEB_CONCURRENCY (common pattern).
# Use --preload only if your startup is fast and safe to preload.
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker main:app \
  --bind 0.0.0.0:3000 \
  --workers ${WEB_CONCURRENCY:-2} \
  --threads ${GUNICORN_THREADS:-1} \
  --timeout ${GUNICORN_TIMEOUT:-60} \
  --graceful-timeout ${GUNICORN_GRACEFUL_TIMEOUT:-30} \
  --keep-alive ${GUNICORN_KEEPALIVE:-5} \
  --access-logfile /var/log/wan-kenobi/gunicorn/access.log \
  --error-logfile /var/log/wan-kenobi/gunicorn/error.log"]
