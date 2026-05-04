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

# Log dirs with symlinks to stdout/stderr so `docker logs` works
RUN mkdir -p /var/log/wan-kenobi/gunicorn \
    && ln -sf /dev/stdout /var/log/wan-kenobi/gunicorn/access.log \
    && ln -sf /dev/stderr /var/log/wan-kenobi/gunicorn/error.log

# Copy application code
COPY . /app

EXPOSE 3000

# ---- Production server ----
# GUNICORN_TIMEOUT  = 600 (10 min) to handle long Anthropic Opus calls
# GUNICORN_GRACEFUL = 620  so in-flight requests finish before forced kill
# GUNICORN_KEEPALIVE = 75  to survive upstream proxy idle timeouts
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker main:app \
  --bind 0.0.0.0:3000 \
  --workers ${WEB_CONCURRENCY:-2} \
  --threads ${GUNICORN_THREADS:-1} \
  --timeout ${GUNICORN_TIMEOUT:-600} \
  --graceful-timeout ${GUNICORN_GRACEFUL_TIMEOUT:-620} \
  --keep-alive ${GUNICORN_KEEPALIVE:-75} \
  --access-logfile /var/log/wan-kenobi/gunicorn/access.log \
  --error-logfile /var/log/wan-kenobi/gunicorn/error.log \
  --log-level ${GUNICORN_LOG_LEVEL:-info}"]