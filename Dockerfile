# ---- Base image ----
FROM python:3.10-slim

# ---- System deps (asyncpg/psycopg builds + ssl) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Python runtime env ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- Install deps first (better caching) ----
# Expect a requirements.txt in your project root.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- Copy app code ----
COPY . /app

# ---- Expose FastAPI port ----
EXPOSE 3000

# ---- Start server ----
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]
