# The API half, for Cloud Run.
#
# One image, one process: uvicorn serving api.main:app. The client's four data
# files ship inside the image (data/source, ~13 MB) so a revision is a complete,
# reproducible statement of "this code over this data" — there is no bucket to
# drift from what was verified. Everything environment-dependent is read from
# the environment by api/config.py: NTT_GEMINI_KEYS, NTT_OPENROUTER_KEY,
# NTT_ALLOWED_ORIGINS, NTT_AS_OF. None of it is baked in here.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY data/source ./data/source
ENV NTT_DATA_DIR=/app/data/source

# Cloud Run injects PORT; 8080 is its default.
ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1
