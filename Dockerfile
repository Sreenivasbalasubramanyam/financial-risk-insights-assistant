FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; the default backend has no compiled/heavy deps.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the vector index at image build time so the container is ready to
# serve queries immediately on startup.
RUN python -m src.ingest

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
