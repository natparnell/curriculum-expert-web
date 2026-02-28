FROM python:3.11-slim

WORKDIR /app

# System dependencies for pdfplumber (PDF extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY rag_server.py .
COPY rag_pipeline.py .
COPY usage_tracker.py .
COPY app_tracker.py .
COPY curriculum-agent-config.json .
COPY curriculum-expert.html .
COPY admin.html .
COPY feedback.html .

# Knowledge files for RAG indexing
COPY knowledge/ /app/knowledge/

# Directories for runtime data
RUN mkdir -p /app/data/chromadb /app/uploads

# Environment defaults
ENV WORKSPACE_DIR=/app
ENV CLOUD_MODE=true
ENV PORT=8080

EXPOSE 8080

# Use gunicorn for production
# --workers 1: ChromaDB loads into memory per worker; 1 keeps memory usage down
# --threads 4: handle concurrent requests via threading
# --timeout 300: allow up to 5 minutes for extended-mode Opus responses
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "300", "rag_server:app"]
