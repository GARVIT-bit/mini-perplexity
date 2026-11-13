# Retrieval + answer service. Build context is the repo root (see docker-compose.yml).
# The index is NOT in the image: compose mounts ./data/index read-only, so
# rebuilding the index never means rebuilding the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app
RUN useradd --create-home app

# CPU-only torch first: the default wheel bundles CUDA libraries (gigabytes) that a CPU never uses
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# chroma-hnswlib: take the prebuilt wheel; building hnswlib from source needs a C++ toolchain
RUN pip install --only-binary=chroma-hnswlib -r requirements.txt

# Bake the embedding model into the image: containers start without a download and work offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && chown -R app /opt/hf
ENV HF_HUB_OFFLINE=1

# Only what the service imports (crawler/ and bench/ are offline tools)
COPY indexer/ indexer/
COPY retrieval/ retrieval/
COPY answer/ answer/
COPY api/ api/

USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
# 0.0.0.0 inside the container; compose does not publish this port, so it is reachable only by `server`
CMD ["python", "-m", "api.app", "--host", "0.0.0.0", "--port", "8000"]
