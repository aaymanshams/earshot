# Default image is small and offline-capable.
# Build with --build-arg REQUIREMENTS=requirements-ml.txt for real embeddings,
# re-ranking and speech.
FROM python:3.11-slim
ARG REQUIREMENTS=requirements.txt
WORKDIR /app
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS}
COPY earshot ./earshot
COPY web ./web
COPY evals/data ./evals/data
RUN useradd --create-home earshot
USER earshot
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "earshot.api:app", "--host", "0.0.0.0", "--port", "8000"]
