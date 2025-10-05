# Sample Dockerfile for running the multi-page RAG pipeline (TF-IDF fallback only)
# This image is intended for demonstration. For production use pin versions and use multi-stage builds.

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH="/app"
EXPOSE 8000
# Run the FastAPI server by default in the container
CMD ["uvicorn", "src.server_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
