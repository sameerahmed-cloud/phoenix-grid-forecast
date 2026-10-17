FROM python:3.10-slim

# LightGBM needs the OpenMP runtime, which the slim image doesn't include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first: this layer is cached, so code edits don't reinstall everything
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY models ./models

ENV PYTHONUNBUFFERED=1

# Don't run as root inside the container
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]