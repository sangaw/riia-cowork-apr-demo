# RITA — Docker image
# Build:  docker build -t rita .
# Run UI: docker run -p 8501:8501 -v /path/to/data:/data -e NIFTY_CSV_PATH=/data/merged.csv rita ui
# Run API:docker run -p 8000:8000 -v /path/to/data:/data -e NIFTY_CSV_PATH=/data/merged.csv rita api

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for ta-lib + matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source
COPY src/ src/
COPY run_ui.py run_api.py ./

# Outputs directory
RUN mkdir -p /app/rita_output

ENV OUTPUT_DIR=/app/rita_output
ENV PYTHONPATH=/app/src

EXPOSE 8501 8000

# Entrypoint: "ui" or "api"
ENTRYPOINT ["python"]
CMD ["run_api.py"]
