FROM python:3.8-slim-buster

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mlflow server and application files
COPY mlflow_server.py .
COPY salesprediction.py .
COPY deploy.py .
COPY data_preprocessing.py .

# Expose mlflow server port
EXPOSE 5000

CMD ["mlflow", "server", "--default-artifact-root", "file:/app/artifacts", "--host", "0.0.0.0"]
