FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fase-2/ ./fase-2/

ENV PYTHONPATH="/app/fase-2/src"

WORKDIR /app/fase-2

CMD ["python", "scripts/train.py", "--config", "configs/config.yaml"]
