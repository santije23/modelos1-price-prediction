FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH="/app/fase-2/src"

WORKDIR /app/fase-2

CMD ["python", "scripts/train.py", "--help"]
