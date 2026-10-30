# Лёгкий образ для CI: код + тесты, без моделей (они монтируются томом при запуске)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ci.txt .
RUN pip install --no-cache-dir -r requirements-ci.txt

COPY pyproject.toml config.yaml ./
COPY aikoru ./aikoru
COPY tests ./tests

# файлы с Windows приходят с битом исполнения, а ruff (EXE002) его не принимает
RUN find /app -type f -exec chmod 644 {} +

# По умолчанию — прогон тестов; для запуска ассистента: -v ./models:/app/models
CMD ["pytest", "tests", "-v"]
