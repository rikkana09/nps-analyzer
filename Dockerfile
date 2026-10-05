FROM python:3.12-slim

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код
COPY src/ ./src/
COPY app/ ./app/
COPY prompts/ ./prompts/
COPY data/ ./data/
COPY entrypoint.sh ./entrypoint.sh

# Делаем entrypoint исполняемым
RUN chmod +x entrypoint.sh

EXPOSE 8501

# Один entrypoint запускает всё
ENTRYPOINT ["./entrypoint.sh"]
