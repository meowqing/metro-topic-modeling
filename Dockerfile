FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_inference.txt .

# Устанавливаем отдельно для избежания конфликтов
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir scipy
RUN pip install --no-cache-dir -r requirements_inference.txt

COPY app/ ./app/

EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]