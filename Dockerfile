FROM python:3.10-slim-bullseye

# Установка системных зависимостей
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        xz-utils \
        fontconfig \
        libxrender1 \
        libxtst6 \
        libjpeg62-turbo \
        libssl1.1 \
        libpng16-16 \
        libxcb1 \
        libx11-6 \
        libpq5 \
        && \
    wget -q https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6.1-2/wkhtmltox_0.12.6.1-2.bullseye_amd64.deb -O /tmp/wkhtmltox.deb && \
    apt-get install -y --no-install-recommends /tmp/wkhtmltox.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/wkhtmltox.deb

WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ВСЕ файлы проекта
COPY main.py config.py requirements.txt ./
COPY database/ ./database/
COPY services/ ./services/
COPY clients/ ./clients/
COPY utils/ ./utils/

# Запуск приложения
CMD ["python", "main.py"]