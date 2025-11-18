FROM python:3.10-slim

# Для wkhtmltopdf нужны некоторые системные библиотеки
RUN apt-get update && \
    apt-get install -y wget xz-utils xfonts-base fontconfig \
    libxrender1 libxtst6 libjpeg62-turbo libssl3 libpng16-16 \
    libxcb1 libx11-6 build-essential libpq-dev git && \
    rm -rf /var/lib/apt/lists/*

# Установка wkhtmltopdf (версия подходит для большинства pdfkit)
RUN wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.4/wkhtmltox-0.12.4_linux-generic-amd64.tar.xz -O /tmp/wkhtmltox.tar.xz && \
    tar -xJf /tmp/wkhtmltox.tar.xz -C /tmp && \
    cp /tmp/wkhtmltox/bin/wkhtmltopdf /usr/local/bin/ && \
    chmod +x /usr/local/bin/wkhtmltopdf && \
    rm -rf /tmp/wkhtmltox*

# Копируем архив зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё приложение
COPY . .

# По умолчанию main.py лежит в корне репозитория
CMD ["python", "main.py"]