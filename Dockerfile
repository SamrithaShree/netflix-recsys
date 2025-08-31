FROM python:3.12-slim

WORKDIR /app

# Install system dependencies Pillow and other libraries need, including libjpeg
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
