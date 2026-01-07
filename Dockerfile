FROM python:3.12.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git gcc libpq-dev libssl-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements (without typing) and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]