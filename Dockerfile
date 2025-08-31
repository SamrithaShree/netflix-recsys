FROM pytorch/pytorch:2.3.0-cpu

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt --no-deps

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
