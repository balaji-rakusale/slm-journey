FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY day22_api.py .

EXPOSE 8000

CMD ["uvicorn", "day22_api:app", "--host", "0.0.0.0", "--port", "8000"]