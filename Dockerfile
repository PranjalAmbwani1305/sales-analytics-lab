# Dockerfile — Streamlit Sales Analytics App
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "lab_sales_analytics_app.py", "--server.port=8501", "--server.enableCORS=false"]