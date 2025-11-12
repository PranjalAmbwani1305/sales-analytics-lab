# 📊 Sales Analytics Lab — Streamlit + Docker + Kubernetes

## 🚀 Overview
This project visualizes and forecasts sales data using Streamlit and Linear Regression. 
It includes Docker and Kubernetes configurations for cloud deployment.

## 🧩 Files
- `lab_sales_analytics_app.py` — Main Streamlit App
- `requirements.txt` — Python dependencies
- `Dockerfile` — Container build file
- `k8s-deployment.yaml` — Kubernetes deployment & service
- `sales_data.csv` — Sample dataset

## 🐳 Docker Commands
```bash
docker build -t sales-analytics-app:latest .
docker run -p 8501:8501 sales-analytics-app:latest
```

## ☸️ Kubernetes Deployment
```bash
kubectl apply -f k8s-deployment.yaml
kubectl get pods
kubectl get svc
```

## 🌍 Online Hosting (Streamlit Cloud)
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io
3. Deploy directly using `lab_sales_analytics_app.py`