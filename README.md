# Advanced_Ai-AutoML-Dashboard_App
# 🚀 Universal AutoML Dashboard (Streamlit + PyCaret + Docker)

## 📌 Overview

This project is a **production-ready AutoML web application** that allows users to upload any dataset and automatically perform:

* Data Analysis (EDA)
* Model Training (Classification & Regression)
* Model Evaluation
* Predictions Download

Built using:

* PyCaret (AutoML)
* Streamlit (UI Dashboard)
* Docker (Deployment)

---

## 🎯 Features

### 📊 Exploratory Data Analysis (EDA)

* Dataset preview
* Missing values analysis
* Column-wise visualization
* Correlation heatmap

### ⚙️ AutoML Training

* Supports **Classification & Regression**
* Automatic model comparison (10+ algorithms)
* Best model selection
* Fully automated pipeline

### 📈 Results Dashboard

* Predictions table
* Accuracy (Classification)
* RMSE (Regression)
* Visualization charts

### 💾 Export

* Download predictions as CSV

---

## 🧱 Tech Stack

* Python
* Streamlit
* PyCaret
* Pandas, NumPy
* Matplotlib, Seaborn
* Docker

---

## 📂 Project Structure

```
automl-app/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## ⚙️ Installation (Local)

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/automl-app.git
cd automl-app
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run App

```
streamlit run app.py
```

---

## 🐳 Docker Setup

### 1️⃣ Build Image

```
docker build -t automl-app .
```

### 2️⃣ Run Container

```
docker run -p 8501:8501 automl-app
```

### 3️⃣ Open in Browser

```
http://localhost:8501
```

---

## 📊 Usage Guide

1. Upload a CSV dataset
2. Select target column
3. Choose problem type (Classification/Regression)
4. Click **Run AutoML**
5. View results and download predictions

---

## ⚠️ Important Notes

* Target column must not contain all missing values
* Classification requires at least **2 classes**
* Large datasets may take longer to train

---

## 🚀 Future Enhancements

* SHAP Explainability
* Feature Importance Visualization
* Model Download (.pkl)
* REST API (FastAPI)
* Cloud Deployment (AWS / GCP)
* CI/CD Pipeline

---

## 🧠 Key Highlights

* Works with **any tabular dataset**
* Fully automated ML pipeline
* Clean UI with tab-based navigation
* Dockerized for easy deployment
* Production-ready design

---

## 📬 Contact

For queries or collaboration:

* Name: Prasanna Kumar
* Role: Data Science & AI

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
