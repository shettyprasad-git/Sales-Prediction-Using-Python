# 📈 Sales Prediction Using Python

## Model Development & Deployment Documentation

🔗 **Live Streamlit App:**  
https://salespredictionusingpython.streamlit.app/

---

## 📌 Project Overview

This project focuses on predicting product sales based on advertising expenditure across multiple marketing channels using machine learning techniques in Python. The objective is to analyze historical advertising data, build a regression model, and deploy the trained model as an interactive Streamlit web application.

The project demonstrates an end-to-end data science workflow, covering data analysis, feature engineering, model training, evaluation, and deployment.

The work is divided into two major components:
1. Data Analysis & Model Development (Jupyter Notebook)  
2. Model Deployment using Streamlit  

---

## 📊 Dataset Description

The dataset used in this project is the **Advertising Dataset**, commonly used for regression analysis and sales prediction tasks.

**Dataset Source:**  
Kaggle – Advertising Dataset

**Dataset Features:**
- **TV** – Advertising budget spent on TV promotions  
- **Radio** – Advertising budget spent on radio promotions  
- **Newspaper** – Advertising budget spent on newspaper promotions  
- **Sales** – Product sales (target variable)

---

## 🔍 Exploratory Data Analysis (EDA)

- Performed univariate, bivariate, and multivariate analysis  
- Correlation analysis confirmed TV as the strongest predictor of Sales  
- Linear Regression identified as a suitable model  

---

## 🤖 Model Training & Evaluation

- Model Used: Linear Regression  
- Evaluation Metrics: R² Score, MSE  
- Trained model saved as `lr_model.pkl`  

---

## 🌐 Streamlit Application

An interactive Streamlit app allows real-time sales prediction based on advertising spend.

🔗 **Live App:**  
https://salespredictionusingpython.streamlit.app/

---

## 📁 Project Structure

```
Sales-Prediction-Using-Python/
├── SALES_PREDICTION_USING_PYTHON.ipynb
├── lr_model.pkl
├── Sales_Prediction_Using_Python_Documentation.pdf
├── app.py
├── requirements.txt
├── sales_predictions.csv
└── README.md
```

---

## 👨‍💻 Author

**Durga Prasad**  
GitHub: https://github.com/shettyprasad-git  
LinkedIn: https://www.linkedin.com/in/durgaprasadshetty  
