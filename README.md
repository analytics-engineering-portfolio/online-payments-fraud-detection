# Online Payments Fraud Detection

A Machine Learning based web application that detects fraudulent online payment transactions using a Random Forest classification model.

## Problem Statement

Online payment fraud is a major challenge in digital transactions. Fraudulent transactions can cause financial losses to customers and businesses.

This project uses Machine Learning to analyze transaction details and predict whether a transaction is **Fraud** or **Not Fraud**.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- FastAPI
- Pydantic
- Joblib
- HTML
- CSS
- JavaScript
- REST API
- Docker

## Dataset

The project uses an online payments transaction dataset containing transaction information such as transaction type, amount, account balances, and fraud labels.

The target variable is:

- `isFraud` — indicates whether a transaction is fraudulent.

## Project Structure

```text
Online-Payments-Fraud-Detection/
│
├── dataset/
│   └── PS_20174392719_1491204439457_log.csv
│
├── models/
│   ├── fraud_detection_model.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── fraud_detection_model.ipynb
│
├── src/
│   └── main.py
│
├── frontend/
│   └── index.html
│
├── requirements.txt
├── README.md
└── venv/