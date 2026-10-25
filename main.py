from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
app = FastAPI(title="Online Payments Fraud Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load trained model
model = joblib.load("models/fraud_detection_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


# Input data format
class Transaction(BaseModel):
    step: int
    transaction_type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int


@app.post("/predict")
def predict_fraud(transaction: Transaction):

    # Calculate derived features
    balance_diff_orig = (
        transaction.oldbalanceOrg - transaction.newbalanceOrig
    )

    balance_diff_dest = (
        transaction.newbalanceDest - transaction.oldbalanceDest
    )

    # Create input data
    input_data = {
        "step": transaction.step,
        "amount": transaction.amount,
        "oldbalanceOrg": transaction.oldbalanceOrg,
        "newbalanceOrig": transaction.newbalanceOrig,
        "oldbalanceDest": transaction.oldbalanceDest,
        "newbalanceDest": transaction.newbalanceDest,
        "isFlaggedFraud": transaction.isFlaggedFraud,
        "balance_diff_orig": balance_diff_orig,
        "balance_diff_dest": balance_diff_dest,
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 0,
        "type_TRANSFER": 0
    }

    # Set transaction type
    if transaction.transaction_type == "CASH_OUT":
        input_data["type_CASH_OUT"] = 1

    elif transaction.transaction_type == "DEBIT":
        input_data["type_DEBIT"] = 1

    elif transaction.transaction_type == "PAYMENT":
        input_data["type_PAYMENT"] = 1

    elif transaction.transaction_type == "TRANSFER":
        input_data["type_TRANSFER"] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Arrange columns in the same order as training
    input_df = input_df[feature_columns]

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Get fraud probability
    probability = model.predict_proba(input_df)[0][1]

    # Result
    if prediction == 1:
        result = "Fraud"
    else:
        result = "Not Fraud"

    return {
        "prediction": int(prediction),
        "result": result,
        "fraud_probability": round(float(probability), 4)
    }