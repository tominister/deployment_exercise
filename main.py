from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and scaler
model = joblib.load("logistic_model.joblib")
scaler = joblib.load("scaler.joblib")

# Define the input data model (update fields to match your CSV columns except 'target')
class PredictRequest(BaseModel):
    Age: float
    Rating: float
    Total_Price: float
    Unit_Price: float
    Quantity: float
    Gender_Male: int
    Loyalty_Member_Yes: int
    Product_Type_Laptop: int
    Product_Type_Smartphone: int
    Product_Type_Smartwatch: int
    Product_Type_Tablet: int
    Payment_Method_Cash: int
    Payment_Method_Credit_Card: int
    Payment_Method_Debit_Card: int
    Payment_Method_PayPal: int
    Payment_Method_Paypal: int
    Shipping_Type_Express: int
    Shipping_Type_Overnight: int
    Shipping_Type_Same_Day: int
    Shipping_Type_Standard: int

@app.post("/predict")
def predict(request: PredictRequest):
    # Convert request to numpy array and scale
    features = np.array([
        request.Age,
        request.Rating,
        request.Total_Price,
        request.Unit_Price,
        request.Quantity,
        request.Gender_Male,
        request.Loyalty_Member_Yes,
        request.Product_Type_Laptop,
        request.Product_Type_Smartphone,
        request.Product_Type_Smartwatch,
        request.Product_Type_Tablet,
        request.Payment_Method_Cash,
        request.Payment_Method_Credit_Card,
        request.Payment_Method_Debit_Card,
        request.Payment_Method_PayPal,
        request.Payment_Method_Paypal,
        request.Shipping_Type_Express,
        request.Shipping_Type_Overnight,
        request.Shipping_Type_Same_Day,
        request.Shipping_Type_Standard
    ])
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)[0]
    return {"prediction": int(prediction)}
