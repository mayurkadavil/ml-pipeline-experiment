from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="My First ML API")

# Load the model we trained earlier
model = joblib.load("rf_model.pkl")

# Define what the input data should look like
class SensorInput(BaseModel):
    sensor_1: float
    sensor_2: float

@app.get("/")
def home():
    return {"message": "API is running! Go to /docs to test it."}

@app.post("/predict")
def predict(data: SensorInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([[data.sensor_1, data.sensor_2]], columns=["Feature_0", "Feature_1"])
    
    # Run the model
    prediction = model.predict(input_df)[0]
    result = "Class 1 (High)" if prediction == 1 else "Class 0 (Low)"
    
    return {"prediction": result, "sensor_1_received": data.sensor_1, "sensor_2_received": data.sensor_2}
