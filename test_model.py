import joblib
import pandas as pd

def test_prediction():
    print("Loading model for testing...")
    model = joblib.load("rf_model.pkl")
    
    # Create a fake sensor reading
    test_data = pd.DataFrame([[0.5, -1.2]], columns=["Feature_0", "Feature_1"])
    
    # Get prediction
    prediction = model.predict(test_data)[0]
    
    # Assert that the prediction is either 0 or 1
    assert prediction in [0, 1], "Error: Model returned an invalid class!"
    print("Test Passed! Model predicted successfully.")

if __name__ == "__main__":
    test_prediction()
