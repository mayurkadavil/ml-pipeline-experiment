import gradio as gr
import joblib
import pandas as pd

model = joblib.load("rf_model.pkl")

def predict(sensor_1, sensor_2):
    input_df = pd.DataFrame([[sensor_1, sensor_2]], columns=["Feature_0", "Feature_1"])
    prediction = model.predict(input_df)[0]
    return "Class 1 (High/Active)" if prediction == 1 else "Class 0 (Low/Inactive)"

ui = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label="Sensor 1", value=0.5), gr.Number(label="Sensor 2", value=-1.2)],
    outputs=gr.Text(label="Prediction"),
    title="My First ML App"
)
ui.launch()
