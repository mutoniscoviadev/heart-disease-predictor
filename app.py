import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from tensorflow import keras

# Ages the model learned from. Set these to the real minimum and maximum of your data.
AGE_MIN = 28
AGE_MAX = 77

# Load everything saved from the notebook
model = keras.models.load_model("models/heart_model.keras")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
chol_median = joblib.load("models/chol_median.pkl")
with open("models/config.json") as f:
    THRESHOLD = json.load(f)["threshold"]


def predict(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
            resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    if age is None or age <= 0:
        return None, "Please enter a valid age (greater than 0)."

    # Cholesterol 0 means "unknown": use the training median, as in the notebook
    if cholesterol == 0:
        cholesterol = chol_median

    # Build one row with the same 15 columns used in training
    row = {col: 0 for col in feature_columns}
    row["Age"] = age
    row["RestingBP"] = resting_bp
    row["Cholesterol"] = cholesterol
    row["FastingBS"] = int(fasting_bs)
    row["MaxHR"] = max_hr
    row["Oldpeak"] = oldpeak

    one_hot = {
        f"Sex_{sex}": 1,
        f"ChestPainType_{chest_pain}": 1,
        f"RestingECG_{resting_ecg}": 1,
        f"ExerciseAngina_{exercise_angina}": 1,
        f"ST_Slope_{st_slope}": 1,
    }
    for key, value in one_hot.items():
        if key in row:          # the dropped baseline categories stay 0
            row[key] = value

    X = pd.DataFrame([row])[feature_columns]
    X_scaled = scaler.transform(X)
    prob = float(model.predict(X_scaled, verbose=0).ravel()[0])

    if prob >= THRESHOLD:
        label = "Heart disease likely"
        advice = ("The measurements entered look similar to patients who had heart "
                  "disease. Please see a doctor for a proper check-up and tests.")
    else:
        label = "Heart disease unlikely"
        advice = ("The measurements entered look similar to patients without heart "
                  "disease. This does not rule it out completely, so please see a "
                  "doctor if you have symptoms such as chest pain or shortness of breath.")

    note = (f"{label}\n"
            f"Estimated probability: {prob:.1%}\n"
            f"Decision threshold: {THRESHOLD}\n\n"
            f"{advice}\n\n")

    if age < AGE_MIN or age > AGE_MAX:
        note += (f"Warning: the model learned from patients aged {AGE_MIN} to {AGE_MAX}, "
                 f"so this result is less reliable for age {age:g}.\n\n")

    note += "This is a screening aid for learning, not a medical diagnosis."
    return {"Disease": prob, "No disease": 1 - prob}, note


interface_args = dict(
    fn=predict,
    inputs=[
        gr.Number(value=50, label="Age (years)"),
        gr.Radio(["M", "F"], value="M", label="Sex"),
        gr.Dropdown(["ASY", "ATA", "NAP", "TA"], value="ASY", label="Chest pain type"),
        gr.Number(value=120, label="Resting blood pressure (mm Hg)"),
        gr.Number(value=200, label="Cholesterol (mg/dL, 0 if unknown)"),
        gr.Radio([0, 1], value=0, label="Fasting blood sugar > 120 mg/dL (1 = yes)"),
        gr.Dropdown(["Normal", "ST", "LVH"], value="Normal", label="Resting ECG"),
        gr.Number(value=150, label="Maximum heart rate"),
        gr.Radio(["N", "Y"], value="N", label="Exercise-induced angina"),
        gr.Number(value=0.0, label="Oldpeak (ST depression)"),
        gr.Dropdown(["Up", "Flat", "Down"], value="Up", label="ST slope"),
    ],
    outputs=[gr.Label(label="Prediction"), gr.Textbox(label="Result")],
    title="Heart Disease Risk Predictor",
    description="FNN trained on the Heart Failure Prediction dataset (918 patients). "
                "For learning purposes only, not for medical use.",
)

# Switch off the Flag button (the setting name depends on the Gradio version)
try:
    demo = gr.Interface(**interface_args, flagging_mode="never")
except TypeError:
    demo = gr.Interface(**interface_args, allow_flagging="never")

if __name__ == "__main__":
    demo.launch()