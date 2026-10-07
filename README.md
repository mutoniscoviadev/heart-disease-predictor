# Heart Disease Risk Predictor

Deep learning project (ITLPA701, Python and Fundamentals of AI, CAT2).
A feedforward neural network (FNN) built with TensorFlow/Keras that predicts whether a patient is likely to have heart disease from clinical measurements. It has a Gradio web demo.

## Data
Kaggle Heart Failure Prediction dataset (`data/heart.csv`): 918 patients, 11 inputs and the target `HeartDisease`.

## Method
- Preprocessing: removed impossible zeros (RestingBP, Cholesterol), one-hot encoding, 80/20 train/test split, StandardScaler.
- Model: Dense 32 (ReLU) > Dropout 0.3 > Dense 16 (ReLU) > Dropout 0.2 > Dense 1 (sigmoid), Adam, early stopping.
- Decision threshold lowered from 0.5 to 0.4 to miss fewer sick patients.

## Results (184 test patients, threshold 0.4)
Accuracy 0.891, Precision 0.902, Recall 0.902, F1-score 0.902.

## Files
- `heart_disease.ipynb`: full notebook (data, preprocessing, model, evaluation, responsible use)
- `app.py`: Gradio demo
- `models/`: saved model, scaler, feature columns, cholesterol median and config
- `requirements.txt`: packages

## Run the demo
    pip install -r requirements.txt
    python app.py

## Responsible use
A screening aid for learning, not a medical diagnosis. The model learned from only 918 patients aged 28 to 77, so it may be less reliable for other groups. A doctor must always confirm the result.