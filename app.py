from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from utils import load_data

app = Flask(__name__)
CORS(app)

# Load encoders & data
(_, _, _, _, X_train_r, X_test_r, y_train_r, y_test_r, encoders) = load_data(return_mapping=True)

# Load all classification models
models = {
    "KNN": joblib.load("models/KNN_model.pkl"),
    "SVM": joblib.load("models/SVM_model.pkl"),
    "DecisionTree": joblib.load("models/DecisionTree_model.pkl"),
    "RandomForest": joblib.load("models/RandomForest_model.pkl")
}

# Load regression model
regressor = joblib.load("models/regressor_model.pkl")


def encode_value(column_name, raw_input):
    le = encoders.get(column_name)
    if le is None:
        return 0
    try:
        return int(le.transform([raw_input])[0])
    except:
        return 0


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Encode categorical inputs
    crop_enc = encode_value("Crop", data.get("Crop"))
    region_enc = encode_value("Region", data.get("Region"))

    # Create row for prediction
    X_new = pd.DataFrame([[
        float(data["Rainfall_mm"]),
        float(data["Temperature_Celsius"]),
        int(data["Irrigation_Used"]),
        int(data["Fertilizer_Used"]),
        crop_enc,
        region_enc
    ]], columns=[
        'Rainfall_mm',
        'Temperature_Celsius',
        'Irrigation_Used',
        'Fertilizer_Used',
        'Crop_enc',
        'Region_enc'
    ])

    # MODEL PREDICTIONS
    prob_dict = {}
    class_mapping = {0: "Low", 1: "Medium", 2: "High"}
    class_votes = []

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_new)[0]
            prob_dict[name] = {
                "Low": float(probs[0]),
                "Medium": float(probs[1]),
                "High": float(probs[2])
            }
            class_votes.append(np.argmax(probs))
        else:
            pred = model.predict(X_new)[0]
            class_votes.append(pred)
            prob_dict[name] = {"Low": None, "Medium": None, "High": None}

    # Majority Vote Final Prediction
    values, counts = np.unique(class_votes, return_counts=True)
    final_class = class_mapping[int(values[np.argmax(counts)])]

    # Regression exact yield
    exact_yield = float(regressor.predict(X_new)[0])

    return jsonify({
        "exact_yield": exact_yield,
        "predicted_class": final_class,
        "model_probabilities": prob_dict,
        "model_accuracy": [
            {"model": k, "accuracy": None} for k in models.keys()
        ]
    })


if __name__ == "__main__":
    app.run(debug=True)
