from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("../models/model.pkl")
scaler = joblib.load("../models/scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = np.array(request.json["features"]).reshape(1, -1)
    data = scaler.transform(data)
    pred = model.predict(data)
    return jsonify({"prediction": int(pred[0])})

if __name__ == "__main__":
    app.run(debug=True)