from flask import Flask, request, jsonify
import pandas as pd
from ml import run_classification, run_regression

app = Flask(__name__)

@app.route("/")
def home():
    return "ML API Running"


@app.route("/run-model", methods=["POST"])
def run_model():

    file = request.files["file"]
    df = pd.read_csv(file)

    target = request.form["target"]
    features = request.form["features"].split(",")
    test_size = float(request.form["test_size"])
    model_type = request.form["model_type"]

    if model_type == "classification":

        acc, cm = run_classification(df, target, features, test_size)

        return jsonify({
            "accuracy": acc,
            "confusion_matrix": cm.tolist()
        })

    else:

        mse, r2 = run_regression(df, target, features, test_size)

        return jsonify({
            "mse": mse,
            "r2": r2
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)