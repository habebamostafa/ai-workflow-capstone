from flask import Flask, request, jsonify
from model import model_predict, model_train
import os

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Revenue Forecasting API is running!"})

@app.route("/predict", methods=["GET"])
def predict():
    try:
        country = request.args.get("country")
        year = request.args.get("year")
        month = request.args.get("month")
        day = request.args.get("day")

        result = model_predict(country, year, month, day)

        y_pred = result["y_pred"].tolist() if hasattr(result["y_pred"], "tolist") else result["y_pred"]
        y_proba = result["y_proba"].tolist() if hasattr(result["y_proba"], "tolist") else result["y_proba"]

        return jsonify({
            "y_pred": y_pred,
            "y_proba": y_proba
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/train", methods=["GET"])
def train():
    try:
        data_dir = os.path.join("..", "data", "cs-train")
        model_train(data_dir, test=False)
        return jsonify({"status": "Training complete"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
