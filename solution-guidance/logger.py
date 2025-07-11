import os
import json
from datetime import datetime

def update_train_log(country, data_range, eval_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=False):
    log_entry = {
        "country": country,
        "data_range": data_range,
        "eval": eval_test,
        "runtime": runtime,
        "model_version": MODEL_VERSION,
        "model_version_note": MODEL_VERSION_NOTE,
        "timestamp": str(datetime.now())
    }

    log_file = "train-test-log.json" if test else "train-log.json"

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)


def update_predict_log(country, y_pred, y_proba, query_date, runtime, MODEL_VERSION, test=False):
    log_entry = {
        "country": country,
        "y_pred": y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
        "y_proba": y_proba.tolist() if y_proba is not None else None,
        "query_date": query_date,
        "runtime": runtime,
        "model_version": MODEL_VERSION,
        "timestamp": str(datetime.now())
    }

    log_file = "predict-test-log.json" if test else "predict-log.json"

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)
