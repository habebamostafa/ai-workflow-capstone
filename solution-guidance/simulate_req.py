# simulate_requests.py
from model import model_predict
import json

country = "spain"
year = "2018"
month = "01"

results = []

for day in range(1, 29):  # simulate for 28 days
    day_str = str(day).zfill(2)
    try:
        result = model_predict(country, year, month, day_str)
        print(f"{year}-{month}-{day_str} → {result['y_pred']}")
        results.append({
            "date": f"{year}-{month}-{day_str}",
            "y_pred": result["y_pred"][0] if result["y_pred"] else None
        })
    except Exception as e:
        print(f"{year}-{month}-{day_str} → Error:", str(e))

with open("predicted_values.json", "w") as f:
    json.dump(results, f, indent=4)
