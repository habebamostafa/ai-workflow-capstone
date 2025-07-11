import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

with open("./predicted_values.json", "r") as f:
    preds = json.load(f)

pred_df = pd.DataFrame(preds)
pred_df["date"] = pd.to_datetime(pred_df["date"])
pred_df.set_index("date", inplace=True)

df_actual = pd.read_csv("cs-train/ts-data/ts-spain.csv")
df_actual["date"] = pd.to_datetime(df_actual["date"])
df_actual.set_index("date", inplace=True)

df_compare = df_actual[["revenue"]].join(pred_df, how="inner")
df_compare.columns = ["actual", "predicted"]

rmse = np.sqrt(mean_squared_error(df_compare["actual"], df_compare["predicted"]))
print(f" RMSE: {rmse:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(df_compare.index, df_compare["actual"], label="Actual", marker="o")
plt.plot(df_compare.index, df_compare["predicted"], label="Predicted", marker="x")
plt.title("Actual vs Predicted Revenue")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
