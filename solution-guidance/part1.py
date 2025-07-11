from cslib import fetch_ts
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from cslib import engineer_features

data_dir = "cs-train" 
ts_data = fetch_ts(data_dir, clean=True)

ts_uk = ts_data["united_kingdom"]

print(ts_uk.head())
print(ts_uk.describe())

plt.figure(figsize=(14,6))
plt.plot(ts_uk['date'], ts_uk['revenue'], color='royalblue')
plt.title("Daily Revenue - United Kingdom")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()

sns.pairplot(ts_uk[['revenue', 'purchases', 'unique_invoices', 'total_views']])
X, y, dates = engineer_features(ts_uk, training=True)

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("First few dates:", dates[:5])

X.to_csv("features.csv", index=False)
pd.DataFrame(y, columns=["target"]).to_csv("target.csv", index=False)
