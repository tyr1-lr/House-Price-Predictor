import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

model = joblib.load("model/house_price_model.pkl")

data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = model.predict(X_test)

plt.figure(figsize=(6, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6,
    color="#E26A2C",
    edgecolors="white",
    linewidths=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="#E26A2C",
    linewidth=2
)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.grid(alpha=0.2)

plt.savefig("website/static/graph.png", bbox_inches="tight")

print("Graph saved successfully!")
