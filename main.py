import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("WineQT.csv")

# 1.1 Data Exploration
"""
print("First 5 rows of dataset:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nSummary Statistics:\n")
print(df.describe().sort_values(by="std", axis=1, ascending=False))
"""


#1.2 Corralation Analysis
"""
corr = df.corr()
print(corr)
plt.figure(figsize=(12, 8))
plt.title("Correlation Matrix")
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
"""
"""
quality_corr = df.corr()["quality"].drop("quality")

strongest_pos = quality_corr.idxmax(), quality_corr.max()
strongest_neg = quality_corr.idxmin(), quality_corr.min()

print("Strongest positive correlation with quality:", strongest_pos)
print("Strongest negative correlation with quality:", strongest_neg)

plt.show()
"""

# 1.3 Linear Regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("WineQT.csv")
X = df["chlorides"].values
y = df["quality"].values

# Add bias term (intercept)
X_b = np.c_[np.ones((len(X), 1)), X]

# -----------------------------
# 2. Normal Equation Solution (for comparison)
# -----------------------------
XTX = X_b.T.dot(X_b)
XTX_inv = np.linalg.inv(XTX)
XTy = X_b.T.dot(y)
theta_best = XTX_inv.dot(XTy)

print("Optimal parameters (Normal Equation):", theta_best)

# -----------------------------
# 3. Gradient Descent
# -----------------------------
alpha = 0.01        # learning rate
iterations = 1000   # number of steps

theta = np.random.randn(2)  # random initialization
cost_history = []

for _ in range(iterations):
    error = y - X_b.dot(theta)
    cost = np.mean(error**2)
    cost_history.append(cost)
    
    gradients = -2/len(X) * X_b.T.dot(error)
    theta -= alpha * gradients

print("Final parameters (Gradient Descent):", theta)
"""

import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("WineQT.csv")

# Feature (X) and target (y)
X = df["chlorides"].values
y = df["quality"].values

# Normalize features
X = (X - X.mean()) / X.std()
y = (y - y.mean()) / y.std()   # normalize y too!

# Parameters
m = 0.0
b = 0.0
learning_rate = 0.001   # smaller learning rate
epochs = 1000
n = len(X)

# Gradient descent
for _ in range(epochs):
    y_pred = m * X + b
    error = y_pred - y

    # Gradients
    dm = (2/n) * np.dot(error, X)
    db = (2/n) * np.sum(error)

    # Update step
    m -= learning_rate * dm
    b -= learning_rate * db

print(f"Final model: normalized_quality = {m:.4f} * normalized_chlorides + {b:.4f}")


