import pandas as pd
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# =========================================
# EXPERIMENT SETTINGS  (CHANGE ONLY THESE)
# =========================================
MODEL_TYPE = "lasso"   # "linear" or "lasso"
LASSO_ALPHA = 0.1       # used only if MODEL_TYPE="lasso"
TEST_SIZE = 0.3
USE_SCALER = True
# =========================================

os.makedirs("output", exist_ok=True)

# Load dataset
data = pd.read_csv("dataset/winequality.csv", sep=";")

# Split features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)

# Scaling
if USE_SCALER:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Select model
if MODEL_TYPE == "linear":
    model = LinearRegression()
elif MODEL_TYPE == "lasso":
    model = Lasso(alpha=LASSO_ALPHA)
else:
    raise ValueError("Invalid MODEL_TYPE")

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print (GitHub Actions reads this)
print(f"MODEL: {MODEL_TYPE}")
print(f"TEST_SIZE: {TEST_SIZE}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Save model
model_file = f"output/model_{MODEL_TYPE}_ts{TEST_SIZE}.pkl"
joblib.dump(model, model_file)

# Save results
results = {
    "Model": MODEL_TYPE,
    "Test_Size": TEST_SIZE,
    "MSE": mse,
    "R2": r2
}

results_file = f"output/results_{MODEL_TYPE}_ts{TEST_SIZE}.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)
