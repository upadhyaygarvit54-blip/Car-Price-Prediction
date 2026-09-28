"""
Car Price Prediction using Regression
-------------------------------------
Dataset: car data.csv

Workflow:
1. Load data
2. Create Car_Age and Brand features
3. Preprocess numerical/categorical features
4. Train regression models
5. Evaluate using MAE, RMSE and R2
6. Plot results and feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("car data.csv")

# Feature engineering
df["Car_Age"] = 2026 - df["Year"]
df["Brand"] = df["Car_Name"].str.split().str[0].str.title()

X = df.drop(columns=["Selling_Price", "Car_Name", "Year"])
y = df["Selling_Price"]

numeric_features = ["Present_Price", "Driven_kms", "Owner", "Car_Age"]
categorical_features = ["Fuel_Type", "Selling_type", "Transmission", "Brand"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=2, random_state=42
    )
}

results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    print(f"\n{name}")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R2   : {r2:.3f}")

print("\nNote: This dataset does not contain a direct brand-goodwill field.")
print("Brand is used only as a proxy extracted from Car_Name.")
