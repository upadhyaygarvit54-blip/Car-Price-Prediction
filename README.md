# Car Price Prediction using Machine Learning

## Project Objective
Build a regression model that predicts the selling price of a used car from real-world vehicle features.

### Tasks covered
- Collect/use car-related features such as present price, driven kilometres, fuel type, transmission and ownership.
- Perform data preprocessing.
- Engineer `Car_Age` and a `Brand` proxy.
- Train regression models.
- Evaluate models using MAE, RMSE and R².
- Visualize the price distribution, predictions, model comparison and feature importance.

## Dataset
The supplied `car data.csv` contains **301 rows and 9 original columns**:
`Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission`, and `Owner`.

There are no missing values.

> The dataset does not have a direct brand-goodwill score. The project therefore derives a simple `Brand` proxy from the first word of `Car_Name`. This should be described as a proxy, not as a measured goodwill value.

## Feature Engineering
- `Car_Age = 2026 - Year`
- `Brand = first word of Car_Name`

The original `Year` and `Car_Name` are then excluded from the model input.

## Models
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Categorical variables are one-hot encoded and numerical variables are standardized using a Scikit-learn `ColumnTransformer` pipeline.

## Test Results
Using an 80/20 train-test split with `random_state=42`:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1.064 | 1.632 | 0.884 |
| Random Forest | 0.610 | 0.920 | 0.963 |
| Gradient Boosting | 0.798 | 1.184 | 0.939 |

The metrics are specific to this train/test split and should not be interpreted as a guarantee of performance on new markets or future car listings.

## How to Run

```bash
pip install -r requirements.txt
python car_price_prediction.py
```

## Project Files
- `car data.csv` — input dataset
- `car_price_prediction.py` — complete ML workflow
- `model_results.csv` — evaluation results
- `feature_importance.csv` — Random Forest feature importance
- `price_distribution.png` — target distribution
- `actual_vs_predicted.png` — prediction plot
- `model_comparison.png` — RMSE comparison
- `feature_importance.png` — top feature importance
- `requirements.txt` — Python dependencies

## Real-World Application
Used-car platforms can use regression models to estimate listing prices from vehicle age, current market value, mileage, fuel type, transmission and ownership history. Such systems can support price estimation, inventory analysis and pricing decisions, while requiring regular retraining as market conditions change.
