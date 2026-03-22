import joblib
import pandas as pd
from sklearn.datasets import load_diabetes

# 1. Load saved model artifact
model = joblib.load("lasso_diabetes_pipeline.joblib")

# 2. Example new input
X, y = load_diabetes(return_X_y=True, as_frame=True)
x_new = X.iloc[0:5]    # first 5 rows as DataFrame

# 3. Predict
y_pred = model.predict(x_new)

print("Input row:")
print(x_new)

print("\nPrediction:")
print(y_pred)