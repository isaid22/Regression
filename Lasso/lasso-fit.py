import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoLarsCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load well-known open-source dataset
X, y = load_diabetes(return_X_y=True, as_frame=True)

print("Features:")
print(X.columns.tolist())
print()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Lasso with alpha chosen by cross-validation using LARS path
model = make_pipeline(
    StandardScaler(),
    LassoLarsCV(cv=5)
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Get fitted LassoLarsCV step
lasso_cv = model.named_steps["lassolarscv"]

print("Chosen alpha:", lasso_cv.alpha_)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", rmse)
print("Test R^2:", r2_score(y_test, y_pred))
print()

# Save trained model pipeline to disk
joblib.dump(model, "lasso_diabetes_pipeline.joblib")


# Show coefficients
coef = pd.Series(lasso_cv.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
print("Coefficients:")
print(coef)
print()

print("Selected features (non-zero coefficients):")
print(coef[coef != 0])


## Inference

# The Lasso regression model has identified the most relevant features for predicting the target variable in the diabetes dataset. The chosen alpha value indicates the level of regularization applied, which helps to prevent overfitting by shrinking some coefficients to zero. The non-zero coefficients indicate the features that have a significant impact on the target variable, while the zero coefficients suggest that those features are not contributing to the model's predictions. The performance metrics (RMSE and R^2) provide insight into how well the model is performing on the test set, with a lower RMSE and higher R^2 indicating better predictive accuracy.

x_new = X_test.iloc[0:5]  # New data for prediction (first 5 samples from test set)
y_new_pred = model.predict(x_new)
print("Predictions for new data:")
print(y_new_pred)