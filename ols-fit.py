import numpy as np
from sklearn.datasets import load_diabetes

# Load data
data = load_diabetes()
X = data.data
y = data.target

# Select your features (by index)
feature_names = data.feature_names
selected_features = ["s1", "s5", "bmi", "bp", "s2", "sex", "s4", "age", "s6"]
indices = [feature_names.index(f) for f in selected_features]

X_sel = X[:, indices]

# Add intercept column
X_design = np.column_stack([np.ones(X_sel.shape[0]), X_sel])

# OLS solution
XtX = X_design.T @ X_design
XtX_inv = np.linalg.inv(XtX)
beta = XtX_inv @ X_design.T @ y

# Residuals
y_hat = X_design @ beta
residuals = y - y_hat

n, p = X_design.shape
sigma2 = (residuals @ residuals) / (n - p)

# Covariance matrix
cov_beta = sigma2 * XtX_inv

print("Beta:", beta)
print("Covariance matrix:\n", cov_beta)