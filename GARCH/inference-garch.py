import pickle
import pandas as pd
import numpy as np

# 1. Load the saved fitted model
with open("garch_result.pkl", "rb") as f:
    res = pickle.load(f)
print("Model loaded successfully.")

# 2. Define the forecast horizon and generate future dates
# Change last_date and horizon as needed
last_date = pd.Timestamp("today").normalize() - pd.Timedelta(days=1)
horizon = 35

future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

# 3. Build the regressor matrix (must match training columns exactly)
from features import build_features
future_x = build_features(future_dates)

x_forecast_map = {col: future_x[col].values for col in future_x.columns}

# 4. Run forecast
forecasts = res.forecast(horizon=horizon, x=x_forecast_map, reindex=False)

# 5. Extract results
mean_f = forecasts.mean.iloc[-1].values
vol_f = np.sqrt(forecasts.variance.iloc[-1].values)
upper_95 = mean_f + 1.645 * vol_f

# 6. Output
results_df = pd.DataFrame({
    "date": future_dates,
    "mean_demand": mean_f.round(0).astype(int),
    "volatility": vol_f.round(0).astype(int),
    "upper_95": upper_95.round(0).astype(int),
})
results_df.set_index("date", inplace=True)
print(results_df)
