import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# 1. Load your existing data
df = pd.read_csv("atm_demand_data.csv", index_col='date', parse_dates=True)

# 2. Re-create the training features (Monday is the baseline)
df['dow'] = df.index.dayofweek
train_x = pd.get_dummies(df['dow'], prefix='day', drop_first=True).astype(float)

# 3. Fit the model (The GARCH-X setup that worked well for you)
model = arch_model(df['demand'], x=train_x, mean='ARX', vol='Garch', p=1, q=1)
res = model.fit(disp='off')

# 4. Generate the FUTURE dates and features (35 days ahead)
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=35)
future_dow = future_dates.dayofweek

# This ensures the future 'x' matches the training 'x' columns exactly
future_x = pd.DataFrame(index=future_dates)
for d in range(1, 7): # Tuesday (1) through Sunday (6)
    future_x[f'day_{d}'] = (future_dow == d).astype(float)

# 5. Execute Forecast
# The library is very specific: with multiple x variables, it often wants 
# a dictionary where keys are the column names and values are the future paths.
x_forecast_map = {}
for col in train_x.columns:
    x_forecast_map[col] = future_x[col].values

# We tell it to forecast starting from the last date in our data
forecasts = res.forecast(horizon=35, x=x_forecast_map, reindex=False)

# 6. Extract Mean and Volatility
# Since we used reindex=False, the results are in the last row of the returned DataFrames
mean_f = forecasts.mean.iloc[-1].values
vol_f = np.sqrt(forecasts.variance.iloc[-1].values)

# 7. Calculate "Safety Stock" (95th Percentile)
# 1.645 is the Z-score for a 95% one-tailed confidence interval
upper_95 = mean_f + 1.645 * vol_f

# 8. Visualization
plt.figure(figsize=(12, 6))
plt.plot(future_dates, mean_f, label='Expected Demand (Mean)', color='#2980b9', lw=2)
plt.fill_between(future_dates, mean_f, upper_95, color='#f39c12', alpha=0.3, label='95% Safety Buffer (Risk)')
plt.title("ATM Replenishment Strategy: 35-Day Risk-Adjusted Forecast", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Number of $20 Bills")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("atm_forecast_plot.png")
print("Plot saved successfully as atm_forecast_plot.png")