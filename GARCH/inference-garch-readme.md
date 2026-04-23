# ATM Cash Demand: GARCH-X Inference Guide

This document explains how inference works in `inference-garch.py` — what inputs it needs, how they flow through the model, and why this differs fundamentally from neural network inference.

---

## 1. The Core Idea: No Historical Data Required at Inference Time

When you run `inference-garch.py`, you do **not** pass any historical ATM demand values. The model already "knows" the recent history because it was frozen inside `garch_result.pkl` at fitting time. The only thing you supply is a calendar — specifically, the features describing what each future date looks like.

This is unlike a feedforward neural network, where you must explicitly construct every input feature (including lags) and pass them as a numeric tensor. GARCH inference is closer to an **RNN with a frozen hidden state**: the terminal state from training is stored and the recursion simply continues forward from it.

| | GARCH-X (`inference-garch.py`) | Neural Network |
|---|---|---|
| Endogenous history | Baked into `.pkl` (frozen state) | Must be manually constructed and passed |
| What you provide | Exogenous calendar features only | All input features, including lags |
| Model statefulness | Stateful — stored $\varepsilon_T$, $\sigma^2_T$, $y_T$ | Stateless — pure function of inputs |

---

## 2. What Is Stored in `garch_result.pkl`

When `fit-garch.py` pickles `res`, it captures the complete fitted state:

- **Estimated parameters**: AR coefficient $\phi_1$, GARCH coefficients $\omega, \alpha, \beta$, and one regression weight per exogenous feature
- **Terminal residual** $\varepsilon_T$: the last in-sample prediction error
- **Terminal conditional variance** $\sigma^2_T$: the volatility level at the last training observation
- **Last observed demand** $y_T$: the anchor for the AR(1) forward recursion

These three terminal values are the "hidden state" that the forecast unrolls from.

---

## 3. The Input: Exogenous Features Only

The only input constructed at inference time is the exogenous regressor matrix — features that describe each future date but are **not** derivable from the demand series itself.

`build_features(future_dates)` in `features.py` produces a DataFrame with the following columns:

| Column | Type | Description |
|---|---|---|
| `day_1` … `day_6` | Binary (0/1) | Day-of-week dummies; Tuesday through Sunday (Monday is the baseline) |
| `is_payday` | Binary (0/1) | 1 on the 1st business day, 15th business day, and last business day of the month |
| `holiday_proximity` | Float (0–7) | Ramps up to 7 on a US public holiday; decreases by 1 for each day further away |

Example (35-row slice):

```
            day_1  day_2  day_3  day_4  day_5  day_6  is_payday  holiday_proximity
2026-04-23    0      0      1      0      0      0       0.0           0.0
2026-04-30    0      0      0      0      0      0       1.0           0.0   ← last biz day
2026-05-25    0      0      0      0      0      0       0.0           7.0   ← Memorial Day
```

### Why these features are sufficient

The GARCH-X mean equation is:

$$\hat{y}_t = \phi_1 \hat{y}_{t-1} + \sum_i \beta_i x_{i,t}$$

The AR(1) term $\phi_1 \hat{y}_{t-1}$ handles all endogenous momentum — it is equivalent to an exponentially weighted moving average of past demand. The $x_{i,t}$ terms shift the baseline for predictable calendar effects. Adding rolling means or standard deviations of past demand as extra columns would be **redundant** with what the AR term already captures.

---

## 4. Data Structures at Inference Time

### 4a. `future_dates` — `pd.DatetimeIndex`

```python
future_dates = pd.date_range(start='2026-04-23', periods=35)
# DatetimeIndex(['2026-04-23', '2026-04-24', ..., '2026-05-27'], dtype='datetime64[ns]', freq='D')
```

A sequence of 35 calendar dates. **No demand values** — purely positional anchors used to derive features. `dtype` is `datetime64[ns]`.

---

### 4b. `future_x` — `pd.DataFrame` (35 rows × 8 columns)

Returned by `build_features(future_dates)`. Each row is one future day; each column is a float.

```
type:  pd.DataFrame
shape: (35, 8)
index: DatetimeIndex (same as future_dates)
dtypes: all float64
```

Concrete example:

```
            day_1  day_2  day_3  day_4  day_5  day_6  is_payday  holiday_proximity
2026-04-23    0.0    0.0    1.0    0.0    0.0    0.0       0.0           0.0
2026-04-24    0.0    0.0    0.0    1.0    0.0    0.0       0.0           0.0
2026-04-27    1.0    0.0    0.0    0.0    0.0    0.0       0.0           0.0   ← Tuesday
2026-04-30    0.0    0.0    0.0    0.0    0.0    0.0       1.0           0.0   ← last biz day
2026-05-25    0.0    0.0    0.0    0.0    0.0    0.0       0.0           7.0   ← Memorial Day
```

Column semantics:

| Column | Dtype | Values | Meaning |
|---|---|---|---|
| `day_1` | float64 | 0 or 1 | Tuesday (Monday is the dropped baseline) |
| `day_2` | float64 | 0 or 1 | Wednesday |
| `day_3` | float64 | 0 or 1 | Thursday |
| `day_4` | float64 | 0 or 1 | Friday |
| `day_5` | float64 | 0 or 1 | Saturday |
| `day_6` | float64 | 0 or 1 | Sunday |
| `is_payday` | float64 | 0 or 1 | 1st / 15th / last business day of the month |
| `holiday_proximity` | float64 | 0.0–7.0 | 7 on a US holiday; decreases by 1 per day away; 0 beyond 7 days |

---

### 4c. `x_forecast_map` — `dict[str, np.ndarray]`

The `arch` library cannot consume a DataFrame directly, so `future_x` is converted to a dict of 1D numpy arrays:

```python
x_forecast_map = {col: future_x[col].values for col in future_x.columns}
```

```
type:  dict
keys:  8 strings (column names, must match training exactly)
values: np.ndarray of shape (35,), dtype float64
```

Concrete structure:

```python
{
  'day_1':             np.array([0., 0., 0., 1., 0., 0., 0., 0., ...]),  # len=35
  'day_2':             np.array([0., 0., 0., 0., 1., 0., 0., 0., ...]),
  'day_3':             np.array([1., 0., 0., 0., 0., 1., 0., 0., ...]),
  'day_4':             np.array([0., 1., 0., 0., 0., 0., 1., 0., ...]),
  'day_5':             np.array([0., 0., 1., 0., 0., 0., 0., 1., ...]),
  'day_6':             np.array([0., 0., 0., 0., 0., 0., 0., 0., ...]),
  'is_payday':         np.array([0., 0., 0., 0., 0., 0., 0., 1., ...]),
  'holiday_proximity': np.array([0., 0., 0., 0., 0., 0., 0., 0., ...]),
}
```

At forecast step `t`, the `arch` library reads index `t` from **every array simultaneously** and plugs those 8 values into the mean equation for that step. This is why all arrays must have the same length as `horizon`.

**Constraints:**
- Keys must exactly match the column names in `train_x` used during fitting (same names, same order)
- Array length must equal the `horizon` argument passed to `res.forecast()`

> **Important:** If you add or rename a column in `features.py`, you must refit the model. A column mismatch between the pkl and `x_forecast_map` will raise a shape error at runtime.

---

### 4d. `forecasts.mean` and `forecasts.variance` — `pd.DataFrame` (1 row × 35 columns)

The raw output from `res.forecast()` has an unusual shape:

```
type:  pd.DataFrame
shape: (1, 35)   ← 1 row (because reindex=False), 35 columns (one per horizon step)
columns: ['h.01', 'h.02', ..., 'h.35']
```

`.iloc[-1].values` extracts the single row as a flat numpy array of 35 floats, which is what gets used to build the output table.

---

## 5. How the Forecast Recursion Works

Given the stored terminal state and the `x_forecast_map`, the library performs two coupled recursions across the 35-step horizon:

**Mean recursion (ARX):**
$$\hat{y}_{T+h} = \phi_1 \hat{y}_{T+h-1} + \sum_i \beta_i x_{i,T+h}$$

Starts from $y_T$ (stored). Each step's predicted demand feeds the next as the AR input.

**Variance recursion (GARCH 1,1):**
$$\hat{\sigma}^2_{T+h} = \omega + \alpha \hat{\varepsilon}^2_{T+h-1} + \beta \hat{\sigma}^2_{T+h-1}$$

Starts from $\varepsilon_T$ and $\sigma^2_T$ (stored). Because true future residuals are unknown, the library substitutes the expected squared residual $\hat{\sigma}^2_{T+h-1}$ at each step. This causes variance forecasts to decay toward the long-run mean:

$$\sigma^2_{\infty} = \frac{\omega}{1 - \alpha - \beta}$$

---

## 6. Outputs

| Output column | Formula | Interpretation |
|---|---|---|
| `mean_demand` | $\hat{y}_{T+h}$ | Expected number of $20 bills withdrawn |
| `volatility` | $\sqrt{\hat{\sigma}^2_{T+h}}$ | Conditional standard deviation — current risk level |
| `upper_95` | $\hat{y}_{T+h} + 1.645 \cdot \hat{\sigma}_{T+h}$ | 95th percentile demand — recommended replenishment target |

---

## 7. Usage

```bash
# Activate the virtual environment
source ~/ai_venv/bin/activate

# One-time: fit and save the model
python fit-garch.py

# Run inference at any time without refitting
python inference-garch.py
```

### When to refit

| Situation | Action |
|---|---|
| Routine daily/weekly forecast | Run `inference-garch.py` only |
| New historical data available | Refit with `fit-garch.py` |
| `features.py` columns changed | Refit with `fit-garch.py` |
| Unexpected demand shock (e.g. bank promotion) | Refit with `fit-garch.py` |
