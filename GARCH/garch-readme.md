# ATM Cash Demand: Risk & Volatility Modeling

## 1. Project Architecture
The objective is to manage the inventory of 20,000 ATMs by modeling not just the expected volume of bills, but the **volatility** (risk) of that demand. This ensures that safety stock levels are calculated dynamically to prevent stock-outs during high-impact events.

---

## 2. Core Equations (LaTeX)

### Mean Equation (Expected Volume)
$$y_t = \mu + \gamma \mathbf{X}_t + \epsilon_t$$

### Volatility Equation (Risk)
$$\sigma^2_t = \omega + \alpha \epsilon^2_{t-1} + \beta \sigma^2_{t-1} + \delta \mathbf{X}_t$$

---

## 3. Selecting Lags: Understanding p and q
In a GARCH(p, q) model, the parameters $p$ and $q$ define how far back the model looks to determine today's risk.

### The ARCH term ($p$): Past Shocks
This controls the impact of unexpected events ($t-1, t-2, \dots$).
* **When to use $p > 1$:** If an ATM is near a location with multi-day "shocks" (e.g., a 3-day music festival), a higher $p$ captures the sustained high variance of that specific period.

### The GARCH term ($q$): Past Volatility
This controls the "memory" of the risk level itself.
* **When to use $q > 1$:** Rarely required for daily data. A $q=1$ value already captures an exponentially decaying memory of all past volatility. Only increase $q$ if you observe extremely slow transitions between high and low risk regimes.

### General Formula for Higher Orders
$$\sigma^2_t = \omega + \sum_{i=1}^{p} \alpha_i \epsilon^2_{t-i} + \sum_{j=1}^{q} \beta_j \sigma^2_{t-j}$$



---

## 4. Feature Engineering & Exogenous Variables
Major events (US Holidays, Sporting Events) are included as $X_t$ to adjust both the mean and the variance.

* **Fixed Events:** July 4th, Christmas.
* **Major 2026 Events:** Super Bowl LX, FIFA World Cup, The Masters.
* **Implementation:** Use binary flags for the event day and a "lead-up" flag for the 3 days prior to capture pre-event withdrawals.

---

## 5. Units of Measurement
* **Demand ($y_t$):** Discrete number of physical banknotes.
* **Actual Volatility ($\sigma_t$):** Standard deviation of the forecast error (Units: bills).
* **Variance ($\sigma^2_t$):** Predicted uncertainty (Units: $\text{bills}^2$).

---

## 6. Large-Scale Implementation (Spark)
For a fleet of 20,000 ATMs, use Apache Spark to parallelize the fitting process:
1. **Group by** `ATM_ID`.
2. **Apply** a GARCH(1,1) model via a Pandas UDF.
3. **Store** parameters ($\alpha, \beta, \omega$) for each ATM to generate localized 35-day risk forecasts.