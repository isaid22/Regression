# ATM Cash Demand: GARCH-X Volatility Engine

This repository contains the implementation of a **GARCH-X** (Generalized Autoregressive Conditional Heteroskedasticity with Exogenous Regressors) model designed to forecast 35 days of banknote demand and risk for a fleet of ATMs.

## 1. Project Overview
The core challenge in ATM cash management is not just predicting the *average* withdrawal, but the **uncertainty** (risk). This project uses GARCH to calculate dynamic safety stocks, ensuring the 2026 World Cup and other major events don't trigger "Out of Cash" (OOC) events.

## 2. Technical Framework
We utilize a two-stage model architecture implemented in `fit-garch.py`:

### The Mean Model (Regression)
We account for predictable weekly seasonality using indicator variables for days of the week ($t-1 \dots t-6$).
$$y_t = \mu + \sum \gamma_i \text{Day}_i + \epsilon_t$$

### The Volatility Model (GARCH 1,1)
The residuals ($\epsilon$) are modeled to determine the current risk regime.
$$\sigma^2_t = \omega + \alpha \epsilon^2_{t-1} + \beta \sigma^2_{t-1}$$

---

## 3. Key Findings from `fit-garch.py`
During the fitting process, we discovered that omitting seasonality causes the model to over-estimate baseline chaos. By adding exogenous features, we achieved:
* **Alpha ($\alpha$):** ~0.15 (High sensitivity to recent withdrawal shocks).
* **Beta ($\beta$):** ~0.70 (Strong persistence of risk clustering).
* **Mean Reversion:** Volatility successfully returns to the long-run average over the 35-day forecast horizon.

---

## 4. Usage: `fit-garch.py`

### Prerequisites
```bash
pip install pandas numpy matplotlib arch