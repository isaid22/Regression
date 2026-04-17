import pandas as pd
import numpy as np
import argparse
import os

def generate_atm_data(n_days, base_demand, omega, alpha, beta):
    """Generates a synthetic GARCH(1,1) dataset for one ATM."""
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq='D')
    
    # 1. Weekly Seasonality (Mean Equation)
    # Demand spikes on Friday (4) and Saturday (5)
    dow_effect = {0: 0.8, 1: 0.9, 2: 1.0, 3: 1.1, 4: 1.5, 5: 1.6, 6: 1.0}
    
    # 2. Volatility Simulation (GARCH Equation)
    vols = np.zeros(n_days)
    shocks = np.zeros(n_days)
    vols[0] = np.sqrt(omega / (1 - alpha - beta)) # Unconditional variance start
    
    for t in range(1, n_days):
        # sigma^2_t = omega + alpha*epsilon^2_{t-1} + beta*sigma^2_{t-1}
        vols[t] = np.sqrt(omega + alpha * (shocks[t-1]**2) + beta * (vols[t-1]**2))
        shocks[t] = np.random.normal(0, vols[t])
    
    # 3. Combine Mean + Volatility
    demand = []
    for i, date in enumerate(dates):
        # Apply seasonality to the mean, then add the GARCH shock
        seasonal_mean = base_demand * dow_effect[date.weekday()]
        daily_val = seasonal_mean + shocks[i]
        demand.append(max(0, int(daily_val))) # ATM counts cannot be negative
        
    df = pd.DataFrame({
        'date': dates,
        'demand': demand,
        'actual_vol': vols # Saving the 'truth' for learning/validation
    })
    return df

def main():
    # Setup Argument Parser for User Input
    parser = argparse.ArgumentParser(description="Generate GARCH ATM Demand Data")
    
    parser.add_argument("--days", type=int, default=730, help="Number of days to simulate")
    parser.add_argument("--mean", type=int, default=200, help="Base daily bill count")
    parser.add_argument("--alpha", type=float, default=0.1, help="ARCH parameter (reaction to shocks)")
    parser.add_argument("--beta", type=float, default=0.8, help="GARCH parameter (persistence of vol)")
    parser.add_argument("--output", type=str, default="atm_demand_data.csv", help="Output CSV filename")

    args = parser.parse_args()

    # Derived parameter (baseline variance)
    omega = 0.5 

    print(f"--- Generating data for {args.days} days ---")
    df = generate_atm_data(args.days, args.mean, omega, args.alpha, args.beta)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Success! Data saved to {os.path.abspath(args.output)}")
    print(df.head(10))

if __name__ == "__main__":
    main()