import pandas as pd
import numpy as np


def _payday_dates(date_range):
    """Return a set of Timestamps for the 1st business day, 15th business day,
    and last business day of every month covered by date_range."""
    paydays = set()
    months = pd.period_range(date_range.min(), date_range.max(), freq='M')
    for month in months:
        # 1st business day of month
        d = month.to_timestamp()
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        paydays.add(d.normalize())

        # 15th (or next business day if it falls on a weekend)
        d = month.to_timestamp().replace(day=15)
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        paydays.add(d.normalize())

        # Last business day of month
        d = month.to_timestamp(how='E')
        while d.weekday() >= 5:
            d -= pd.Timedelta(days=1)
        paydays.add(d.normalize())

    return paydays


def build_features(dates, holiday_window=7):
    """
    Build the exogenous regressor matrix for the ARX-GARCH model.

    Parameters
    ----------
    dates : pd.DatetimeIndex
    holiday_window : int
        How many days around a US holiday the proximity feature covers.
        On the holiday itself the value equals holiday_window; it drops by 1
        for each day further away, reaching 0 beyond the window.

    Returns
    -------
    pd.DataFrame with columns:
        day_1 .. day_6      — day-of-week dummies (Monday is the dropped baseline)
        is_payday           — 1 on the 1st/15th/last business day of the month
        holiday_proximity   — ramps from 0 (far from holiday) to holiday_window
                              (on the holiday itself)

    Note: install the holidays package with  pip install holidays
    """
    try:
        import holidays as _holidays
        us = _holidays.US(years=range(dates.min().year, dates.max().year + 2))
        holiday_ts = sorted(pd.Timestamp(d) for d in us.keys())
        has_holidays = True
    except ImportError:
        has_holidays = False
        print("Warning: 'holidays' package not found. "
              "Install with: pip install holidays\n"
              "Skipping holiday_proximity feature.")

    df = pd.DataFrame(index=dates)

    # Day-of-week dummies (Monday = 0 = baseline, encode days 1-6)
    dow = dates.dayofweek
    for d in range(1, 7):
        df[f'day_{d}'] = (dow == d).astype(float)

    # Payday flag
    paydays = _payday_dates(dates)
    df['is_payday'] = dates.normalize().isin(paydays).astype(float)

    # Holiday proximity
    if has_holidays:
        proximity = []
        for date in dates:
            ts = date.normalize()
            future = next((h for h in holiday_ts if h >= ts), None)
            past = next((h for h in reversed(holiday_ts) if h <= ts), None)
            days_until = (future - ts).days if future is not None else holiday_window + 1
            days_since = (ts - past).days if past is not None else holiday_window + 1
            nearest = min(days_until, days_since)
            proximity.append(float(max(0, holiday_window - nearest)))
        df['holiday_proximity'] = proximity

    return df
