"""
Synthetic weekly marketing spend + sales dataset.
Simulates a CPG brand with 4 channels and adstock carry-over effects.
"""
import numpy as np
import pandas as pd


CHANNELS = ["tv", "digital", "ooh", "instore_promo"]

# True ROI per channel (dollars of revenue per dollar spent)
TRUE_ROI = {"tv": 2.1, "digital": 3.5, "ooh": 2.4, "instore_promo": 3.6}

# Adstock decay rates (how quickly the effect wears off)
TRUE_DECAY = {"tv": 0.7, "digital": 0.3, "ooh": 0.5, "instore_promo": 0.2}


def geometric_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    adstocked = np.zeros_like(spend, dtype=float)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay * adstocked[t - 1]
    return adstocked


def generate_marketing_data(n_weeks: int = 156, seed: int = 42) -> pd.DataFrame:
    """156 weeks = 3 years of weekly data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    df = pd.DataFrame({"week": dates})

    # Seasonality
    t = np.arange(n_weeks)
    df["seasonality"] = 1 + 0.25 * np.sin(2 * np.pi * t / 52) + 0.1 * np.sin(4 * np.pi * t / 52)
    df["trend"] = 1 + 0.002 * t

    # Channel spend (weekly, in $000s)
    spend_levels = {"tv": (200, 80), "digital": (150, 60), "ooh": (100, 40), "instore_promo": (50, 20)}
    for ch, (mean, std) in spend_levels.items():
        raw = rng.normal(mean, std, n_weeks).clip(0)
        # Quarterly TV flights
        if ch == "tv":
            flight = np.tile([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_weeks)[:n_weeks]
            raw *= (0.3 + 0.7 * flight)
        df[f"spend_{ch}"] = raw.round(1)

    # Adstock-transformed spend
    base_sales = 1_000  # base weekly revenue ($000s)
    sales = df["seasonality"] * df["trend"] * base_sales

    for ch in CHANNELS:
        adstocked = geometric_adstock(df[f"spend_{ch}"].values, TRUE_DECAY[ch])
        df[f"adstock_{ch}"] = adstocked.round(2)
        sales += TRUE_ROI[ch] * adstocked

    df["sales"] = (sales + rng.normal(0, 50, n_weeks)).clip(0).round(1)

    print(f"Generated {n_weeks} weeks of data")
    print(f"Avg weekly sales: ${df['sales'].mean():,.0f}K")
    return df


if __name__ == "__main__":
    df = generate_marketing_data()
    df.to_csv("data/marketing_data.csv", index=False)
    print(df[["week", "sales"] + [f"spend_{c}" for c in CHANNELS]].tail(8))