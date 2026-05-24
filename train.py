"""
Fit MMM and Markov attribution, then output channel ROI + budget recommendations.
Usage: python train.py
"""
import pathlib
import numpy as np
from data.synthetic_marketing import generate_marketing_data, CHANNELS
from models.mmm_regression import MarketingMixModel
from attribution.markov_attribution import markov_attribution


def simulate_customer_journeys(n: int = 5000, seed: int = 42):
    """Generate synthetic multi-touch customer journeys for Markov attribution."""
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(n):
        length = rng.integers(1, 5)
        path = list(rng.choice(CHANNELS, size=length, replace=True))
        paths.append(path)
    return paths


def main():
    pathlib.Path("results").mkdir(exist_ok=True)

    print("Generating marketing data (3 years weekly)...")
    df = generate_marketing_data(n_weeks=156)

    print("\nFitting Marketing Mix Model...")
    mmm = MarketingMixModel()
    mmm.fit(df)

    print("\nChannel ROI:")
    roi = mmm.channel_roi(df)
    print(roi.to_string(index=False, float_format="{:.2f}".format))
    roi.to_csv("results/channel_roi.csv", index=False)

    print("\nRunning Markov Chain Attribution...")
    journeys = simulate_customer_journeys(n=5000)
    attribution = markov_attribution(
        journeys, channels=CHANNELS, total_revenue=df["sales"].sum())
    print(attribution.to_string(index=False))
    attribution.to_csv("results/markov_attribution.csv", index=False)

    print("\n✓ Results saved to results/")


if __name__ == "__main__":
    main()