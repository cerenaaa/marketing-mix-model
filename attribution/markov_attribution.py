"""
Markov Chain multi-touch attribution.
Estimates each channel's removal effect on conversion probability.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations


def build_transition_matrix(paths: list[list[str]], channels: list[str]) -> pd.DataFrame:
    """Build channel → channel transition probability matrix from journey paths."""
    states = channels + ["(start)", "(conversion)", "(null)"]
    counts = defaultdict(lambda: defaultdict(int))

    for path in paths:
        prev = "(start)"
        for ch in path:
            counts[prev][ch] += 1
            prev = ch
        counts[prev]["(conversion)"] += 1

    # Add null paths (non-converting)
    for ch in channels:
        counts[ch]["(null)"] += max(1, counts[ch]["(conversion)"] // 3)

    df = pd.DataFrame(counts).T.fillna(0)
    df = df.div(df.sum(axis=1), axis=0).fillna(0)
    return df


def removal_effect(transition_matrix: pd.DataFrame, channel: str) -> float:
    """Compute conversion probability with channel removed."""
    mat = transition_matrix.copy()
    if channel in mat.index:
        mat.loc[channel] = 0
        if "(null)" in mat.columns:
            mat.loc[channel, "(null)"] = 1.0
    return _conversion_probability(mat)


def _conversion_probability(mat: pd.DataFrame, max_steps: int = 50) -> float:
    """Simulate conversion probability via matrix powers."""
    if "(start)" not in mat.index or "(conversion)" not in mat.columns:
        return 0.0
    state = mat.index.tolist()
    p = np.zeros(len(state))
    p[state.index("(start)")] = 1.0
    conv_idx = state.index("(conversion)") if "(conversion)" in state else -1
    if conv_idx < 0:
        return 0.0
    A = mat.values
    for _ in range(max_steps):
        p = p @ A
    return float(p[conv_idx])


def markov_attribution(
    paths: list[list[str]],
    channels: list[str],
    total_revenue: float = 1.0,
) -> pd.DataFrame:
    """
    Compute Markov Chain attribution for each channel.
    paths: list of ordered channel sequences per converting customer
    """
    mat = build_transition_matrix(paths, channels)
    baseline = _conversion_probability(mat)

    records = []
    for ch in channels:
        re = removal_effect(mat, ch)
        marginal = max(0, baseline - re)
        records.append({"channel": ch, "removal_effect": round(re, 4),
                        "marginal_contribution": round(marginal, 4)})

    df = pd.DataFrame(records)
    total_marginal = df["marginal_contribution"].sum()
    df["attribution_pct"] = (df["marginal_contribution"] / total_marginal * 100).round(1)
    df["attributed_revenue"] = (df["attribution_pct"] / 100 * total_revenue).round(0)
    return df.sort_values("attribution_pct", ascending=False)