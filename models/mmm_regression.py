"""
Ridge regression MMM with adstock transforms and revenue decomposition.
Includes contribution analysis and ROI estimation per channel.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings

from models.adstock import apply_adstock


CHANNELS = ["tv", "digital", "ooh", "instore_promo"]


class MarketingMixModel:
    """
    Ridge MMM: Sales = Base + Σ β_c · Adstock(Spend_c) + ε
    Reports contribution of each channel to total revenue.
    """

    def __init__(self, alphas: tuple = (0.01, 0.1, 1.0, 10.0, 100.0)):
        self.alphas = alphas
        self.model = None
        self.scaler = StandardScaler()
        self.channel_coefs: dict[str, float] = {}
        self.feature_names: list[str] = []

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = {}
        # Trend and seasonality
        n = len(df)
        features["trend"] = np.arange(n) / n
        features["sin52"] = np.sin(2 * np.pi * np.arange(n) / 52)
        features["cos52"] = np.cos(2 * np.pi * np.arange(n) / 52)

        # Adstocked channel spend
        for ch in CHANNELS:
            if f"spend_{ch}" in df.columns:
                adstocked = apply_adstock(df[f"spend_{ch}"].values)
                features[f"adstock_{ch}"] = adstocked
            elif f"adstock_{ch}" in df.columns:
                features[f"adstock_{ch}"] = df[f"adstock_{ch}"].values

        return pd.DataFrame(features, index=df.index)

    def fit(self, df: pd.DataFrame, target: str = "sales") -> "MarketingMixModel":
        X = self._build_features(df)
        y = df[target].values
        self.feature_names = list(X.columns)

        X_scaled = self.scaler.fit_transform(X)
        self.model = RidgeCV(alphas=self.alphas, cv=5)
        self.model.fit(X_scaled, y)

        print(f"MMM R²: {r2_score(y, self.model.predict(X_scaled)):.4f}")
        print(f"Best alpha: {self.model.alpha_:.4f}")

        # Map coefficients back to original scale
        coefs = self.model.coef_ / self.scaler.scale_
        self.channel_coefs = {
            name: round(coef, 4)
            for name, coef in zip(self.feature_names, coefs)
            if "adstock_" in name
        }
        return self

    def decompose(self, df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
        """Revenue decomposition: base + channel contributions."""
        X = self._build_features(df)
        X_scaled = self.scaler.transform(X)

        coefs = self.model.coef_
        intercept = self.model.intercept_

        decomp = pd.DataFrame(index=df.index)
        decomp["base"] = intercept
        for i, name in enumerate(self.feature_names):
            decomp[name] = X_scaled[:, i] * coefs[i]

        # Channel-level summary
        channel_cols = [c for c in decomp.columns if "adstock_" in c]
        summary = {
            "base": decomp["base"].sum(),
            **{col.replace("adstock_", ""): decomp[col].sum() for col in channel_cols}
        }
        total = sum(summary.values())
        contribution_pct = {k: round(v / total * 100, 1) for k, v in summary.items()}

        print("\nRevenue Decomposition:")
        for k, v in contribution_pct.items():
            print(f"  {k:20s}: {v:5.1f}%")

        return decomp

    def channel_roi(self, df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
        """Estimated ROI per channel: revenue contribution / spend."""
        decomp = self.decompose(df, target)
        rows = []
        for ch in CHANNELS:
            col = f"adstock_{ch}"
            spend_col = f"spend_{ch}"
            if col in decomp.columns and spend_col in df.columns:
                contribution = decomp[col].sum()
                total_spend = df[spend_col].sum()
                roi = contribution / total_spend if total_spend > 0 else 0
                rows.append({"channel": ch, "total_spend": round(total_spend, 0),
                             "revenue_contribution": round(contribution, 0),
                             "roi": round(roi, 2)})
        return pd.DataFrame(rows).sort_values("roi", ascending=False)