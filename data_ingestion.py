import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred

# ── CONFIG ──────────────────────────────────────────────────────────────────
FRED_API_KEY = "ccb9a3ce2ae9f7055288db1764a5f67b"
START_DATE   = "2005-01-01"
END_DATE     = "2024-12-31"

ASSETS = ["SPY", "TLT", "GLD", "AGG"]

FRED_SERIES = {
    "CPIAUCSL":      "CPI (inflation)",
    "GS10":          "10Y Treasury yield",
    "GS2":           "2Y Treasury yield",
    "BAMLH0A0HYM2":  "HY credit spread",
    "UNRATE":        "Unemployment rate",
}

# ── STEP 1: Download asset prices ────────────────────────────────────────────
def fetch_asset_data(tickers, start, end):
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance 1.x returns MultiIndex columns: (Field, Ticker)
    # Handle both old and new API gracefully
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]           # shape: (days, n_tickers)
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    print(f"  Downloaded prices: {prices.shape[0]} rows, "
          f"{prices.shape[1]} tickers")
    print(f"  Price date range: {prices.index[0].date()} "
          f"to {prices.index[-1].date()}")

    # Log returns
    log_returns = np.log(prices / prices.shift(1))
    return log_returns


# ── STEP 2: Download macro data from FRED ────────────────────────────────────
def fetch_macro_data(series_dict, start, end, api_key):
    fred = Fred(api_key=api_key)
    frames = {}

    for series_id, label in series_dict.items():
        try:
            s = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            frames[series_id] = s
            print(f"  {label}: {len(s)} obs  "
                  f"({s.index[0].date()} → {s.index[-1].date()})")
        except Exception as e:
            print(f"  WARNING: Could not fetch {series_id} — {e}")

    macro_df = pd.DataFrame(frames)

    # Reindex to all business days and forward-fill
    bday_index = pd.date_range(start=start, end=end, freq="B")
    macro_df = macro_df.reindex(bday_index).ffill()

    return macro_df


# ── STEP 3: Fetch raw VIX level ───────────────────────────────────────────────
def fetch_vix(start, end):
    raw = yf.download("^VIX", start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        vix = raw["Close"].squeeze()
    else:
        vix = raw["Close"]
    vix.name = "VIX_level"
    print(f"  VIX: {len(vix)} obs  "
          f"({vix.index[0].date()} → {vix.index[-1].date()})")
    return vix


# ── STEP 4: Build feature matrix ─────────────────────────────────────────────
def build_feature_matrix():
    print("\n[1] Downloading asset prices...")
    asset_rets = fetch_asset_data(ASSETS, START_DATE, END_DATE)

    print("\n[2] Downloading macro data from FRED...")
    macro_df = fetch_macro_data(FRED_SERIES, START_DATE, END_DATE, FRED_API_KEY)

    print("\n[3] Downloading VIX...")
    vix = fetch_vix(START_DATE, END_DATE)

    print("\n[4] Building feature matrix...")
    features = pd.DataFrame(index=asset_rets.index)

    # Core returns
    for col in ["SPY", "TLT", "GLD"]:
        if col in asset_rets.columns:
            features[f"{col}_ret"] = asset_rets[col]

    # VIX level — align to asset index
    features["VIX_level"] = vix.reindex(features.index).ffill()

    # Yield curve slope (10Y − 2Y)
    if "GS10" in macro_df.columns and "GS2" in macro_df.columns:
        slope = macro_df["GS10"] - macro_df["GS2"]
        features["yield_curve"] = slope.reindex(features.index).ffill()

    # HY credit spread — daily series, may have shorter history
    if "BAMLH0A0HYM2" in macro_df.columns:
        features["credit_spread"] = (
            macro_df["BAMLH0A0HYM2"].reindex(features.index).ffill()
        )

    # ── KEY FIX: only require CORE columns to be non-NaN ────────────────────
    # Don't drop rows just because a macro series starts late.
    # The HMM can handle a feature being NaN-filled from its first available date.
    core_cols = ["SPY_ret", "TLT_ret", "GLD_ret", "VIX_level"]
    features = features.dropna(subset=core_cols)   # <-- was: dropna()

    # For macro columns: backfill from their earliest available value
    # (only affects the very first few rows before macro data begins)
    features = features.bfill()

    print(f"\n  Feature matrix shape : {features.shape}")
    print(f"  Date range           : {features.index[0].date()} "
          f"→ {features.index[-1].date()}")
    print(f"  Columns              : {list(features.columns)}")
    print(f"\n  NaN counts per column:")
    print(features.isna().sum())

    import os
    os.makedirs("data", exist_ok=True)
    features.to_csv("data/feature_matrix.csv")
    print("\n  Saved → data/feature_matrix.csv")

    return features


if __name__ == "__main__":
    df = build_feature_matrix()
    print("\nSample (first 3 rows):")
    print(df.head(3).to_string())
    print("\nSample (last 3 rows):")
    print(df.tail(3).to_string())