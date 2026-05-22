import numpy as np
import pandas as pd
import joblib
import warnings
import os
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from portfolio_optimizer import optimize, rolling_moments, ASSETS, ASSET_COLS

warnings.filterwarnings("ignore")
os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_YEARS  = 3      # re-fit HMM on rolling 3-year window
STEP_DAYS    = 63     # re-fit every quarter (63 trading days ≈ 3 months)
N_STATES     = 3
N_ITER       = 1000   # fewer iters per window — speed vs accuracy tradeoff
N_RESTARTS   = 10     # fewer restarts per window too
FEATURE_COLS = ["SPY_ret", "TLT_ret", "GLD_ret", "VIX_level", "yield_curve"]


# ── LABEL ALIGNMENT ───────────────────────────────────────────────────────────
def align_labels(new_model, new_scaler, prev_label_map):
    """
    Each time we re-fit the HMM on a new window, state integers (0,1,2)
    get re-assigned arbitrarily. We need to match new states to
    Bull/Bear/Crisis consistently using the same composite score logic.
    """
    means_scaled = new_model.means_
    means_raw    = new_scaler.inverse_transform(means_scaled)

    spy_idx = FEATURE_COLS.index("SPY_ret")
    vix_idx = FEATURE_COLS.index("VIX_level")
    yc_idx  = FEATURE_COLS.index("yield_curve")

    composite = (means_scaled[:, spy_idx]
                 - 0.3 * means_scaled[:, vix_idx]
                 + 0.1 * means_scaled[:, yc_idx])

    crisis_state = int(np.argmin(composite))
    bull_state   = int(np.argmax(composite))
    bear_state   = [s for s in range(N_STATES)
                    if s not in {crisis_state, bull_state}][0]

    return {bull_state: "Bull", bear_state: "Bear", crisis_state: "Crisis"}


# ── FIT ONE HMM WINDOW ────────────────────────────────────────────────────────
def fit_window(X_window):
    """Fits HMM on a single training window. Returns best model + scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_window)

    best_model, best_score = None, -np.inf

    for i in range(N_RESTARTS):
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="full",
            n_iter=N_ITER,
            tol=1e-5,
            random_state=42 + i,
            params="stmc",
            init_params="stmc",
        )
        try:
            model.fit(X_scaled)
            score = model.score(X_scaled)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    return best_model, scaler


# ── MAIN WALK-FORWARD LOOP ────────────────────────────────────────────────────
def run_walk_forward(feature_path="data/feature_matrix.csv"):
    """
    True walk-forward validation:

    ├─── TRAIN (3 years) ───┤─ TEST (1 quarter) ─┤
                             ├─── TRAIN (3 years) ───┤─ TEST ─┤
                                                      ├─── TRAIN ───┤─ TEST ─┤

    At each step:
      1. Fit HMM only on the past TRAIN_YEARS of data
      2. Decode regimes for the next STEP_DAYS (test window) — no future data
      3. Run portfolio optimizer on test window using only pre-test returns
      4. Record out-of-sample returns
    """
    feat_df = pd.read_csv(feature_path, index_col="Date", parse_dates=True)
    feat_df = feat_df[FEATURE_COLS + [c for c in feat_df.columns
                                      if c not in FEATURE_COLS]].copy()

    returns_df = feat_df[["SPY_ret", "TLT_ret", "GLD_ret"]].copy()
    returns_df.columns = ASSETS

    features   = feat_df[FEATURE_COLS].values
    dates      = feat_df.index

    train_days = TRAIN_YEARS * 252
    T          = len(feat_df)

    all_dates    = []
    all_returns  = []
    all_weights  = []
    all_regimes  = []

    prev_weights = None
    prev_map     = None
    window_count = 0

    # Start after enough data exists for a full training window
    t = train_days

    print(f"Walk-forward validation")
    print(f"  Training window : {TRAIN_YEARS} years ({train_days} days)")
    print(f"  Step size       : {STEP_DAYS} days (quarterly)")
    print(f"  Total periods   : ~{(T - train_days) // STEP_DAYS}\n")

    while t + STEP_DAYS <= T:

        # ── 1. Fit HMM on training window ────────────────────────────────────
        X_train    = features[t - train_days : t]
        model, scaler = fit_window(X_train)

        if model is None:
            t += STEP_DAYS
            continue

        label_map  = align_labels(model, scaler, prev_map)
        prev_map   = label_map

        # ── 2. Decode test window ─────────────────────────────────────────────
        t_end      = min(t + STEP_DAYS, T)
        X_test_raw = features[t : t_end]
        X_test_sc  = scaler.transform(X_test_raw)
        test_states = model.predict(X_test_sc)
        test_labels = [label_map[s] for s in test_states]
        test_dates  = dates[t : t_end]

        # ── 3. Optimize weights for each day in test window ───────────────────
        for i, (day_t, regime) in enumerate(zip(test_dates, test_labels)):
            abs_t = t + i

            mu, Sigma = rolling_moments(returns_df, abs_t)
            Sigma    += np.eye(len(ASSETS)) * 1e-8

            w = optimize(mu, Sigma, regime, prev_weights)
            if w is None:
                w = prev_weights if prev_weights is not None \
                    else np.ones(len(ASSETS)) / len(ASSETS)

            # Next-day return (out-of-sample)
            if abs_t + 1 < T:
                next_ret = returns_df.iloc[abs_t + 1].values
                port_ret = float(w @ next_ret)

                all_dates.append(day_t)
                all_returns.append(port_ret)
                all_weights.append(w.copy())
                all_regimes.append(regime)

            prev_weights = w.copy()

        window_count += 1
        if window_count % 5 == 0:
            oos_so_far = pd.Series(all_returns)
            sharpe = (oos_so_far.mean() /
                      oos_so_far.std() * np.sqrt(252)) if len(oos_so_far) > 5 else 0
            print(f"  Window {window_count:3d} | t={t:4d} | "
                  f"label_map={label_map} | "
                  f"OOS Sharpe so far: {sharpe:.2f}")

        t += STEP_DAYS

    # ── Compile results ───────────────────────────────────────────────────────
    wf_returns = pd.Series(all_returns, index=all_dates, name="wf_portfolio")
    wf_weights = pd.DataFrame(all_weights, index=all_dates, columns=ASSETS)
    wf_weights["regime"] = all_regimes

    wf_returns.to_csv("data/wf_returns.csv")
    wf_weights.to_csv("data/wf_weights.csv")

    print(f"\nWalk-forward complete: {len(wf_returns)} OOS trading days")
    return wf_returns, wf_weights


# ── BENCHMARKS ────────────────────────────────────────────────────────────────
def build_benchmarks(feature_path="data/feature_matrix.csv",
                     wf_index=None):
    """
    60/40: 60% SPY + 40% TLT, rebalanced monthly (static)
    Equal weight: 33/33/33 SPY+TLT+GLD
    Both trimmed to match the walk-forward OOS period.
    """
    feat  = pd.read_csv(feature_path, index_col="Date", parse_dates=True)
    ret   = feat[["SPY_ret", "TLT_ret", "GLD_ret"]].copy()
    ret.columns = ASSETS

    if wf_index is not None:
        ret = ret.loc[wf_index]

    bench_6040 = ret["SPY"] * 0.60 + ret["TLT"] * 0.40
    bench_ew   = ret.mean(axis=1)

    bench_6040.name = "60/40"
    bench_ew.name   = "Equal weight"

    return bench_6040, bench_ew


# ── STATS ─────────────────────────────────────────────────────────────────────
def compute_stats(returns_series, name=""):
    r   = returns_series.dropna()
    ann = 252

    total  = (1 + r).prod() - 1
    ann_r  = (1 + r).prod() ** (ann / len(r)) - 1
    vol    = r.std() * np.sqrt(ann)
    sharpe = r.mean() / r.std() * np.sqrt(ann)
    cum    = (1 + r).cumprod()
    dd     = (cum / cum.cummax() - 1).min()
    sortino_denom = r[r < 0].std() * np.sqrt(ann)
    sortino = ann_r / sortino_denom if sortino_denom > 0 else np.nan

    return {
        "Strategy":    name,
        "Total ret":   f"{total*100:.1f}%",
        "Ann. ret":    f"{ann_r*100:.1f}%",
        "Ann. vol":    f"{vol*100:.1f}%",
        "Sharpe":      f"{sharpe:.2f}",
        "Sortino":     f"{sortino:.2f}",
        "Max DD":      f"{dd*100:.1f}%",
        "N days":      len(r),
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    wf_returns, wf_weights = run_walk_forward()

    bench_6040, bench_ew = build_benchmarks(wf_index=wf_returns.index)

    print("\n" + "="*60)
    print("OUT-OF-SAMPLE PERFORMANCE COMPARISON")
    print("="*60)

    rows = [
        compute_stats(wf_returns,  "Regime-Shift (OOS)"),
        compute_stats(bench_6040,  "60/40 benchmark"),
        compute_stats(bench_ew,    "Equal weight"),
    ]
    result_df = pd.DataFrame(rows).set_index("Strategy")
    print(result_df.to_string())
    result_df.to_csv("data/oos_comparison.csv")

    print("\nPhase 4 complete — honest OOS numbers above.")
    print("Next: Phase 5 adds transaction friction costs.")