import numpy as np
import pandas as pd
import cvxpy as cp
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("data", exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
ASSETS       = ["SPY", "TLT", "GLD"]   # must match feature_matrix columns
ASSET_COLS = ["SPY_ret", "TLT_ret", "GLD_ret"]  # actual column names in CSV

LOOKBACK     = 252                      # 1 year of daily returns for covariance
RISK_FREE    = 0.0001                   # daily rf rate (~2.5% annual / 252)
TURNOVER_BPS = 7.5                      # transaction cost per unit of turnover

# Per-regime weight bounds  [min, max] for each asset
# Bull:   equity-heavy, bonds light, gold minor
# Bear:   reduce equity, boost bonds
# Crisis: minimal equity, max bonds + gold (safe havens)
BOUNDS = {
    "Bull":   {"SPY": (0.30, 0.70), "TLT": (0.10, 0.40), "GLD": (0.05, 0.25)},
    "Bear":   {"SPY": (0.15, 0.50), "TLT": (0.20, 0.50), "GLD": (0.10, 0.30)},
    "Crisis": {"SPY": (0.00, 0.25), "TLT": (0.30, 0.60), "GLD": (0.15, 0.40)},
}


# ── CORE OPTIMIZER ────────────────────────────────────────────────────────────
def optimize(mu, Sigma, regime, prev_weights=None):
    """
    Solves a convex portfolio optimization problem using CVXPY.

    mu     : (n,) expected returns vector
    Sigma  : (n, n) covariance matrix
    regime : "Bull" | "Bear" | "Crisis"
    prev_weights: (n,) weights from last period (for turnover penalty)

    Returns optimal weight vector (n,)
    """
    n = len(ASSETS)
    w = cp.Variable(n)

    bounds = BOUNDS[regime]
    lo = np.array([bounds[a][0] for a in ASSETS])
    hi = np.array([bounds[a][1] for a in ASSETS])

    # Portfolio variance (quadratic — this is why we need CVXPY)
    port_var = cp.quad_form(w, Sigma)

    # Turnover cost: penalize large weight changes from last period
    if prev_weights is not None:
        turnover_cost = TURNOVER_BPS * 1e-4 * cp.norm1(w - prev_weights)
    else:
        turnover_cost = 0

    if regime in ("Bull", "Bear"):
        # ── Maximize Sharpe ratio (as a convex proxy) ────────────────────────
        # True Sharpe maximization is non-convex.
        # Standard trick: fix denominator = 1, optimize numerator/denominator.
        # We use the Markowitz mean-variance formulation instead:
        # maximize  mu^T w - (lambda/2) * w^T Sigma w
        # where lambda controls risk aversion.
        # For Bull: low lambda (less risk averse)
        # For Bear: higher lambda (more risk averse)
        lam = 1.0 if regime == "Bull" else 3.0

        objective = cp.Maximize(
            mu @ w - (lam / 2) * port_var - turnover_cost
        )

    else:  # Crisis
        # ── Minimize portfolio variance ──────────────────────────────────────
        # Pure capital preservation — don't care about returns, just survive.
        objective = cp.Minimize(port_var + turnover_cost)

    constraints = [
        cp.sum(w) == 1.0,       # fully invested (no cash)
        w >= lo,                 # lower bounds per regime
        w <= hi,                 # upper bounds per regime
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, warm_start=True)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        # Fallback: equal weight within bounds
        print(f"  WARNING: solver status={prob.status} — using equal weight")
        return lo + (hi - lo) / 2 / np.sum(lo + (hi - lo) / 2)

    return w.value


# ── ROLLING ESTIMATORS ────────────────────────────────────────────────────────
def rolling_moments(returns_df, t, lookback=LOOKBACK):
    """
    Estimate mu and Sigma using only data up to and including day t.
    This is the key: we NEVER look forward in time.
    """
    window = returns_df.iloc[max(0, t - lookback):t]

    mu    = window.mean().values * 252        # annualise
    Sigma = window.cov().values  * 252        # annualise

    # Ledoit-Wolf shrinkage: stabilises covariance estimation
    # when lookback < n_assets^2  (not an issue here, but good practice)
    Sigma = _ledoit_wolf_shrink(Sigma, len(window))

    return mu, Sigma


def _ledoit_wolf_shrink(S, T, alpha=0.1):
    """
    Simple constant-alpha shrinkage toward the diagonal.
    Reduces estimation error in covariance matrices from finite samples.
    alpha=0: no shrinkage (use raw sample cov)
    alpha=1: use only diagonal (variances, no correlations)
    """
    n   = S.shape[0]
    mu  = np.trace(S) / n
    F   = mu * np.eye(n)           # shrinkage target: scaled identity
    return (1 - alpha) * S + alpha * F


# ── BACKTEST LOOP ─────────────────────────────────────────────────────────────
def run_backtest(feature_path="data/feature_matrix.csv",
                 regime_path="data/regimes.csv"):
    """
    Walks forward day by day.
    On each day:
      1. Look up today's regime from the HMM
      2. Estimate mu and Sigma from PAST data only
      3. Solve the optimizer for today's regime
      4. Record tomorrow's portfolio return using today's weights
    """
    feat_df   = pd.read_csv(feature_path, index_col="Date", parse_dates=True)
    regime_df = pd.read_csv(regime_path,  index_col="Date", parse_dates=True)

    # Align indices
    common    = feat_df.index.intersection(regime_df.index)
    returns = feat_df.loc[common, ASSET_COLS].copy()
    returns.columns = ASSETS          # rename to clean labels for the rest of the code
    regimes   = regime_df.loc[common, "label"]

    n         = len(ASSETS)
    T         = len(returns)

    weights_history = []
    regime_history  = []
    dates           = []
    prev_weights    = None

    print(f"Running backtest: {T} days, {n} assets")
    print(f"Lookback window : {LOOKBACK} days")
    print(f"Warmup period   : first {LOOKBACK} days skipped\n")

    for t in range(LOOKBACK, T - 1):
        regime = regimes.iloc[t]

        mu, Sigma = rolling_moments(returns, t)

        # Handle near-singular covariance
        Sigma += np.eye(n) * 1e-8

        w = optimize(mu, Sigma, regime, prev_weights)

        if w is None:
            w = prev_weights if prev_weights is not None else np.ones(n) / n

        weights_history.append(w.copy())
        regime_history.append(regime)
        dates.append(returns.index[t])
        prev_weights = w.copy()

        if t % 500 == 0:
            print(f"  t={t:4d}  regime={regime:6s}  "
                  f"w={np.round(w, 2)}")

    weights_df = pd.DataFrame(
        weights_history,
        index=dates,
        columns=ASSETS,
    )
    weights_df["regime"] = regime_history

    # ── Compute portfolio returns ─────────────────────────────────────────────
    # weights_df[t] are decided at end of day t, applied to day t+1 returns
    fwd_returns = returns.shift(-1).loc[dates]

    port_returns = (weights_df[ASSETS].values * fwd_returns.values).sum(axis=1)
    port_returns = pd.Series(port_returns, index=dates, name="portfolio")

    # Save
    weights_df.to_csv("data/weights.csv")
    port_returns.to_csv("data/portfolio_returns.csv")
    print(f"\nSaved weights        → data/weights.csv")
    print(f"Saved port returns   → data/portfolio_returns.csv")

    return weights_df, port_returns


# ── QUICK STATS PREVIEW ───────────────────────────────────────────────────────
def preview_stats(port_returns):
    r   = port_returns.dropna()
    ann = 252

    total_ret  = (1 + r).prod() - 1
    ann_ret    = (1 + r).prod() ** (ann / len(r)) - 1
    ann_vol    = r.std() * np.sqrt(ann)
    sharpe     = (r.mean() - RISK_FREE) / r.std() * np.sqrt(ann)
    max_dd     = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()

    print("\n── Portfolio preview (full in-sample) ──")
    print(f"  Total return  : {total_ret*100:.1f}%")
    print(f"  Ann. return   : {ann_ret*100:.1f}%")
    print(f"  Ann. vol      : {ann_vol*100:.1f}%")
    print(f"  Sharpe ratio  : {sharpe:.2f}")
    print(f"  Max drawdown  : {max_dd*100:.1f}%")
    print("\n  (Full tear sheet comparison vs 60/40 comes in Phase 6)")


if __name__ == "__main__":
    weights_df, port_returns = run_backtest()
    preview_stats(port_returns)
    print("\nPhase 3 complete. Weights and returns saved.")