import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

TURNOVER_BPS = 7.5    # matches what the optimizer used
ASSETS       = ["SPY", "TLT", "GLD"]

def load_data():
    # index_col=0 picks the first column regardless of its name
    weights  = pd.read_csv("data/wf_weights.csv",
                           index_col=0, parse_dates=True)
    returns  = pd.read_csv("data/wf_returns.csv",
                           index_col=0, parse_dates=True).squeeze()
    feat     = pd.read_csv("data/feature_matrix.csv",
                           index_col=0, parse_dates=True)

    feat_ret = feat[["SPY_ret", "TLT_ret", "GLD_ret"]].copy()
    feat_ret.columns = ASSETS

    # Rename weights index to Date for consistency
    weights.index.name  = "Date"
    returns.index.name  = "Date"

    return weights, returns, feat_ret


def compute_turnover(weights_df):
    """
    Daily turnover = sum of absolute weight changes between rebalances.
    Annualised turnover = mean daily turnover × 252.
    """
    w = weights_df[ASSETS]
    daily_turnover = w.diff().abs().sum(axis=1)
    daily_turnover.iloc[0] = w.iloc[0].abs().sum()   # day-0: full investment
    return daily_turnover


def compute_friction_cost(daily_turnover, bps=TURNOVER_BPS):
    """
    Cost per day = turnover × bps × 0.0001
    Returns a daily cost series.
    """
    return daily_turnover * bps * 1e-4


def build_gross_returns(net_returns, daily_cost):
    """
    Gross return (before friction) = net return + cost
    Lets us see exactly how much friction is eating.
    """
    aligned_cost = daily_cost.reindex(net_returns.index).fillna(0)
    gross = net_returns + aligned_cost
    gross.name = "gross_portfolio"
    return gross


def regime_turnover_breakdown(weights_df, daily_turnover):
    """
    Shows which regime drives the most rebalancing activity.
    """
    df = pd.DataFrame({
        "turnover": daily_turnover,
        "regime":   weights_df["regime"]
    }).dropna()

    summary = df.groupby("regime")["turnover"].agg(
        mean_daily="mean",
        total="sum",
        count="count"
    )
    summary["ann_turnover"]    = summary["mean_daily"] * 252
    summary["total_cost_bps"]  = summary["total"] * TURNOVER_BPS
    return summary


def rebalance_frequency(weights_df):
    """
    Counts how often the regime actually changed (triggering a rebalance).
    Days with no regime change = near-zero turnover.
    Days with regime change    = full portfolio shift.
    """
    regimes   = weights_df["regime"]
    n_changes = (regimes != regimes.shift()).sum()
    n_days    = len(regimes)
    return n_changes, n_days


def performance_with_without_friction(net_returns, gross_returns):
    """
    Side-by-side stats: gross (pre-cost) vs net (post-cost).
    Shows the exact drag from transaction costs.
    """
    def stats(r, name):
        ann = 252
        ann_r  = (1 + r).prod() ** (ann / len(r)) - 1
        vol    = r.std() * np.sqrt(ann)
        sharpe = r.mean() / r.std() * np.sqrt(ann)
        cum    = (1 + r).cumprod()
        dd     = (cum / cum.cummax() - 1).min()
        return {"Strategy": name,
                "Ann. ret":  f"{ann_r*100:.2f}%",
                "Sharpe":    f"{sharpe:.3f}",
                "Max DD":    f"{dd*100:.2f}%",
                "Total ret": f"{((1+r).prod()-1)*100:.1f}%"}

    return pd.DataFrame([
        stats(gross_returns, "Gross (pre-friction)"),
        stats(net_returns,   "Net  (post-friction)"),
    ]).set_index("Strategy")


def plot_friction(net_returns, gross_returns, daily_cost, weights_df,
                  save_path="plots/friction_analysis.png"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1]})
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        for lbl in ax.get_yticklabels():
            lbl.set_color("white")

    # Panel 1: gross vs net cumulative return
    ax1 = axes[0]
    gross_cum = (1 + gross_returns).cumprod()
    net_cum   = (1 + net_returns).cumprod()

    ax1.plot(gross_cum.index, gross_cum.values,
             color="#5DCAA5", lw=1.5, label="Gross (pre-friction)", alpha=0.9)
    ax1.plot(net_cum.index, net_cum.values,
             color="#ffffff", lw=1.2, label="Net (post-friction)", ls="--")
    ax1.fill_between(gross_cum.index,
                     gross_cum.values, net_cum.values,
                     alpha=0.2, color="#e74c3c", label="Friction drag")
    ax1.set_ylabel("Cumulative return", color="white")
    ax1.set_title("Transaction friction analysis",
                  color="white", fontsize=13, pad=10)
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    # Panel 2: rolling 63-day annualised turnover
    ax2 = axes[1]
    roll_turnover = daily_cost.reindex(net_returns.index).fillna(0)
    roll_ann = roll_turnover.rolling(63).sum() * (252 / 63) * 100  # % per year
    ax2.bar(roll_ann.index, roll_ann.values,
            color="#e67e22", alpha=0.6, width=1)
    ax2.set_ylabel("Ann. cost (% p.a.)", color="white")
    ax2.axhline(roll_ann.mean(), color="white", lw=0.8,
                ls="--", alpha=0.5, label=f"Mean: {roll_ann.mean():.2f}%")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    # Panel 3: weight allocation over time
    ax3 = axes[2]
    w = weights_df[ASSETS].reindex(net_returns.index).ffill()

    colors3 = ["#5DCAA5", "#7F77DD", "#EF9F27"]
    bottom  = np.zeros(len(w))
    for col, c in zip(ASSETS, colors3):
        ax3.bar(w.index, w[col].values, bottom=bottom,
                color=c, alpha=0.8, width=1, label=col)
        bottom += w[col].values
    ax3.set_ylabel("Weights", color="white")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white",
               fontsize=8, ncol=3, loc="upper left")

    plt.tight_layout(h_pad=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved → {save_path}")


if __name__ == "__main__":
    weights, net_returns, feat_ret = load_data()

    daily_turnover = compute_turnover(weights)
    daily_cost     = compute_friction_cost(daily_turnover)
    gross_returns  = build_gross_returns(net_returns, daily_cost)

    # ── Report ────────────────────────────────────────────────────────────────
    n_changes, n_days = rebalance_frequency(weights)

    print("=" * 55)
    print("TRANSACTION FRICTION REPORT")
    print("=" * 55)

    print(f"\nPortfolio activity ({n_days} OOS trading days):")
    print(f"  Regime changes (rebalance triggers) : {n_changes}")
    print(f"  Rebalance frequency                 : "
          f"every ~{n_days//n_changes:.0f} days")
    print(f"  Cost per unit turnover              : {TURNOVER_BPS} bps")

    total_cost_pct = daily_cost.sum() * 100
    ann_cost_pct   = daily_cost.mean() * 252 * 100
    print(f"\nCost summary:")
    print(f"  Total friction cost (OOS period)    : {total_cost_pct:.2f}%")
    print(f"  Annualised friction drag            : {ann_cost_pct:.4f}% p.a.")
    print(f"  Mean daily turnover                 : "
          f"{daily_turnover.mean()*100:.3f}%")

    print("\nPer-regime turnover breakdown:")
    breakdown = regime_turnover_breakdown(weights, daily_turnover)
    print(breakdown.round(4).to_string())

    print("\nGross vs net performance:")
    comparison = performance_with_without_friction(net_returns, gross_returns)
    print(comparison.to_string())

    comparison.to_csv("data/friction_comparison.csv")

    plot_friction(net_returns, gross_returns, daily_cost, weights)

    print("\nPhase 5 complete. Ready for Phase 6 (full tear sheet).")