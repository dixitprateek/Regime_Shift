import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

ASSETS   = ["SPY", "TLT", "GLD"]
RISK_FREE_ANNUAL = 0.025   # 2.5% annualised


# ── LOAD ALL DATA ─────────────────────────────────────────────────────────────
def load_all():
    wf_ret  = pd.read_csv("data/wf_returns.csv",
                          index_col=0, parse_dates=True).squeeze()
    wf_w    = pd.read_csv("data/wf_weights.csv",
                          index_col=0, parse_dates=True)
    feat    = pd.read_csv("data/feature_matrix.csv",
                          index_col=0, parse_dates=True)

    ret3    = feat[["SPY_ret","TLT_ret","GLD_ret"]].copy()
    ret3.columns = ASSETS

    # Trim all to OOS window
    idx       = wf_ret.index
    ret3      = ret3.reindex(idx)
    b6040     = ret3["SPY"]*0.60 + ret3["TLT"]*0.40
    b_ew      = ret3.mean(axis=1)
    spy_only  = ret3["SPY"]

    b6040.name    = "60/40"
    b_ew.name     = "Equal weight"
    spy_only.name = "SPY only"

    return wf_ret, wf_w, b6040, b_ew, spy_only


# ── METRICS ───────────────────────────────────────────────────────────────────
def full_metrics(r, name):
    r   = r.dropna()
    ann = 252
    rf  = RISK_FREE_ANNUAL / ann

    cum       = (1 + r).cumprod()
    total_ret = cum.iloc[-1] - 1
    ann_ret   = (1 + total_ret) ** (ann / len(r)) - 1
    vol       = r.std() * np.sqrt(ann)
    sharpe    = (r.mean() - rf) / r.std() * np.sqrt(ann)

    neg       = r[r < rf]
    sortino   = (ann_ret - RISK_FREE_ANNUAL) / (neg.std() * np.sqrt(ann)) \
                if len(neg) > 0 else np.nan

    roll_max  = cum.cummax()
    dd_series = cum / roll_max - 1
    max_dd    = dd_series.min()

    # Calmar ratio = ann_ret / abs(max_dd)
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    # Recovery: days from trough to new high
    in_dd     = dd_series < 0
    recovery_days = int(in_dd.sum())

    # Win rate
    win_rate  = (r > 0).mean()

    # VaR and CVaR (95%)
    var95     = np.percentile(r, 5)
    cvar95    = r[r <= var95].mean()

    # Best/worst month
    monthly   = r.resample("ME").apply(lambda x: (1+x).prod()-1)
    best_mo   = monthly.max()
    worst_mo  = monthly.min()

    return {
        "Strategy":        name,
        "Ann. Return":     f"{ann_ret*100:.2f}%",
        "Ann. Volatility": f"{vol*100:.2f}%",
        "Sharpe Ratio":    f"{sharpe:.3f}",
        "Sortino Ratio":   f"{sortino:.3f}",
        "Calmar Ratio":    f"{calmar:.3f}",
        "Max Drawdown":    f"{max_dd*100:.2f}%",
        "Days in Drawdown":f"{recovery_days}",
        "Win Rate":        f"{win_rate*100:.1f}%",
        "VaR 95%":         f"{var95*100:.2f}%",
        "CVaR 95%":        f"{cvar95*100:.2f}%",
        "Best Month":      f"{best_mo*100:.2f}%",
        "Worst Month":     f"{worst_mo*100:.2f}%",
        "Total Return":    f"{total_ret*100:.1f}%",
    }


# ── DRAWDOWN SERIES ───────────────────────────────────────────────────────────
def drawdown_series(r):
    cum = (1 + r).cumprod()
    return cum / cum.cummax() - 1


# ── ROLLING METRICS ───────────────────────────────────────────────────────────
def rolling_sharpe(r, window=252):
    rf = RISK_FREE_ANNUAL / 252
    roll_mean = r.rolling(window).mean() - rf
    roll_std  = r.rolling(window).std()
    return roll_mean / roll_std * np.sqrt(252)


def rolling_vol(r, window=63):
    return r.rolling(window).std() * np.sqrt(252)


# ── MONTHLY RETURN HEATMAP ───────────────────────────────────────────────────
def monthly_heatmap_data(r):
    monthly = r.resample("ME").apply(lambda x: (1+x).prod()-1) * 100
    df = monthly.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
    return pivot


# ── FULL TEAR SHEET PLOT ──────────────────────────────────────────────────────
def plot_tearsheet(wf_ret, wf_w, b6040, b_ew, spy):

    COLORS = {
        "Regime-Shift": "#5DCAA5",
        "60/40":        "#7F77DD",
        "Equal weight": "#EF9F27",
        "SPY only":     "#aaaaaa",
    }

    fig = plt.figure(figsize=(18, 22), facecolor="#0d1117")
    gs  = gridspec.GridSpec(
        5, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        height_ratios=[2.5, 2, 1.8, 2.2, 2.2]
    )

    def style(ax, title=""):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines[:].set_color("#222")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color("#aaaaaa")
        if title:
            ax.set_title(title, color="white", fontsize=9, pad=6)
        return ax

    pct_fmt  = FuncFormatter(lambda x, _: f"{x*100:.0f}%")
    pct2_fmt = FuncFormatter(lambda x, _: f"{x:.2f}")

    # ── Row 0: Cumulative return (full width) ─────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    style(ax0, "Cumulative Wealth (OOS Walk-Forward)")
    for r_s, name in [(wf_ret,"Regime-Shift"),(b6040,"60/40"),
                      (b_ew,"Equal weight"),(spy,"SPY only")]:
        cum = (1 + r_s).cumprod()
        ax0.plot(cum.index, cum.values,
                 color=COLORS[name], lw=1.5 if name=="Regime-Shift" else 1.0,
                 label=name,
                 ls="-" if name=="Regime-Shift" else "--")
    ax0.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9,
               loc="upper left")
    ax0.yaxis.set_major_formatter(pct_fmt)
    ax0.set_ylabel("Return", color="#aaaaaa", fontsize=9)

    # ── Row 1 col 0: Drawdown ─────────────────────────────────────────────────
    ax_dd = fig.add_subplot(gs[1, :])
    style(ax_dd, "Drawdown")
    for r_s, name in [(wf_ret,"Regime-Shift"),(b6040,"60/40"),(spy,"SPY only")]:
        dd = drawdown_series(r_s)
        ax_dd.fill_between(dd.index, dd.values, 0,
                           alpha=0.35, color=COLORS[name], label=name)
        ax_dd.plot(dd.index, dd.values, color=COLORS[name], lw=0.7)
    ax_dd.yaxis.set_major_formatter(pct_fmt)
    ax_dd.set_ylabel("Drawdown", color="#aaaaaa", fontsize=9)
    ax_dd.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8,
                 loc="lower left")

    # ── Row 2: Rolling 1Y Sharpe / Rolling Vol / Regime weights ──────────────
    ax_rs  = fig.add_subplot(gs[2, 0])
    ax_rv  = fig.add_subplot(gs[2, 1])
    ax_rw  = fig.add_subplot(gs[2, 2])

    style(ax_rs, "Rolling 1Y Sharpe")
    for r_s, name in [(wf_ret,"Regime-Shift"),(b6040,"60/40")]:
        rs = rolling_sharpe(r_s)
        ax_rs.plot(rs.index, rs.values, color=COLORS[name], lw=1, label=name)
    ax_rs.axhline(0, color="#555", lw=0.7, ls="--")
    ax_rs.axhline(1, color="#5DCAA5", lw=0.5, ls=":", alpha=0.5)
    ax_rs.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7)
    ax_rs.set_ylabel("Sharpe", color="#aaaaaa", fontsize=8)

    style(ax_rv, "Rolling 63d Volatility (Ann.)")
    for r_s, name in [(wf_ret,"Regime-Shift"),(b6040,"60/40"),(spy,"SPY only")]:
        rv = rolling_vol(r_s)
        ax_rv.plot(rv.index, rv.values, color=COLORS[name], lw=1, label=name)
    ax_rv.yaxis.set_major_formatter(pct_fmt)
    ax_rv.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7)
    ax_rv.set_ylabel("Volatility", color="#aaaaaa", fontsize=8)

    style(ax_rw, "Portfolio Weights Over Time")
    w = wf_w[ASSETS].reindex(wf_ret.index).ffill()
    wcol = ["#5DCAA5","#7F77DD","#EF9F27"]
    bot  = np.zeros(len(w))
    for col, c in zip(ASSETS, wcol):
        ax_rw.bar(w.index, w[col].values, bottom=bot,
                  color=c, alpha=0.85, width=1, label=col)
        bot += w[col].values
    ax_rw.legend(facecolor="#1a1a2e", labelcolor="white",
                 fontsize=7, ncol=3, loc="upper left")
    ax_rw.set_ylim(0, 1)

    # ── Row 3: Monthly return heatmap (Regime-Shift) ──────────────────────────
    ax_hm = fig.add_subplot(gs[3, :])
    style(ax_hm, "Regime-Shift Monthly Returns (%)")
    pivot = monthly_heatmap_data(wf_ret)
    im    = ax_hm.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                         vmin=-8, vmax=8)
    ax_hm.set_xticks(range(12))
    ax_hm.set_xticklabels(pivot.columns, color="#aaaaaa", fontsize=8)
    ax_hm.set_yticks(range(len(pivot.index)))
    ax_hm.set_yticklabels(pivot.index, color="#aaaaaa", fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax_hm.text(j, i, f"{val:.1f}", ha="center", va="center",
                           fontsize=6.5,
                           color="black" if abs(val) < 5 else "white")
    fig.colorbar(im, ax=ax_hm, fraction=0.015, pad=0.01).ax.tick_params(
        colors="#aaaaaa", labelsize=7)

    # ── Row 4: Return distribution + Regime distribution ─────────────────────
    ax_dist = fig.add_subplot(gs[4, 0:2])
    ax_reg  = fig.add_subplot(gs[4, 2])

    style(ax_dist, "Daily Return Distribution")
    for r_s, name in [(wf_ret,"Regime-Shift"),(b6040,"60/40"),(spy,"SPY only")]:
        ax_dist.hist(r_s.dropna()*100, bins=100, alpha=0.4,
                     color=COLORS[name], label=name, density=True)
    ax_dist.axvline(0, color="white", lw=0.7, ls="--")
    ax_dist.set_xlabel("Daily return (%)", color="#aaaaaa", fontsize=8)
    ax_dist.set_ylabel("Density", color="#aaaaaa", fontsize=8)
    ax_dist.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    style(ax_reg, "Regime Distribution (OOS)")
    reg_counts = wf_w["regime"].reindex(wf_ret.index).ffill().value_counts()
    reg_colors = {"Bull":"#5DCAA5","Bear":"#EF9F27","Crisis":"#e74c3c"}
    bars = ax_reg.bar(reg_counts.index,
                      reg_counts.values / reg_counts.sum() * 100,
                      color=[reg_colors.get(r,"gray") for r in reg_counts.index],
                      alpha=0.8, edgecolor="#222")
    for bar, val in zip(bars, reg_counts.values / reg_counts.sum() * 100):
        ax_reg.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", color="white", fontsize=9)
    ax_reg.set_ylabel("% of OOS days", color="#aaaaaa", fontsize=8)
    ax_reg.set_ylim(0, 60)

    # ── Main title ────────────────────────────────────────────────────────────
    fig.text(0.5, 0.985,
             "REGIME-SHIFT  ·  Macro-Aware Tactical Allocation  ·  Performance Tear Sheet",
             ha="center", va="top", color="white", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.975,
             "Out-of-sample walk-forward validation  ·  2008–2024  ·  FEC IIT Guwahati",
             ha="center", va="top", color="#aaaaaa", fontsize=9)

    path = "plots/tearsheet.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Tear sheet saved → {path}")


# ── PRINT STATS TABLE ─────────────────────────────────────────────────────────
def print_stats_table(wf_ret, b6040, b_ew, spy):
    rows = [
        full_metrics(wf_ret, "Regime-Shift (OOS)"),
        full_metrics(b6040,  "60/40 benchmark"),
        full_metrics(b_ew,   "Equal weight"),
        full_metrics(spy,    "SPY only"),
    ]
    df = pd.DataFrame(rows).set_index("Strategy").T
    print("\n" + "="*70)
    print("FULL PERFORMANCE TEAR SHEET")
    print("="*70)
    print(df.to_string())
    df.to_csv("data/tearsheet_stats.csv")
    print("\nSaved → data/tearsheet_stats.csv")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    wf_ret, wf_w, b6040, b_ew, spy = load_all()

    print_stats_table(wf_ret, b6040, b_ew, spy)
    plot_tearsheet(wf_ret, wf_w, b6040, b_ew, spy)

    print("\n" + "="*70)
    print("PROJECT COMPLETE — Regime-Shift tear sheet generated.")
    print("Deliverables:")
    print("  data/tearsheet_stats.csv   — full metrics table")
    print("  plots/tearsheet.png        — visual tear sheet")
    print("  plots/regimes.png          — HMM regime overlay")
    print("  plots/friction_analysis.png— transaction cost breakdown")
    print("  models/hmm_model.pkl       — trained HMM")
    print("  models/scaler.pkl          — feature scaler")
    print("="*70)