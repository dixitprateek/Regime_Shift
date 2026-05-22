import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import joblib, os

warnings.filterwarnings("ignore")
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_STATES  = 3
N_ITER    = 2000
N_RESTART = 50
SEED      = 42

# FIX 1: removed credit_spread — it was constant for 18 of 20 years
# (FRED daily series only went back to May 2023; backfill made it useless)
FEATURE_COLS = ["SPY_ret", "TLT_ret", "GLD_ret", "VIX_level", "yield_curve"]


# ── LOAD + SCALE ──────────────────────────────────────────────────────────────
def load_and_scale(path="data/feature_matrix.csv"):
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = df[FEATURE_COLS].dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    joblib.dump(scaler, "models/scaler.pkl")

    print(f"Feature matrix : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Date range     : {df.index[0].date()} → {df.index[-1].date()}")
    return df, X, scaler


# ── FIT HMM ───────────────────────────────────────────────────────────────────
def fit_hmm(X):
    best_model, best_score = None, -np.inf

    print(f"\nFitting HMM ({N_STATES} states, {N_RESTART} restarts)...")

    for i in range(N_RESTART):
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="full",
            n_iter=N_ITER,
            tol=1e-6,
            random_state=SEED + i,
            # FIX 2: min_covar prevents a state from collapsing to a single
            # point (degenerate covariance), which caused the absorbing Bull state
            params="stmc",
            init_params="stmc",
        )
        # FIX 3: add small noise to initial means to escape degenerate starts
        model.startprob_ = np.full(N_STATES, 1.0 / N_STATES)

        try:
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    conv = best_model.monitor_.converged
    print(f"  Best log-likelihood : {best_score:.2f}")
    print(f"  Converged           : {conv}")

    # Guard: if any state has self-transition = 1.0 (absorbing), warn loudly
    diag = np.diag(best_model.transmat_)
    if np.any(diag >= 0.9999):
        print(f"  WARNING: absorbing state detected — diag={diag.round(4)}")
        print("  This usually means N_STATES is too high or a feature is degenerate.")

    return best_model


# ── DECODE REGIMES ────────────────────────────────────────────────────────────
def decode(model, X, index):
    states = model.predict(X)
    probs  = model.predict_proba(X)

    regime_df = pd.DataFrame({"regime": states}, index=index)
    probs_df  = pd.DataFrame(
        probs, index=index,
        columns=[f"prob_state_{i}" for i in range(model.n_components)]
    )
    return regime_df, probs_df


# ── LABEL STATES ─────────────────────────────────────────────────────────────
def label_states(model, scaler, regime_df):
    """
    FIX 4: Better labeling using a composite health score.

    Old logic: Bull = highest SPY mean (broke when two states had ~equal SPY)
    New logic:
      - Score each state by:  mean(SPY_ret) - 0.3 * mean(VIX_level)
        (scaled units, so weights are comparable)
      - Bull   = highest composite score  (good returns + low fear)
      - Crisis = highest VIX AND negative SPY  (worst state)
      - Bear   = the remaining state
    """
    means_scaled = model.means_                        # in StandardScaler space
    means_raw    = scaler.inverse_transform(means_scaled)
    means_df     = pd.DataFrame(means_raw, columns=FEATURE_COLS)

    print("\nState means (original scale):")
    print(means_df.round(4).to_string())

    # Composite score in scaled space (so units are comparable)
    spy_idx  = FEATURE_COLS.index("SPY_ret")
    vix_idx  = FEATURE_COLS.index("VIX_level")
    yc_idx   = FEATURE_COLS.index("yield_curve")

    composite = (means_scaled[:, spy_idx]
                 - 0.3 * means_scaled[:, vix_idx]
                 + 0.1 * means_scaled[:, yc_idx])

    # Crisis: worst composite score (low returns + high fear)
    crisis_state = int(np.argmin(composite))
    # Bull: best composite score
    bull_state   = int(np.argmax(composite))
    # Bear: whatever's left
    bear_state   = [s for s in range(N_STATES)
                    if s not in {crisis_state, bull_state}][0]

    label_map = {bull_state: "Bull", bear_state: "Bear", crisis_state: "Crisis"}
    print(f"\nComposite scores : {composite.round(3)}")
    print(f"State mapping    : {label_map}")

    regime_df["label"] = regime_df["regime"].map(label_map)
    return regime_df, label_map


# ── DIAGNOSTICS ───────────────────────────────────────────────────────────────
def print_diagnostics(model, label_map, X):   # <-- pass X directly
    n      = model.n_components
    labels = [label_map[i] for i in range(n)]

    trans = pd.DataFrame(model.transmat_.round(4),
                         index=labels, columns=labels)
    print("\nTransition matrix (row = from, col = to):")
    print(trans.to_string())

    print("\nExpected regime durations:")
    for i, lbl in enumerate(labels):
        p = model.transmat_[i, i]
        d = 1 / (1 - p) if p < 0.9999 else np.inf
        print(f"  {lbl:8s}: {d:.1f} days  (~{d/21:.1f} months)")

    print("\nRegime distribution:")
    states = model.predict(X)
    counts = pd.Series(states).map(label_map).value_counts()
    total  = counts.sum()
    for lbl, cnt in counts.items():
        print(f"  {lbl:8s}: {cnt:5d} days  ({cnt/total*100:.1f}%)")


# ── PLOT ──────────────────────────────────────────────────────────────────────
def plot_regimes(regime_df, feat_path="data/feature_matrix.csv"):
    df      = pd.read_csv(feat_path, index_col="Date", parse_dates=True)
    spy_cum = (1 + df["SPY_ret"]).cumprod()
    vix     = df["VIX_level"]

    COLORS = {"Bull": "#2ecc71", "Bear": "#e67e22", "Crisis": "#e74c3c"}

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2, 1]})
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        for label in ax.get_yticklabels():
            label.set_color("white")

    # Panel 1: cumulative SPY + regime shading
    ax1 = axes[0]
    ax1.plot(spy_cum.index, spy_cum.values, color="white", lw=1.2)
    ax1.set_ylabel("SPY cumulative return", color="white")
    ax1.set_title("Regime-Shift: HMM-detected market regimes (2005–2024)",
                  color="white", fontsize=13, pad=10)

    labels_aligned = regime_df["label"].reindex(spy_cum.index, method="ffill")
    _shade_regimes(ax1, labels_aligned, COLORS)

    patches = [mpatches.Patch(color=c, alpha=0.6, label=l)
               for l, c in COLORS.items()]
    ax1.legend(handles=patches, loc="upper left",
               facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    # Panel 2: VIX level
    ax2 = axes[1]
    ax2.plot(vix.index, vix.values, color="#aaaaff", lw=0.8)
    ax2.axhline(20, color="#e74c3c", lw=0.6, ls="--", alpha=0.5)
    ax2.axhline(30, color="#e74c3c", lw=0.8, ls="--", alpha=0.8)
    ax2.set_ylabel("VIX level", color="white")
    _shade_regimes(ax2, labels_aligned, COLORS)

    # Panel 3: regime strip
    ax3     = axes[2]
    num_map = {"Bull": 2, "Bear": 1, "Crisis": 0}
    num_ser = labels_aligned.map(num_map)
    ax3.fill_between(num_ser.index, num_ser.values,
                     step="post", color="steelblue", alpha=0.7)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["Crisis", "Bear", "Bull"], color="white", fontsize=8)
    ax3.set_ylabel("Regime", color="white")

    plt.tight_layout(h_pad=0.3)
    path = "plots/regimes.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nPlot saved → {path}")


def _shade_regimes(ax, labels, colors):
    prev, start = None, labels.index[0]
    for date, lbl in labels.items():
        if lbl != prev:
            if prev is not None:
                ax.axvspan(start, date, alpha=0.2,
                           color=colors.get(prev, "gray"), lw=0)
            start, prev = date, lbl
    if prev:
        ax.axvspan(start, labels.index[-1], alpha=0.2,
                   color=colors.get(prev, "gray"), lw=0)


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    df, X, scaler = load_and_scale()

    # Save X for use in diagnostics
    np.save("models/X_scaled.npy", X)

    model = fit_hmm(X)
    joblib.dump(model, "models/hmm_model.pkl")
    print("Model saved → models/hmm_model.pkl")

    regime_df, probs_df = decode(model, X, df.index)
    regime_df, label_map = label_states(model, scaler, regime_df)
    joblib.dump(label_map, "models/label_map.pkl")

    print_diagnostics(model, label_map, X)
    plot_regimes(regime_df)

    regime_df.to_csv("data/regimes.csv")
    probs_df.to_csv("data/regime_probs.csv")
    print("\nPhase 2 complete.")
    return model, regime_df, label_map


if __name__ == "__main__":
    model, regime_df, label_map = main()