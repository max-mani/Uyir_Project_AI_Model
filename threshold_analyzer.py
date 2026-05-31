"""
UYIR — Threshold Analyzer
Manikandan's analytical tool for finding correct threshold values.

After running data_logger.py on multiple accident and normal clips,
run this script on the generated CSV to find optimal thresholds.

Usage:
    python threshold_analyzer.py --csv data_logs/uyir_data_log.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import argparse
import os

from paths import DEFAULT_DATA_LOG, OUTPUTS_DIR, ensure_dirs


def analyze(csv_path: str):
    ensure_dirs()
    print(f"\n[Analyzer] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"[Analyzer] Total rows:     {len(df)}")
    print(f"[Analyzer] Accident rows:  {len(df[df['label']=='accident'])}")
    print(f"[Analyzer] Normal rows:    {len(df[df['label']=='normal'])}")
    print(f"[Analyzer] Unique videos:  {df['video_file'].nunique()}")

    accident = df[df["label"] == "accident"]
    normal   = df[df["label"] == "normal"]

    factors = [
        {
            "col":       "iou_with_nearest",
            "name":      "Factor 1 — IOU Overlap",
            "threshold": 0.5,
            "x_max":     1.0,
        },
        {
            "col":       "speed_drop_percent",
            "name":      "Factor 2 — Speed Drop %",
            "threshold": 70.0,
            "x_max":     100.0,
        },
        {
            "col":       "trajectory_deviation_px",
            "name":      "Factor 3 — Trajectory Deviation (px)",
            "threshold": 40.0,
            "x_max":     None,
        },
        {
            "col":       "optical_flow_ratio",
            "name":      "Factor 4 — Optical Flow Ratio",
            "threshold": 2.5,
            "x_max":     None,
        },
        {
            "col":       "speed_px_per_frame",
            "name":      "Speed (px/frame) — reference",
            "threshold": None,
            "x_max":     None,
        }
    ]

    # ── Plot distributions ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("UYIR — Threshold Analysis\nAccident vs Normal Traffic Factor Distributions",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(len(factors))]

    for ax, factor in zip(axes, factors):
        col     = factor["col"]
        acc_col = accident[col].dropna().clip(upper=np.percentile(accident[col].dropna(), 99))
        nor_col = normal[col].dropna().clip(upper=np.percentile(normal[col].dropna(), 99))

        ax.hist(nor_col,      bins=50, alpha=0.6, color="#2196F3",
                label=f"Normal (n={len(nor_col)})",  density=True)
        ax.hist(acc_col,      bins=50, alpha=0.6, color="#F44336",
                label=f"Accident (n={len(acc_col)})", density=True)

        if factor["threshold"] is not None:
            ax.axvline(factor["threshold"], color="orange", linewidth=2,
                       linestyle="--", label=f"Threshold = {factor['threshold']}")

        ax.set_title(factor["name"], fontweight="bold", fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        if factor["x_max"]:
            ax.set_xlim(0, factor["x_max"])

        # Print statistics
        print(f"\n── {factor['name']} ──")
        print(f"   Normal  mean={nor_col.mean():.3f}  median={nor_col.median():.3f}  "
              f"p90={np.percentile(nor_col,90):.3f}  p99={np.percentile(nor_col,99):.3f}")
        print(f"   Accident mean={acc_col.mean():.3f} median={acc_col.median():.3f}  "
              f"p90={np.percentile(acc_col,90):.3f}  p99={np.percentile(acc_col,99):.3f}")

        if factor["threshold"] is not None:
            tp_rate = (acc_col > factor["threshold"]).mean() * 100
            fp_rate = (nor_col > factor["threshold"]).mean() * 100
            print(f"   At threshold {factor['threshold']}:")
            print(f"     → Catches {tp_rate:.1f}% of accident frames  (want HIGH)")
            print(f"     → False alarm on {fp_rate:.1f}% of normal frames  (want LOW)")

    threshold_plot = OUTPUTS_DIR / "threshold_analysis.png"
    plt.savefig(threshold_plot, dpi=150, bbox_inches="tight")
    print(f"\n[Analyzer] Plot saved: {threshold_plot}")

    # ── Score distribution ────────────────────────────────────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("UYIR — Total Factor Score Distribution",
                  fontsize=13, fontweight="bold")

    max_score = df["total_factor_score"].max()
    bins      = range(0, int(max_score) + 2)

    ax1.hist(normal["total_factor_score"].dropna(),   bins=bins,
             color="#2196F3", alpha=0.8, edgecolor="white")
    ax1.axvline(4, color="orange", linewidth=2, linestyle="--",
                label="Score threshold = 4")
    ax1.set_title("Score Distribution — Normal Traffic")
    ax1.set_xlabel("Total Score"); ax1.set_ylabel("Frame Count")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.hist(accident["total_factor_score"].dropna(), bins=bins,
             color="#F44336", alpha=0.8, edgecolor="white")
    ax2.axvline(4, color="orange", linewidth=2, linestyle="--",
                label="Score threshold = 4")
    ax2.set_title("Score Distribution — Accident Clips")
    ax2.set_xlabel("Total Score"); ax2.set_ylabel("Frame Count")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    score_plot = OUTPUTS_DIR / "score_distribution.png"
    plt.savefig(score_plot, dpi=150, bbox_inches="tight")
    print(f"[Analyzer] Score plot saved: {score_plot}")

    # ── Recommendation ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  THRESHOLD RECOMMENDATION SUMMARY")
    print("="*60)
    for factor in factors:
        if factor["threshold"] is None:
            continue
        col       = factor["col"]
        acc_vals  = accident[col].dropna()
        nor_vals  = normal[col].dropna()
        best_t, best_f1 = _find_best_threshold(acc_vals, nor_vals)
        print(f"\n  {factor['name']}")
        print(f"    Starting threshold: {factor['threshold']}")
        print(f"    Data-suggested:     {best_t:.3f}  (F1 score: {best_f1:.3f})")
    print("="*60 + "\n")
    plt.show()


def _find_best_threshold(pos_vals, neg_vals):
    """Find threshold that maximises F1 score."""
    all_vals = pd.concat([pos_vals, neg_vals]).dropna().sort_values()
    best_t   = 0.0
    best_f1  = 0.0

    for t in np.linspace(all_vals.quantile(0.1), all_vals.quantile(0.9), 50):
        tp = (pos_vals > t).sum()
        fp = (neg_vals > t).sum()
        fn = (pos_vals <= t).sum()
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    return best_t, best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UYIR Threshold Analyzer")
    parser.add_argument("--csv", default=str(DEFAULT_DATA_LOG),
                        help="CSV file from data_logger.py")
    args = parser.parse_args()
    analyze(args.csv)
