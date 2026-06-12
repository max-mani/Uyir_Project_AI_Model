"""UYIR Threshold Analyzer — analyze data_logger CSV output."""

import argparse
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def analyze(csv_path: str):
    print(f"\n[Analyzer] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    acc = df[df["label"] == "accident"]
    nor = df[df["label"] == "normal"]

    factors = [
        {"col": "nearest_euclidean_dist", "name": "Phase A — Nearest Euclidean Distance (px)",
         "threshold": config.PROXIMITY_THRESHOLD, "direction": "below"},
        {"col": "speed_drop_percent", "name": "Phase B — Speed Drop %",
         "threshold": config.SPEED_DROP_PERCENT, "direction": "above"},
        {"col": "trajectory_deviation_px", "name": "Phase B — Trajectory Deviation (px)",
         "threshold": 40.0, "direction": "above"},
        {"col": "optical_flow_ratio", "name": "Phase C — Optical Flow Ratio",
         "threshold": config.OPTICAL_FLOW_SPIKE, "direction": "above"},
        {"col": "bbox_area_change_ratio", "name": "Phase C — BBox Area Change Ratio",
         "threshold": config.BBOX_DEFORM_RATIO, "direction": "above"},
        {"col": "accident_model_confidence", "name": "Stage 1 — Accident Model Confidence",
         "threshold": config.STAGE1_GATE_CONFIDENCE, "direction": "above"},
    ]

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.30)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(len(factors))]
    suggestions = {}

    for ax, f in zip(axes, factors):
        col = f["col"]
        if col not in df.columns:
            continue
        p99_a = np.percentile(acc[col].dropna(), 99) if len(acc) else 1
        p99_n = np.percentile(nor[col].dropna(), 99) if len(nor) else 1
        clip_val = max(p99_a, p99_n)
        acc_col = acc[col].dropna().clip(upper=clip_val)
        nor_col = nor[col].dropna().clip(upper=clip_val)
        ax.hist(nor_col, bins=50, alpha=0.6, color="#2196F3", label="Normal", density=True)
        ax.hist(acc_col, bins=50, alpha=0.6, color="#F44336", label="Accident", density=True)
        if f["threshold"] is not None:
            ax.axvline(f["threshold"], color="orange", linewidth=2, linestyle="--",
                       label=f"Current = {f['threshold']}")
        best_t, best_f1 = _find_best_threshold(acc_col, nor_col, f["direction"])
        ax.axvline(best_t, color="green", linewidth=1.5, linestyle=":",
                   label=f"Suggested = {best_t:.3f}")
        suggestions[col] = best_t
        ax.set_title(f["name"], fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    out_plot = csv_path.replace(".csv", "_distributions.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"\n[Analyzer] Saved: {out_plot}")
    print("\nSuggested config.py updates:")
    mapping = {
        "nearest_euclidean_dist": "PROXIMITY_THRESHOLD",
        "speed_drop_percent": "SPEED_DROP_PERCENT",
        "optical_flow_ratio": "OPTICAL_FLOW_SPIKE",
        "bbox_area_change_ratio": "BBOX_DEFORM_RATIO",
        "accident_model_confidence": "STAGE1_GATE_CONFIDENCE",
    }
    for col, cfg_key in mapping.items():
        if col in suggestions:
            print(f"  {cfg_key:35s} = {suggestions[col]:.3f}")


def _find_best_threshold(pos_vals, neg_vals, direction: str):
    all_vals = pd.concat([pos_vals, neg_vals]).dropna().sort_values()
    if len(all_vals) == 0:
        return 0.5, 0.0
    best_t = float(all_vals.median())
    best_f1 = 0.0
    lo = float(all_vals.quantile(0.05))
    hi = float(all_vals.quantile(0.95))
    for t in np.linspace(lo, hi, 60):
        if direction == "above":
            tp = (pos_vals > t).sum(); fp = (neg_vals > t).sum(); fn = (pos_vals <= t).sum()
        else:
            tp = (pos_vals < t).sum(); fp = (neg_vals < t).sum(); fn = (pos_vals >= t).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1; best_t = t
    return best_t, best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UYIR Threshold Analyzer")
    parser.add_argument("--csv", required=True, help="CSV file from data_logger.py")
    args = parser.parse_args()
    if not os.path.exists(args.csv):
        print(f"[Analyzer] File not found: {args.csv}")
    else:
        analyze(args.csv)
