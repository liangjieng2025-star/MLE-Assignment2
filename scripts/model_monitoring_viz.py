import matplotlib
matplotlib.use("Agg")           # headless backend — must be set before pyplot import

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import glob
import os
import argparse

# to call: python model_monitoring_viz.py --modelname "credit_model_2024_09_01.pkl"


def main(modelname, output_dir="monitoring_plots"):
    os.chdir("/opt/airflow")
    model_name_no_ext = modelname.replace(".pkl", "")
    monitoring_dir    = os.path.join("datamart", "gold", "model_monitoring", model_name_no_ext)

    # load full monitoring table (one plain parquet per month)
    files = sorted(glob.glob(
        os.path.join(monitoring_dir, "{}_monitoring_*.parquet".format(model_name_no_ext))
    ))
    if not files:
        raise FileNotFoundError("No monitoring files found in: {}".format(monitoring_dir))

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df = df.sort_values("snapshot_date").reset_index(drop=True)
    print("Loaded {} monitoring rows ({} to {})".format(
        len(df),
        df["snapshot_date"].min().strftime("%Y-%m-%d"),
        df["snapshot_date"].max().strftime("%Y-%m-%d"),
    ))

    os.makedirs(output_dir, exist_ok=True)

    date_fmt = mdates.DateFormatter("%Y-%m")
    plt.rcParams.update({"font.size": 11, "axes.titlepad": 10})


    # Plot 1: Performance over time (AUC & Gini) 
    fig, ax = plt.subplots(figsize=(13, 5))

    valid = df[df["auc"].notna()].copy()
    ax.plot(valid["snapshot_date"], valid["auc"],
            "o-", color="steelblue", linewidth=2, markersize=7, label="AUC")
    ax.plot(valid["snapshot_date"], valid["gini"],
            "s--", color="darkorange", linewidth=2, markersize=6, label="Gini")

    null_months = df[df["auc"].isna()]["snapshot_date"]
    if len(null_months):
        ylim_bottom = min(df["gini"].dropna().min() - 0.05, 0.0)
        ax.scatter(null_months, [ylim_bottom] * len(null_months),
                   marker="^", facecolors="none", edgecolors="grey",
                   s=70, zorder=5, label="No labels (mob-6 not matured)")

    ax.set_title("Model Performance Over Time (AUC & Gini)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Snapshot Date", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(date_fmt)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "1_performance_over_time.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: {}".format(p1))


    # Plot 2: Score distribution stability (PSI) 
    fig, ax = plt.subplots(figsize=(13, 5))

    psi_valid = df[df["psi"].notna()].copy()
    psi_valid["psi"] = psi_valid["psi"].astype(float)
    ax.plot(psi_valid["snapshot_date"], psi_valid["psi"],
            "o-", color="purple", linewidth=2, markersize=7)
    ax.axhline(0.10, linestyle="--", color="goldenrod",  linewidth=1.8,
               label="PSI = 0.10  (watch)")
    ax.axhline(0.25, linestyle="--", color="crimson",    linewidth=1.8,
               label="PSI = 0.25  (drift — investigate)")
    ax.fill_between(psi_valid["snapshot_date"], 0.10, 0.25,
                    alpha=0.06, color="goldenrod", label="_nolegend_")
    ax.fill_between(psi_valid["snapshot_date"],
                    psi_valid["psi"].clip(lower=0.25), 0.25,
                    where=(psi_valid["psi"] > 0.25),
                    alpha=0.12, color="crimson", label="_nolegend_")

    ax.set_title("Score Distribution Stability Over Time (PSI)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Snapshot Date", fontsize=12)
    ax.set_ylabel("Population Stability Index (PSI)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(date_fmt)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "2_psi_stability.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: {}".format(p2))


    # Plot 3: Coverage sanity (n_scored + actual_bad_rate) 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"hspace": 0.12})

    # top panel: prediction volume
    x_num = mdates.date2num(df["snapshot_date"].dt.to_pydatetime())
    ax1.bar(x_num, df["n_scored"], width=18, color="steelblue", alpha=0.75,
            label="n_scored")
    ax1.set_title("Coverage Sanity: Prediction Volume & Actual Bad Rate",
                  fontsize=14, fontweight="bold")
    ax1.set_ylabel("Predictions Count (n_scored)", fontsize=12)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.xaxis_date()

    # bottom panel: actual bad rate from labelled subset
    br_valid = df[df["actual_bad_rate"].notna()].copy()
    ax2.plot(br_valid["snapshot_date"], br_valid["actual_bad_rate"],
             "o-", color="crimson", linewidth=2, markersize=7, label="Actual bad rate")
    ax2.set_ylabel("Actual Bad Rate (labelled subset)", fontsize=12)
    ax2.set_xlabel("Snapshot Date", fontsize=12)
    ax2.legend(fontsize=11, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(date_fmt)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    p3 = os.path.join(output_dir, "3_coverage_sanity.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: {}".format(p3))


    # Plot 4: Score drift (mean predicted probability) 
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(df["snapshot_date"], df["mean_pred"],
            "o-", color="teal", linewidth=2, markersize=7)
    overall_mean = df["mean_pred"].mean()
    ax.axhline(overall_mean, linestyle=":", color="grey", linewidth=1.5,
               label="Overall mean = {:.4f}".format(overall_mean))

    ax.set_title("Score Drift: Mean Predicted Probability Over Time",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Snapshot Date", fontsize=12)
    ax.set_ylabel("Mean Predicted Probability", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(date_fmt)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    p4 = os.path.join(output_dir, "4_score_drift.png")
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: {}".format(p4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument(
        "--modelname",   type=str, required=True,
        help="model pkl filename e.g. credit_model_2024_09_01.pkl",
    )
    parser.add_argument(
        "--output_dir",  type=str, required=False, default="monitoring_plots",
        help="directory to save PNGs (default: monitoring_plots)",
    )
    args = parser.parse_args()
    main(args.modelname, args.output_dir)
