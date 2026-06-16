"""
Generate all deck-ready PNGs into deck_assets/ (200 DPI, 16:9, white bg).
Run from the repo root: python scripts/gen_deck_assets.py

Outputs
-------
deck_assets/1_architecture.png          end-to-end pipeline diagram
deck_assets/2_medallion_dataflow.png    bronze / silver / gold layer map
deck_assets/3_data_leakage_timeline.png temporal alignment illustration
deck_assets/4_model_selection.png       model comparison grouped bar chart
deck_assets/5_performance_over_time.png monitoring AUC & Gini  (200 DPI re-export)
deck_assets/6_psi_stability.png         monitoring PSI          (200 DPI re-export)
deck_assets/7_coverage_sanity.png       monitoring n_scored + bad rate
deck_assets/8_score_drift.png           monitoring mean prediction
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import glob
import os

# palette
BLU  = "#2B6CB0"
BLUM = "#4299E1"
BLUL = "#BEE3F8"
ORG  = "#DD6B20"
ORGL = "#FEEBC8"
GRN  = "#276749"
RED  = "#C53030"
PUR  = "#553C9A"
GRY  = "#718096"
GRYL = "#E2E8F0"
BLK  = "#1A202C"
WHT  = "#FFFFFF"
BG   = "#EBF4FF"

DPI = 200
OUT = "deck_assets"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.facecolor": WHT,
    "axes.facecolor": WHT,
    "grid.color": GRYL,
    "grid.linewidth": 0.8,
})


def _box(ax, cx, cy, w, h, title, sub=None,
         fill=BLU, tc=WHT, fs=9.5, z=3):
    r = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=fill, edgecolor=GRY,
        linewidth=0.9, zorder=z,
    )
    ax.add_patch(r)
    if sub:
        ax.text(cx, cy + h * 0.17, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=z + 1)
        ax.text(cx, cy - h * 0.22, sub, ha="center", va="center",
                fontsize=fs - 1.5, color=tc, alpha=0.88, zorder=z + 1)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=z + 1)


def _arr(ax, x1, y1, x2, y2, color=BLK, lw=1.4, rad=0.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            mutation_scale=13,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=6,
    )


# ── 1. Architecture diagram ───────────────────────────────────────────────────
def draw_architecture():
    fig = plt.figure(figsize=(16, 9), facecolor=WHT)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # outer Docker/Airflow frame — top at y=0.93 so title has clearance
    frame = mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.92,
        boxstyle="round,pad=0.01",
        facecolor=BG, edgecolor=BLU, linewidth=2.2,
        linestyle="--", zorder=1,
    )
    ax.add_patch(frame)
    ax.text(0.5, 0.967,
            "Docker Container  ·  Apache Airflow 2.6.1  ·  Python 3.7",
            ha="center", va="center", fontsize=10.5,
            color=BLU, fontweight="bold")

    # Airflow task labels across the top
    tasks   = ["build_datamart", "train_select", "inference_backfill",
               "monitoring_backfill", "visualise"]
    task_xs = [0.17, 0.34, 0.51, 0.70, 0.88]
    for tx, tl in zip(task_xs, tasks):
        ax.text(tx, 0.910, tl, ha="center", va="center",
                fontsize=7.5, color=GRY, fontstyle="italic")
    ax.axhline(0.898, xmin=0.03, xmax=0.97, color=GRYL, lw=1.0)

    # ── Row 1: Raw data sources ────────────────────────────────────────────────
    y_s = 0.865
    srcs = [
        ("lms_loan_daily.csv", "daily loan records"),
        ("feature_clickstream.csv", "fe_1..fe_20 signals"),
        ("features_attributes.csv", "demographics"),
        ("features_financials.csv", "credit bureau"),
    ]
    sxs = [0.13, 0.37, 0.63, 0.87]
    for sx, (s, d) in zip(sxs, srcs):
        _box(ax, sx, y_s, 0.20, 0.065, s, d, fill="#4A5568", fs=8.2)

    # ── Row 2: Bronze ─────────────────────────────────────────────────────────
    y_b = 0.748
    _box(ax, 0.50, y_b, 0.88, 0.068,
         "BRONZE  (datamart/bronze/)",
         "Partitioned CSVs  ·  no cleaning  ·  96 files  (24 months × 4 sources)",
         fill="#B7791F", fs=9.5)

    # ── Row 3: Silver ─────────────────────────────────────────────────────────
    y_sil = 0.625
    _box(ax, 0.50, y_sil, 0.88, 0.068,
         "SILVER  (datamart/silver/)",
         "Type-cast  ·  +mob +dpd (LMS)  ·  negatives→null (fin)  ·  Parquet  ·  96 files",
         fill="#4A5568", fs=9.5)

    # ── Row 4: Gold layer (3 stores) ──────────────────────────────────────────
    y_g = 0.502
    _box(ax, 0.22, y_g, 0.27, 0.068,
         "label_store/",
         "dpd≥30 @ mob=6  |  ~499 rows/month (matured)",
         fill=GRN, fs=8.5)
    _box(ax, 0.50, y_g, 0.22, 0.068,
         "feature_store/eng/",
         "6-month fe_1 rollup",
         fill="#2F855A", fs=8.5)
    _box(ax, 0.76, y_g, 0.26, 0.068,
         "feature_store/cust_fin_risk/",
         "Credit KPIs",
         fill="#2F855A", fs=8.5)

    # ── Row 5: Train + model bank ─────────────────────────────────────────────
    y_t = 0.365
    _box(ax, 0.27, y_t, 0.35, 0.075,
         "model_train.py",
         "LR · RF · XGBoost  →  winner by OOT AUC",
         fill=BLU, fs=9)
    _box(ax, 0.70, y_t, 0.30, 0.075,
         "model_bank/",
         "credit_model_2024_09_01.pkl  (LogisticRegression)",
         fill=BLUM, fs=8.8)

    # ── Row 6: Inference + Monitoring ─────────────────────────────────────────
    y_i = 0.225
    _box(ax, 0.27, y_i, 0.35, 0.072,
         "model_inference.py  (×24)",
         "scores ~8,974 customers / month",
         fill="#2C5282", fs=9)
    _box(ax, 0.70, y_i, 0.30, 0.072,
         "model_monitoring.py  (×24)",
         "AUC/Gini + PSI per month",
         fill="#2A4365", fs=9)

    # ── Row 7: Output gold tables ─────────────────────────────────────────────
    y_o = 0.09
    _box(ax, 0.27, y_o, 0.35, 0.060,
         "gold/model_predictions/  (24 parquets)", None,
         fill="#1A365D", fs=8.5)
    _box(ax, 0.70, y_o, 0.30, 0.060,
         "gold/model_monitoring/ + monitoring_plots/", None,
         fill="#1A365D", fs=8.5)

    # ── Arrows ────────────────────────────────────────────────────────────────
    # sources → bronze
    for sx in sxs:
        _arr(ax, sx, y_s - 0.033, sx, y_b + 0.034)
    # bronze → silver
    _arr(ax, 0.50, y_b - 0.034, 0.50, y_sil + 0.034)
    # silver → three gold stores
    _arr(ax, 0.36, y_sil - 0.034, 0.22, y_g + 0.034)
    _arr(ax, 0.50, y_sil - 0.034, 0.50, y_g + 0.034)
    _arr(ax, 0.64, y_sil - 0.034, 0.76, y_g + 0.034)
    # label_store → model_train
    _arr(ax, 0.22, y_g - 0.034, 0.22, y_t + 0.038)
    ax.annotate("", xy=(0.27 - 0.175, y_t), xytext=(0.22, y_t),
                arrowprops=dict(arrowstyle="-|>", color=BLK, lw=1.2,
                                mutation_scale=12), zorder=6)
    # raw clickstream → model_train (bypass gold, orange)
    ax.annotate(
        "", xy=(0.185, y_t + 0.020),
        xytext=(sxs[1], y_s - 0.033),
        arrowprops=dict(arrowstyle="-|>", color=ORG, lw=1.3,
                        mutation_scale=12,
                        connectionstyle="arc3,rad=-0.30"), zorder=6,
    )
    ax.text(0.08, 0.59, "raw CSV\n(train/serve\nconsistency)",
            fontsize=7.2, color=ORG, ha="center", fontstyle="italic")
    # model_train → model_bank
    _arr(ax, 0.27 + 0.175, y_t, 0.70 - 0.15, y_t)
    # model_bank → inference
    _arr(ax, 0.62, y_t - 0.010, 0.45, y_i + 0.036)
    # model_bank → monitoring
    _arr(ax, 0.70, y_t - 0.038, 0.70, y_i + 0.036)
    # inference → predictions
    _arr(ax, 0.27, y_i - 0.036, 0.27, y_o + 0.030)
    # predictions → monitoring
    _arr(ax, 0.27 + 0.175, y_o, 0.70 - 0.15, y_o)
    # label_store → monitoring (green curved)
    ax.annotate(
        "", xy=(0.555, y_i + 0.020),
        xytext=(0.22, y_g - 0.034),
        arrowprops=dict(arrowstyle="-|>", color=GRN, lw=1.2,
                        mutation_scale=11,
                        connectionstyle="arc3,rad=0.28"), zorder=6,
    )
    # monitoring → output
    _arr(ax, 0.70, y_i - 0.036, 0.70, y_o + 0.030)

    plt.savefig(os.path.join(OUT, "1_architecture.png"),
                dpi=DPI, bbox_inches="tight", facecolor=WHT)
    plt.close(fig)
    print("Saved: deck_assets/1_architecture.png")


# ── 2. Medallion data-flow map ────────────────────────────────────────────────
def draw_medallion():
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=WHT)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Medallion Architecture — Bronze → Silver → Gold",
                 fontsize=15, fontweight="bold", color=BLK, pad=16)

    cols = [
        ("#B7791F", "BRONZE", [
            "datamart/bronze/lms/",
            "datamart/bronze/clks/",
            "datamart/bronze/attr/",
            "datamart/bronze/fin/",
            "",
            "FORMAT: CSV",
            "96 files (24 months × 4 sources)",
            "",
            "No cleaning applied.",
            "Exact replica of source data.",
            "Purpose: raw immutable archive.",
        ]),
        ("#4A5568", "SILVER", [
            "datamart/silver/lms/   + mob, dpd",
            "datamart/silver/clks/  negatives→0",
            "datamart/silver/attr/  name cleaned",
            "datamart/silver/fin/   sign-fix, caps",
            "",
            "FORMAT: Parquet",
            "96 files (24 months × 4 sources)",
            "",
            "Schema enforced, types cast.",
            "Derived columns added (mob, dpd).",
            "Outliers capped, strings parsed.",
        ]),
        (GRN, "GOLD", [
            "label_store/",
            "  mob=6 rows  |  dpd≥30 → label=1",
            "  ~499 per matured month (18 of 24)",
            "",
            "feature_store/eng/",
            "  6-month fe_1 rollup",
            "  click_1m .. click_6m columns",
            "",
            "feature_store/cust_fin_risk/",
            "  debt/salary, EMI/salary, etc.",
            "",
            "FORMAT: Parquet",
            "Business-ready, query-optimised.",
        ]),
    ]

    cw = 0.82
    for i, (fill, title, lines) in enumerate(cols):
        cx = 0.5 + i

        header = mpatches.FancyBboxPatch(
            (cx - cw / 2 + 0.04, 0.83), cw - 0.08, 0.11,
            boxstyle="round,pad=0.015",
            facecolor=fill, edgecolor="none", zorder=2,
        )
        ax.add_patch(header)
        ax.text(cx, 0.895, title, ha="center", va="center",
                fontsize=14, fontweight="bold", color=WHT, zorder=3)

        body = mpatches.FancyBboxPatch(
            (cx - cw / 2 + 0.04, 0.04), cw - 0.08, 0.77,
            boxstyle="round,pad=0.015",
            facecolor="#F7FAFC", edgecolor=fill, linewidth=1.8, zorder=2,
        )
        ax.add_patch(body)

        y_t = 0.77
        for line in lines:
            indent = line.startswith("  ")
            bold = any(line.startswith(k) for k in
                       ("FORMAT", "96 files", "No clean", "Schema",
                        "Business", "Derived", "Outlier"))
            mono = "{" in line or "/" in line or "+" in line
            ax.text(
                cx - cw / 2 + 0.09 + (0.04 if indent else 0), y_t,
                line.strip(),
                fontsize=8.6,
                color=GRY if indent else (BLK if not bold else "#2D3748"),
                fontweight="bold" if bold else "normal",
                fontfamily="monospace" if mono else "DejaVu Sans",
                ha="left", va="top", zorder=3,
            )
            y_t -= 0.052

        # arrows between columns
        if i < 2:
            _arr(ax, cx + cw / 2 - 0.04 + 0.005, 0.45,
                 cx + cw / 2 - 0.04 + 0.075, 0.45, lw=2.0)
            ax.text(cx + 0.5, 0.40, "clean +\ntype-cast",
                    ha="center", va="center", fontsize=8, color=GRY)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "2_medallion_dataflow.png"),
                dpi=DPI, bbox_inches="tight", facecolor=WHT)
    plt.close(fig)
    print("Saved: deck_assets/2_medallion_dataflow.png")


# ── 3. Temporal alignment / data-leakage timeline ────────────────────────────
def draw_leakage_timeline():
    fig, ax = plt.subplots(figsize=(16, 6.5), facecolor=WHT)
    ax.set_xlim(-0.8, 7.5)
    ax.set_ylim(-0.6, 3.8)
    ax.axis("off")
    ax.set_title("Temporal Alignment: Features vs Label Window",
                 fontsize=15, fontweight="bold", color=BLK, pad=18)

    # month grid
    for m in range(7):
        ax.axvline(m, color=GRYL, lw=1.0, zorder=0)
    month_labels = ["Month 0\n(Origination)", "1", "2", "3", "4", "5",
                    "Month 6\n(Label date)"]
    for m, ml in enumerate(month_labels):
        ax.text(m, -0.45, ml, ha="center", va="top", fontsize=9, color=GRY)

    # ── Ideal: application-time model ────────────────────────────────────────
    y_a = 2.85
    ax.text(-0.75, y_a, "Ideal\n(apply at\norigination)",
            ha="right", va="center", fontsize=9, color=GRN, fontweight="bold")

    ax.barh(y_a, 0.65, left=-0.1, height=0.38, color=GRN, alpha=0.75, zorder=3)
    ax.text(0.23, y_a + 0.25, "Features\n@ origination",
            ha="center", va="center", fontsize=8, color=GRN, fontweight="bold")
    ax.barh(y_a, 5.5, left=0.6, height=0.14, color=GRYL, alpha=0.9, zorder=2)
    ax.barh(y_a, 0.65, left=5.85, height=0.38, color=RED, alpha=0.7, zorder=3)
    ax.text(6.2, y_a + 0.25, "Label\n@ mob=6",
            ha="center", va="center", fontsize=8, color=RED, fontweight="bold")
    _arr(ax, 0.6, y_a, 5.85, y_a, color=GRN, lw=1.4)
    ax.text(3.2, y_a - 0.08,
            "Predict default before it happens  →  true application-time model",
            ha="center", va="top", fontsize=8.5, color=GRN, fontstyle="italic")

    # ── Our approach ─────────────────────────────────────────────────────────
    y_b = 1.65
    ax.text(-0.75, y_b, "Our approach\n(features @\nmob=6)",
            ha="right", va="center", fontsize=9, color=ORG, fontweight="bold")

    ax.barh(y_b, 5.5, left=0.1, height=0.14, color=GRYL, alpha=0.9, zorder=2)
    ax.barh(y_b, 0.65, left=5.85, height=0.38, color=ORG, alpha=0.80, zorder=3)
    ax.text(6.2, y_b + 0.25, "Features\n@ mob=6",
            ha="center", va="center", fontsize=8, color=ORG, fontweight="bold")
    ax.barh(y_b, 0.65, left=5.85, height=0.38, color=RED, alpha=0.30, zorder=4)
    ax.text(6.2, y_b - 0.28, "Label\n@ mob=6",
            ha="center", va="top", fontsize=8, color=RED, fontweight="bold")

    # overlap bracket
    ax.annotate("", xy=(5.85, y_b + 0.48), xytext=(6.5, y_b + 0.48),
                arrowprops=dict(arrowstyle="|-|", color=ORG, lw=1.5,
                                mutation_scale=5))
    ax.text(6.18, y_b + 0.57, "same month", ha="center",
            fontsize=7.5, color=ORG)

    # warning box
    warn = mpatches.FancyBboxPatch(
        (1.0, y_b - 0.62), 4.2, 0.40,
        boxstyle="round,pad=0.03",
        facecolor="#FFF5F5", edgecolor=RED, lw=1.3, zorder=3,
    )
    ax.add_patch(warn)
    ax.text(3.1, y_b - 0.42,
            "⚠  Features & label are contemporaneous — not a causal application-time predictor.\n"
            "   Not target leakage (fe_* are independent of DPD), but the model reflects\n"
            "   month-6 behaviour, not loan-origination risk.",
            ha="center", va="center", fontsize=8.2, color=RED)

    # ── What the model can still do ───────────────────────────────────────────
    y_c = 0.50
    action = mpatches.FancyBboxPatch(
        (0.0, y_c - 0.32), 7.0, 0.44,
        boxstyle="round,pad=0.03",
        facecolor="#EBF8FF", edgecolor=BLU, lw=1.6, zorder=3,
    )
    ax.add_patch(action)
    ax.text(3.5, y_c - 0.10,
            "✓  Still actionable:  scoring at mob=6 lets the bank flag high-risk customers"
            " before they reach 90 DPD (NPL status).\n"
            "   Early intervention at month 6 (collections, restructuring, payment holiday)"
            " is far more cost-effective than loss recovery at month 12+.",
            ha="center", va="center", fontsize=9.0, color=BLU)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "3_data_leakage_timeline.png"),
                dpi=DPI, bbox_inches="tight", facecolor=WHT)
    plt.close(fig)
    print("Saved: deck_assets/3_data_leakage_timeline.png")


# ── 4. Model-selection bar chart ─────────────────────────────────────────────
def draw_model_selection():
    models = ["Logistic\nRegression", "Random\nForest", "XGBoost"]
    train  = [0.6555, 0.7880, 0.7569]
    test   = [0.6457, 0.6210, 0.6190]
    oot    = [0.6297, 0.6140, 0.6265]
    gaps   = [tr - ot for tr, ot in zip(train, oot)]

    x = np.arange(len(models))
    w = 0.23

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=WHT)

    b1 = ax.bar(x - w, train, w, label="Train AUC", color=BLU,  alpha=0.90, zorder=3)
    b2 = ax.bar(x,     test,  w, label="Test AUC",  color=BLUM, alpha=0.90, zorder=3)
    b3 = ax.bar(x + w, oot,   w, label="OOT AUC",   color=ORG,  alpha=0.90, zorder=3)

    # value labels
    for bars, vals in [(b1, train), (b2, test), (b3, oot)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=7.8, fontweight="bold", color=BLK)

    # highlight winner OOT bar
    b3[0].set_edgecolor(GRN)
    b3[0].set_linewidth(2.8)
    ax.text(x[0] + w, oot[0] + 0.026, "SELECTED",
            ha="center", fontsize=8.5, color=GRN, fontweight="bold")

    # overfit gap annotation — text boxes to the right of each group (no arrow/label collision)
    for i, (tr, ot, gap) in enumerate(zip(train, oot, gaps)):
        col = RED if gap > 0.05 else GRN
        ax.text(x[i] + w + 0.10, (tr + ot) / 2,
                f"gap\n+{gap:.3f}",
                fontsize=7.8, color=col, ha="left", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor=WHT,
                          edgecolor=col, linewidth=0.9, alpha=0.92))

    ax.axhline(0.5, color=GRY, lw=1.0, linestyle=":", alpha=0.5)
    ax.text(0.03, 0.503, "Random (AUC = 0.50)",
            fontsize=8, color=GRY, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12.5)
    ax.set_ylabel("AUC Score", fontsize=12)
    ax.set_ylim(0.46, 0.87)
    ax.set_xlim(-0.55, 3.05)
    ax.set_title(
        "Model Selection — Train / Test / OOT AUC\n"
        "LR wins: smallest overfit gap (train−OOT = +0.026) vs RF (+0.174) and XGBoost (+0.130)",
        fontsize=13, fontweight="bold", color=BLK, pad=12,
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, axis="y", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "4_model_selection.png"),
                dpi=DPI, bbox_inches="tight", facecolor=WHT)
    plt.close(fig)
    print("Saved: deck_assets/4_model_selection.png")


# ── 5-8. Re-export monitoring PNGs at 200 DPI ────────────────────────────────
def reexport_monitoring():
    mon_dir = os.path.join("datamart", "gold", "model_monitoring",
                           "credit_model_2024_09_01")
    files = sorted(glob.glob(
        os.path.join(mon_dir, "credit_model_2024_09_01_monitoring_*.parquet")
    ))
    if not files:
        print(f"WARNING: no monitoring files at {mon_dir} — skipping re-export.")
        return

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df = df.sort_values("snapshot_date").reset_index(drop=True)
    for c in ["auc", "gini", "psi", "mean_pred", "actual_bad_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"Loaded {len(df)} monitoring rows.")

    datefmt = mdates.DateFormatter("%Y-%m")

    # 5. Performance over time
    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=WHT)
    v = df[df["auc"].notna()]
    ax.plot(v["snapshot_date"], v["auc"],
            "o-", color=BLU, lw=2, ms=7, label="AUC")
    ax.plot(v["snapshot_date"], v["gini"],
            "s--", color=ORG, lw=2, ms=6, label="Gini")
    null_m = df[df["auc"].isna()]["snapshot_date"]
    if len(null_m):
        ybot = min(df["gini"].dropna().min() - 0.05, 0.0)
        ax.scatter(null_m, [ybot] * len(null_m),
                   marker="^", facecolors="none", edgecolors=GRY,
                   s=70, zorder=5, label="No labels (mob-6 not matured)")
    ax.set_title("Model Performance Over Time (AUC & Gini)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Snapshot Date", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(datefmt)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "5_performance_over_time.png"),
                dpi=DPI, bbox_inches="tight", facecolor=WHT)
    plt.close(fig)
    print("Saved: deck_assets/5_performance_over_time.png")

    # 6. PSI stability
    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=WHT)
    pv = df[df["psi"].notna()].copy()
    pv["psi"] = pv["psi"].astype(float)
    ax.plot(pv["snapshot_date"], pv["psi"],
            "o-", color=PUR, lw=2, ms=7)
    ax.axhline(0.10, linestyle="--", color="#D69E2E", lw=1.8,
               label="PSI = 0.10  (watch)")
    ax.axhline(0.25, linestyle="--", color=RED,       lw=1.8,
               label="PSI = 0.25  (investigate)")
    ax.fill_between(pv["snapshot_date"], 0.10, 0.25,
                    alpha=0.07, color="#D69E2E")
    ax.fill_between(pv["snapshot_date"],
                    pv["psi"].clip(lower=0.25), 0.25,
                    where=(pv["psi"] > 0.25),
                    alpha=0.13, color=RED)

    # annotate maximum PSI value
    max_row = pv.loc[pv["psi"].idxmax()]
    ax.annotate(
        f"max PSI = {max_row['psi']:.4f}\n({pd.Timestamp(max_row['snapshot_date']).strftime('%Y-%m-%d')})",
        xy=(max_row["snapshot_date"], float(max_row["psi"])),
        xytext=(max_row["snapshot_date"], 0.032),
        ha="center", fontsize=8.5, color=PUR, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PUR, lw=1.0),
    )

    ax.set_title("Score Distribution Stability Over Time (PSI)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Snapshot Date", fontsize=12)
    ax.set_ylabel("Population Stability Index (PSI)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(datefmt)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "6_psi_stability.png"),
                dpi=DPI, bbox_inches="tight", facecolor=WHT)
    plt.close(fig)
    print("Saved: deck_assets/6_psi_stability.png")

    # 7. Coverage sanity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                    gridspec_kw={"hspace": 0.12},
                                    facecolor=WHT)
    x_num = mdates.date2num(df["snapshot_date"].dt.to_pydatetime())
    ax1.bar(x_num, df["n_scored"], width=18, color=BLU, alpha=0.75, label="n_scored")
    ax1.set_title("Coverage Sanity: Prediction Volume & Actual Bad Rate",
                  fontsize=14, fontweight="bold")
    ax1.set_ylabel("Predictions (n_scored)", fontsize=12)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.xaxis_date()

    br = df[df["actual_bad_rate"].notna()]
    ax2.plot(br["snapshot_date"], br["actual_bad_rate"],
             "o-", color=RED, lw=2, ms=7, label="Actual bad rate")
    ax2.set_ylabel("Actual Bad Rate (labelled subset)", fontsize=12)
    ax2.set_xlabel("Snapshot Date", fontsize=12)
    ax2.legend(fontsize=11, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(datefmt)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "7_coverage_sanity.png"),
                dpi=DPI, bbox_inches="tight", facecolor=WHT)
    plt.close(fig)
    print("Saved: deck_assets/7_coverage_sanity.png")

    # 8. Score drift
    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=WHT)
    ax.plot(df["snapshot_date"], df["mean_pred"],
            "o-", color="#2C7A7B", lw=2, ms=7)
    om = df["mean_pred"].mean()
    ax.axhline(om, linestyle=":", color=GRY, lw=1.5,
               label=f"Overall mean = {om:.4f}")
    # widen y-axis so the near-flat line isn't exaggerated by auto-scaling
    ax.set_ylim(0.45, 0.53)
    # annotate total range to make stability explicit
    pred_range = df["mean_pred"].max() - df["mean_pred"].min()
    ax.text(0.99, 0.97,
            f"Total range = {pred_range:.4f}  (negligible drift)",
            transform=ax.transAxes, fontsize=9.5, color="#2C7A7B",
            ha="right", va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=ORGL,
                      edgecolor=ORG, alpha=0.85))
    ax.set_title("Score Drift: Mean Predicted Probability Over Time",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Snapshot Date", fontsize=12)
    ax.set_ylabel("Mean Predicted Probability", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(datefmt)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "8_score_drift.png"),
                dpi=DPI, bbox_inches="tight", facecolor=WHT)
    plt.close(fig)
    print("Saved: deck_assets/8_score_drift.png")


if __name__ == "__main__":
    # ensure we run from repo root regardless of where the script lives
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    print(f"Working directory: {os.getcwd()}\n")

    draw_architecture()
    draw_medallion()
    draw_leakage_timeline()
    draw_model_selection()
    reexport_monitoring()

    print(f"\nAll assets saved to {os.path.join(repo_root, OUT)}/")
