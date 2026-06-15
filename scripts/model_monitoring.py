import argparse
import os
import glob
import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# to call: python model_monitoring.py --snapshotdate "2024-09-01" --modelname "credit_model_2024_09_01.pkl"

EPSILON = 1e-6
N_BINS  = 10


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_date_str(col):
    """Coerce any snapshot_date representation to YYYY-MM-DD strings."""
    return pd.to_datetime(col).dt.strftime("%Y-%m-%d")


def _clean_id(col):
    """Cast Customer_ID to string and strip whitespace (E: join-key hygiene)."""
    return col.astype(str).str.strip()


def _load_parquet_parts(directory):
    """Concatenate all part-*.parquet files from a Spark output directory."""
    parts = sorted(glob.glob(os.path.join(directory, "part-*.parquet")))
    if not parts:
        return None
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _load_predictions(model_name_no_ext, date_str):
    date_slug = date_str.replace("-", "_")
    pred_dir = os.path.join(
        "datamart", "gold", "model_predictions",
        model_name_no_ext,
        "{}_predictions_{}.parquet".format(model_name_no_ext, date_slug),
    )
    return _load_parquet_parts(pred_dir)


def _compute_psi(ref_scores, tgt_scores):
    """
    PSI of target vs reference.

    D — Binning robustness:
      * Uses pd.qcut with duplicates='drop' to handle tied reference scores.
      * Sets outer bin edges to ±inf so target scores outside the reference
        range land in the end bins instead of being dropped or raising.
      * Returns 0.0 (identical distributions) if the reference has fewer than
        2 distinct quantile breakpoints.
    """
    ref_series = pd.Series(ref_scores)
    try:
        _, bin_edges = pd.qcut(ref_series, q=N_BINS, retbins=True, duplicates="drop")
    except ValueError:
        return 0.0                          # all reference scores are identical

    if len(bin_edges) < 3:                  # fewer than 2 usable bins
        return 0.0

    bin_edges = bin_edges.copy()
    bin_edges[0]  = -np.inf                 # catch target scores below reference min
    bin_edges[-1] =  np.inf                 # catch target scores above reference max
    n_bins = len(bin_edges) - 1

    def _proportions(scores):
        labels = pd.cut(
            pd.Series(scores), bins=bin_edges, labels=False, include_lowest=True
        )
        valid  = labels[labels.notna()].astype(int).values
        counts = np.bincount(valid, minlength=n_bins).astype(float)
        counts[counts == 0.0] += EPSILON    # epsilon on empty bins — avoid log(0)
        return counts / counts.sum()

    p_ref = _proportions(ref_scores)
    p_tgt = _proportions(tgt_scores)
    return float(np.sum((p_tgt - p_ref) * np.log(p_tgt / p_ref)))


def _null_row(snapshotdate, model_name_no_ext):
    """F — Empty-month safety: a fully-null monitoring row for a missing snapshot."""
    return pd.DataFrame([{
        "snapshot_date":   snapshotdate,
        "model_name":      model_name_no_ext,
        "n_scored":        0,
        "mean_pred":       None,
        "psi":             None,
        "n_labeled":       0,
        "actual_bad_rate": None,
        "auc":             None,
        "gini":            None,
    }])


# ── main ─────────────────────────────────────────────────────────────────────

def main(snapshotdate, modelname, referencedate):
    print("\n\n---starting job---\n\n")

    model_name_no_ext = modelname.replace(".pkl", "")
    date_slug         = snapshotdate.replace("-", "_")

    config = {
        "snapshotdate":      snapshotdate,
        "modelname":         modelname,
        "model_name_no_ext": model_name_no_ext,
        "referencedate":     referencedate,
    }
    pprint.pprint(config)

    out_dir  = os.path.join("datamart", "gold", "model_monitoring", model_name_no_ext)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        "{}_monitoring_{}.parquet".format(model_name_no_ext, date_slug),
    )

    # ── predictions for this snapshot ───────────────────────────────────────
    pred_df = _load_predictions(model_name_no_ext, snapshotdate)

    # F — write null row instead of crashing when predictions are absent/empty
    if pred_df is None or len(pred_df) == 0:
        print(
            "WARNING: no predictions found for {} — writing null row.".format(snapshotdate)
        )
        _null_row(snapshotdate, model_name_no_ext).to_parquet(out_path, index=False)
        print("Saved: {}".format(out_path))
        print("\n\n---completed job---\n\n")
        return

    # E — join-key hygiene on predictions
    pred_df["snapshot_date"] = _to_date_str(pred_df["snapshot_date"])
    pred_df["Customer_ID"]   = _clean_id(pred_df["Customer_ID"])

    n_scored  = len(pred_df)
    mean_pred = float(pred_df["model_predictions"].mean())
    print("n_scored: {}  mean_pred: {:.4f}".format(n_scored, mean_pred))

    # ── PSI vs reference ────────────────────────────────────────────────────
    psi = None
    if snapshotdate == referencedate:
        print("snapshot == reference date -> PSI = null")
    else:
        ref_df = _load_predictions(model_name_no_ext, referencedate)
        if ref_df is None or len(ref_df) == 0:
            print(
                "WARNING: reference predictions not found for {} -> PSI = null".format(
                    referencedate
                )
            )
        else:
            psi = _compute_psi(
                ref_df["model_predictions"].values,
                pred_df["model_predictions"].values,
            )
            print("PSI vs {}: {:.4f}".format(referencedate, psi))

    # ── label store ─────────────────────────────────────────────────────────
    label_parts = sorted(
        glob.glob(
            os.path.join(
                "datamart", "gold", "label_store", "*.parquet", "part-*.parquet"
            )
        )
    )
    if label_parts:
        labels_all = pd.concat(
            [pd.read_parquet(p) for p in label_parts], ignore_index=True
        )
        # E — join-key hygiene on labels
        labels_all["snapshot_date"] = _to_date_str(labels_all["snapshot_date"])
        labels_all["Customer_ID"]   = _clean_id(labels_all["Customer_ID"])
        labels_snap = labels_all[labels_all["snapshot_date"] == snapshotdate].copy()
    else:
        labels_snap = pd.DataFrame(columns=["Customer_ID", "snapshot_date", "label"])

    # ── performance metrics ─────────────────────────────────────────────────
    auc             = None
    gini            = None
    n_labeled       = 0
    actual_bad_rate = None

    if len(labels_snap) > 0:
        merged = pred_df.merge(
            labels_snap[["Customer_ID", "snapshot_date", "label"]],
            on=["Customer_ID", "snapshot_date"],
            how="inner",
        )
        n_labeled = len(merged)

        if n_labeled > 0:
            actual_bad_rate = float(merged["label"].mean())

            if merged["label"].nunique() >= 2:
                auc  = float(roc_auc_score(merged["label"], merged["model_predictions"]))
                gini = round(2.0 * auc - 1.0, 4)
                print(
                    "AUC: {:.4f}  Gini: {:.4f}  n_labeled: {}  bad_rate: {:.3f}".format(
                        auc, gini, n_labeled, actual_bad_rate
                    )
                )
            else:
                print(
                    "Only one class present for {} -> AUC/Gini = null".format(snapshotdate)
                )
    else:
        print("No labels found for {}".format(snapshotdate))   # F — safe, no crash

    print(
        "Summary  n_scored={}  n_labeled={}  mean_pred={:.4f}  "
        "psi={}  auc={}  gini={}".format(
            n_scored, n_labeled, mean_pred,
            round(psi,  4) if psi  is not None else None,
            round(auc,  4) if auc  is not None else None,
            gini,
        )
    )

    # ── save gold monitoring row ─────────────────────────────────────────────
    result_df = pd.DataFrame([{
        "snapshot_date":   snapshotdate,
        "model_name":      model_name_no_ext,
        "n_scored":        n_scored,
        "mean_pred":       mean_pred,
        "psi":             psi,
        "n_labeled":       n_labeled,
        "actual_bad_rate": actual_bad_rate,
        "auc":             auc,
        "gini":            gini,
    }])
    result_df.to_parquet(out_path, index=False)
    print("Saved monitoring row to: {}".format(out_path))

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument(
        "--snapshotdate",  type=str, required=True,
        help="YYYY-MM-DD snapshot to monitor",
    )
    parser.add_argument(
        "--modelname",     type=str, required=True,
        help="model pkl filename e.g. credit_model_2024_09_01.pkl",
    )
    parser.add_argument(
        "--referencedate", type=str, required=False, default="2023-01-01",
        help="YYYY-MM-DD baseline for PSI (default: 2023-01-01)",
    )
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname, args.referencedate)
