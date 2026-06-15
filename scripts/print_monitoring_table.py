import glob, os, sys
import pandas as pd

os.chdir("/opt/airflow")

model = "credit_model_2024_09_01"
files = sorted(glob.glob("datamart/gold/model_monitoring/{}/{}_monitoring_*.parquet".format(model, model)))
if not files:
    print("ERROR: no monitoring parquet files found")
    sys.exit(1)

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
df = df.sort_values("snapshot_date").reset_index(drop=True)

def flag(row):
    flags = []
    if pd.isna(row["auc"]):
        flags.append("NULL_AUC")
    if pd.notna(row["psi"]) and row["psi"] > 0.25:
        flags.append("PSI>0.25")
    return " ".join(flags) if flags else "OK"

df["flag"] = df.apply(flag, axis=1)

cols = ["snapshot_date","n_scored","n_labeled","mean_pred","psi","auc","gini","actual_bad_rate","flag"]
df["snapshot_date"]    = df["snapshot_date"].dt.strftime("%Y-%m-%d")
for col in ["mean_pred", "psi", "auc", "gini", "actual_bad_rate"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["mean_pred"]        = df["mean_pred"].round(4)
df["psi"]              = df["psi"].round(4)
df["auc"]              = df["auc"].round(4)
df["gini"]             = df["gini"].round(4)
df["actual_bad_rate"]  = df["actual_bad_rate"].round(3)

print(df[cols].to_string(index=False))
print("\nTotal rows: {}".format(len(df)))
flagged = df[df["flag"] != "OK"]
if len(flagged):
    print("\nFlagged months:")
    print(flagged[cols].to_string(index=False))
else:
    print("\nNo months flagged (PSI <= 0.25 and AUC not null for all labeled months).")
