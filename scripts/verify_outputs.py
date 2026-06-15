import os, glob, pickle
os.chdir("/opt/airflow")

import pandas as pd

SEP = "=" * 72

# ── (1) PKL artifact ──────────────────────────────────────────────────────────
print(SEP)
print("1. MODEL ARTIFACT — model_bank/credit_model_2024_09_01.pkl")
print(SEP)

pkl_path = "model_bank/credit_model_2024_09_01.pkl"
assert os.path.exists(pkl_path), "PKL NOT FOUND"
size_kb = os.path.getsize(pkl_path) / 1024

with open(pkl_path, "rb") as f:
    art = pickle.load(f)

print("File size : {:.1f} KB".format(size_kb))
print("Keys      :", list(art.keys()))
print("model_version :", art.get("model_version"))
print("model type    :", type(art["model"]).__name__)

sel = art.get("model_selection", {})
print("\nmodel_selection.selected_name    :", sel.get("selected_name"))
print("model_selection.selection_criterion:", sel.get("selection_criterion"))
print("\nCandidate comparison:")
hdr = "{:<22}  {:>9}  {:>10}  {:>9}  {:>10}  {:>8}  {:>9}".format(
    "Name", "Tr AUC", "Tr Gini", "Te AUC", "Te Gini", "OOT AUC", "OOT Gini")
print(hdr)
print("-" * len(hdr))
for c in sel.get("candidates", []):
    sel_flag = " <-- WINNER" if c["name"] == sel.get("selected_name") else ""
    print("{:<22}  {:>9}  {:>10}  {:>9}  {:>10}  {:>8}  {:>9}{}".format(
        c["name"],
        c["auc_train"], c["gini_train"],
        c["auc_test"],  c["gini_test"],
        c["auc_oot"],   c["gini_oot"],
        sel_flag,
    ))

print("\nresults (winner):", art.get("results"))

# ── (3) Monitoring table ──────────────────────────────────────────────────────
print()
print(SEP)
print("3. MONITORING TABLE — datamart/gold/model_monitoring/credit_model_2024_09_01/")
print(SEP)

mon_dir = "datamart/gold/model_monitoring/credit_model_2024_09_01"
files = sorted(glob.glob(os.path.join(mon_dir, "*.parquet")))
print("Monitoring files found: {}".format(len(files)))

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
df = df.sort_values("snapshot_date").reset_index(drop=True)

for col in ["mean_pred", "psi", "auc", "gini", "actual_bad_rate"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["snapshot_date"]   = df["snapshot_date"].dt.strftime("%Y-%m-%d")
df["mean_pred"]       = df["mean_pred"].round(4)
df["psi"]             = df["psi"].round(4)
df["auc"]             = df["auc"].round(4)
df["gini"]            = df["gini"].round(4)
df["actual_bad_rate"] = df["actual_bad_rate"].round(3)

def flag(row):
    flags = []
    if pd.isna(row["auc"]):
        flags.append("NULL_AUC")
    if pd.notna(row["psi"]) and row["psi"] > 0.25:
        flags.append("PSI>0.25")
    return " ".join(flags) if flags else "OK"

df["flag"] = df.apply(flag, axis=1)

cols = ["snapshot_date", "n_scored", "n_labeled", "mean_pred",
        "psi", "auc", "gini", "actual_bad_rate", "flag"]
print(df[cols].to_string(index=False))
print("\nTotal rows: {}".format(len(df)))

labeled = df[df["flag"] == "OK"]
if len(labeled):
    auc_vals = pd.to_numeric(labeled["auc"], errors="coerce").dropna()
    if len(auc_vals):
        print("AUC range (labeled months): min={:.4f}  max={:.4f}  mean={:.4f}  std={:.4f}".format(
            auc_vals.min(), auc_vals.max(), auc_vals.mean(), auc_vals.std()))

flagged = df[df["flag"] != "OK"]
if len(flagged):
    print("\nFlagged months:")
    print(flagged[cols].to_string(index=False))
else:
    print("\nNo months flagged (PSI <= 0.25, AUC present for all labeled months).")
