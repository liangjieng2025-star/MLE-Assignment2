"""
Data pipeline correctness investigation.
Run from /opt/airflow inside the container.
"""
import os, glob
os.chdir("/opt/airflow")

import pandas as pd
import numpy as np

SEP = "\n" + "="*70 + "\n"

# ──────────────────────────────────────────────────────────────────────────────
# 1. SIGN-STRIPPING: check features_financials.csv for negative values
# ──────────────────────────────────────────────────────────────────────────────
print(SEP + "1. SIGN-STRIPPING — features_financials.csv negative check")

fin = pd.read_csv("data/features_financials.csv", nrows=50000)
print("Shape:", fin.shape)

cols_decimal3 = [
    'Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt',
    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance'
]
cols_integer = [
    'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan',
    'Delay_from_due_date', 'Num_of_Delayed_Payment',
    'Num_Credit_Inquiries', 'Interest_Rate'
]

def extract_signed(s):
    """Simulate regexp_extract with signed float pattern on a string value."""
    import re
    if pd.isna(s):
        return float('nan')
    m = re.search(r'([-+]?\d*\.?\d+)', str(s))
    return float(m.group(1)) if m else float('nan')

def strip_sign(s):
    """Simulate current [^\\d.] regex — strips the minus sign."""
    import re
    if pd.isna(s):
        return float('nan')
    stripped = re.sub(r'[^\d.]', '', str(s))
    try:
        return float(stripped)
    except ValueError:
        return float('nan')

print("\nDecimal3 columns — sample raw values and negative count:")
for c in cols_decimal3:
    if c not in fin.columns:
        print("  {} — NOT IN CSV".format(c))
        continue
    raw_neg = fin[c].astype(str).apply(lambda s: '-' in str(s)).sum()
    raw_sample_neg = fin[fin[c].astype(str).apply(lambda s: '-' in str(s))][c].head(5).tolist()
    current_val = fin[c].astype(str).apply(strip_sign)
    current_neg_count = (current_val < 0).sum()   # should be 0 after strip
    print("  {:35s} raw_neg_rows={:5d}  sample={}".format(c, raw_neg, raw_sample_neg[:3]))

print("\nInteger columns — negative count:")
for c in cols_integer:
    if c not in fin.columns:
        print("  {} — NOT IN CSV".format(c))
        continue
    raw_neg = fin[c].astype(str).apply(lambda s: '-' in str(s)).sum()
    raw_sample_neg = fin[fin[c].astype(str).apply(lambda s: '-' in str(s))][c].head(5).tolist()
    print("  {:35s} raw_neg_rows={:5d}  sample={}".format(c, raw_neg, raw_sample_neg[:3]))

# Concrete before/after example
print("\nConcrete before→after example for Monthly_Balance:")
if 'Monthly_Balance' in fin.columns:
    neg_rows = fin[fin['Monthly_Balance'].astype(str).apply(lambda s: '-' in str(s))]
    if len(neg_rows):
        for _, row in neg_rows.head(3).iterrows():
            raw = str(row['Monthly_Balance'])
            stripped = strip_sign(raw)
            signed = extract_signed(raw)
            print("  raw='{}'  →  strip=[{:.3f}]  signed_extract=[{:.3f}]".format(raw, stripped, signed))
    else:
        print("  No negative values in Monthly_Balance.")


# ──────────────────────────────────────────────────────────────────────────────
# 2. lit(None) BUG — check what gold engagement files actually exist
# ──────────────────────────────────────────────────────────────────────────────
print(SEP + "2. lit(None) BUG — gold engagement file inventory")

eng_dir = "datamart/gold/feature_store/eng/"
eng_files = sorted(glob.glob(os.path.join(eng_dir, "*.parquet")))
print("Files found: {}".format(len(eng_files)))
for f in eng_files:
    print("  {}".format(os.path.basename(f)))

# Also check: how many prior-month silver clks files exist for early snapshots?
print("\nSilver clks files available (first 8):")
silver_clks = sorted(glob.glob("datamart/silver/clks/*.parquet"))
for f in silver_clks[:8]:
    print("  {}".format(os.path.basename(f)))
print("  ... ({} total)".format(len(silver_clks)))

# Does 2023-02-01 engagement file exist and have click_2m etc.?
feb_eng = os.path.join(eng_dir, "gold_ft_store_engagement_2023_02_01.parquet")
if os.path.exists(feb_eng):
    try:
        df = pd.read_parquet(feb_eng)
        print("\n2023-02-01 engagement sample:")
        print(df.head(3).to_string())
    except Exception as e:
        # It's a Spark directory
        parts = glob.glob(os.path.join(feb_eng, "part-*.parquet"))
        if parts:
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
            print("\n2023-02-01 engagement sample (from dir):")
            print(df.head(3).to_string())
        else:
            print("Cannot read 2023-02-01 engagement: {}".format(e))
else:
    print("\n2023-02-01 engagement file/dir: NOT FOUND at {}".format(feb_eng))
    # Check if directory form exists
    feb_eng_dir = feb_eng  # same path if .parquet is a dir
    all_eng = glob.glob(os.path.join(eng_dir, "*"))
    print("All entries in eng dir:")
    for e in sorted(all_eng)[:10]:
        print("  {}".format(os.path.basename(e)))


# ──────────────────────────────────────────────────────────────────────────────
# 3. CLICKSTREAM CORRELATION: fe_1..fe_20 correlation matrix summary
# ──────────────────────────────────────────────────────────────────────────────
print(SEP + "3. CLICKSTREAM CORRELATION — fe_1..fe_20 (one month sample)")

clks = pd.read_csv("data/feature_clickstream.csv", nrows=15000)
fe_cols = [c for c in clks.columns if c.startswith("fe_")]
print("Feature columns found: {}".format(fe_cols))

if fe_cols:
    corr = clks[fe_cols].corr()
    np.fill_diagonal(corr.values, np.nan)   # hide self-correlations

    # Summarise: max abs correlation with fe_1
    with_fe1 = corr["fe_1"].dropna().abs().sort_values(ascending=False)
    print("\nAbsolute correlation with fe_1:")
    print(with_fe1.to_string())

    # Global high-correlation pairs
    corr_vals = corr.abs()
    pairs = []
    for i, r in enumerate(fe_cols):
        for j, c in enumerate(fe_cols):
            if i < j and not np.isnan(corr_vals.loc[r, c]):
                pairs.append((corr_vals.loc[r, c], r, c))
    pairs.sort(reverse=True)
    print("\nTop-10 highest-correlation pairs (abs):")
    for v, r, c in pairs[:10]:
        print("  {:<6} {:<6}  r={:.3f}".format(r, c, v))
    print("\nPairs with |r| > 0.50:")
    high = [(v, r, c) for v, r, c in pairs if v > 0.50]
    if high:
        for v, r, c in high:
            print("  {:<6} {:<6}  r={:.3f}".format(r, c, v))
    else:
        print("  None")


# ──────────────────────────────────────────────────────────────────────────────
# 4. CREDIT_HISTORY_AGE parse-failure rate
# ──────────────────────────────────────────────────────────────────────────────
print(SEP + "4. CREDIT_HISTORY_AGE — parse-failure rate")

import re
if 'Credit_History_Age' in fin.columns:
    non_null = fin['Credit_History_Age'].dropna()
    total = len(non_null)
    matched = non_null.str.match(r"^\d+ Years and \d+ Months$").sum()
    failed = total - matched
    print("Total non-null:  {}".format(total))
    print("Matched pattern: {} ({:.1f}%)".format(matched, 100*matched/total if total else 0))
    print("Failed (→ null): {} ({:.1f}%)".format(failed, 100*failed/total if total else 0))
    if failed > 0:
        fail_samples = non_null[~non_null.str.match(r"^\d+ Years and \d+ Months$")].head(5).tolist()
        print("Failure samples: {}".format(fail_samples))
else:
    print("Credit_History_Age not found in columns: {}".format(fin.columns.tolist()))


# ──────────────────────────────────────────────────────────────────────────────
# 5. TEMPORAL ALIGNMENT: label_store snapshot_date = application month or mob-6?
# ──────────────────────────────────────────────────────────────────────────────
print(SEP + "5. TEMPORAL ALIGNMENT — label store vs loan origination")

label_parts = sorted(glob.glob("datamart/gold/label_store/*.parquet/part-*.parquet"))
if label_parts:
    labels = pd.concat([pd.read_parquet(p) for p in label_parts], ignore_index=True)
    labels['snapshot_date'] = pd.to_datetime(labels['snapshot_date'])
    print("Label store shape: {}".format(labels.shape))
    print("Label snapshot_dates (unique): {}".format(sorted(labels['snapshot_date'].dt.strftime('%Y-%m-%d').unique())))

    # Load a slice of the raw loan data to check loan_start_date relative to snapshot_date at mob=6
    bronze_loan = sorted(glob.glob("datamart/bronze/lms/bronze_loan_daily_2023_07_01.csv"))
    if bronze_loan:
        loan_sample = pd.read_csv(bronze_loan[0])
        loan_sample['loan_start_date'] = pd.to_datetime(loan_sample['loan_start_date'])
        loan_sample['snapshot_date'] = pd.to_datetime(loan_sample['snapshot_date'])
        mob6 = loan_sample[loan_sample['installment_num'] == 6]
        if len(mob6):
            sample = mob6.head(3)[['Customer_ID','loan_start_date','snapshot_date','installment_num']].copy()
            sample['months_diff'] = ((sample['snapshot_date'].dt.year - sample['loan_start_date'].dt.year)*12
                                     + (sample['snapshot_date'].dt.month - sample['loan_start_date'].dt.month))
            print("\nSample mob=6 rows from 2023-07-01 bronze loan:")
            print(sample.to_string(index=False))
            print("\nConclusion: snapshot_date = {} months after loan origination".format(
                sample['months_diff'].mean()))
else:
    print("No label store parts found.")


# ──────────────────────────────────────────────────────────────────────────────
# 6. PAYMENT_BEHAVIOUR encoding check
# ──────────────────────────────────────────────────────────────────────────────
print(SEP + "6. PAYMENT_BEHAVIOUR — category distribution")
if 'Payment_Behaviour' in fin.columns:
    print(fin['Payment_Behaviour'].value_counts(dropna=False).head(10).to_string())

print("\n\n=== INVESTIGATION COMPLETE ===\n")
