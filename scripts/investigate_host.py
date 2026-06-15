import csv, re, os, sys
from collections import Counter

BASE = r"c:\Users\nlian\Desktop\SMU MITB\SMU Term 3\04 - ML Engineering CS611 - G1\Assignment 2 Repo"
os.chdir(BASE)

cols_decimal3 = ['Annual_Income','Monthly_Inhand_Salary','Outstanding_Debt',
    'Total_EMI_per_month','Amount_invested_monthly','Monthly_Balance']
cols_integer = ['Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan',
    'Delay_from_due_date','Num_of_Delayed_Payment','Num_Credit_Inquiries','Interest_Rate']
all_check = set(cols_decimal3 + cols_integer)

neg_counts = Counter()
neg_samples = {}
cred_hist_total = 0; cred_hist_match = 0
cred_hist_fail = []
pb_counts = Counter()

with open("data/features_financials.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i > 80000: break
        for c in all_check:
            v = row.get(c, "")
            if "-" in v:
                neg_counts[c] += 1
                if c not in neg_samples: neg_samples[c] = []
                if len(neg_samples[c]) < 3: neg_samples[c].append(v)
        cha = row.get("Credit_History_Age", "")
        if cha and cha.strip():
            cred_hist_total += 1
            if re.match(r"^\d+ Years and \d+ Months$", cha.strip()):
                cred_hist_match += 1
            elif len(cred_hist_fail) < 5:
                cred_hist_fail.append(cha.strip())
        pb_counts[row.get("Payment_Behaviour","(missing)")] += 1

print("=" * 65)
print("1. SIGN-STRIPPING -- negative values in raw features_financials.csv")
print("=" * 65)
for c in cols_decimal3 + cols_integer:
    n = neg_counts.get(c, 0)
    s = neg_samples.get(c, [])
    print("  {:35s}  neg={:5d}  samples={}".format(c, n, s))

print()
print("Monthly_Balance raw vs strip vs signed_extract:")
with open("data/features_financials.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    shown = 0
    for row in reader:
        v = row.get("Monthly_Balance", "")
        if "-" in v and shown < 4:
            stripped = re.sub(r"[^\d.]", "", v)
            m = re.search(r"([-+]?\d*\.?\d+)", v)
            signed = m.group(1) if m else "NOMATCH"
            print("  raw={!r:<28} strip={!r:<14} signed_extract={!r}".format(v, stripped, signed))
            shown += 1
    if shown == 0:
        print("  No negatives found in Monthly_Balance")

print()
print("=" * 65)
print("2. CREDIT_HISTORY_AGE -- parse failure rate")
print("=" * 65)
failed = cred_hist_total - cred_hist_match
pct_fail = 100.0 * failed / cred_hist_total if cred_hist_total else 0
print("  Total non-null rows: {}".format(cred_hist_total))
print("  Matched pattern:     {} ({:.1f}%)".format(cred_hist_match, 100-pct_fail))
print("  Failed -> null:      {} ({:.1f}%)".format(failed, pct_fail))
if cred_hist_fail:
    print("  Failure samples:     {}".format(cred_hist_fail))

print()
print("=" * 65)
print("3. PAYMENT_BEHAVIOUR -- ordinal encoding values")
print("=" * 65)
for pb, cnt in sorted(pb_counts.items(), key=lambda x: -x[1])[:12]:
    print("  {!r:<52}  {:6d}".format(pb, cnt))

print()
print("=" * 65)
print("4. CLICKSTREAM fe_1..fe_20 -- correlation check (sampling 10k rows)")
print("=" * 65)
clks_counts = Counter()
fe_totals = {}  # not doing full correlation without numpy -- just check zero-variance
fe_vals = {f"fe_{i}": [] for i in range(1, 21)}
with open("data/feature_clickstream.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 10000: break
        for k in fe_vals:
            v = row.get(k, "")
            try: fe_vals[k].append(int(v))
            except ValueError: pass

# Compute mean/std and correlation with fe_1 using basic statistics
def mean(lst): return sum(lst)/len(lst) if lst else 0
def std(lst):
    if len(lst) < 2: return 0
    m = mean(lst)
    return (sum((x-m)**2 for x in lst)/len(lst))**0.5

def corr(a, b):
    if len(a) != len(b) or len(a) < 2: return float("nan")
    ma, mb = mean(a), mean(b)
    sa, sb = std(a), std(b)
    if sa == 0 or sb == 0: return float("nan")
    cov = sum((a[i]-ma)*(b[i]-mb) for i in range(len(a))) / len(a)
    return cov / (sa * sb)

fe1 = fe_vals["fe_1"]
print("  Correlation with fe_1 (10k sample):")
for k in sorted(fe_vals.keys(), key=lambda x: int(x.split("_")[1])):
    if k == "fe_1": continue
    r = corr(fe1, fe_vals[k])
    flag = " <<< HIGH" if abs(r) > 0.5 else ""
    print("  {:6s}  r={:+.3f}{}".format(k, r, flag))

print()
print("  Mean / Std per feature (check for zero-variance):")
for k in sorted(fe_vals.keys(), key=lambda x: int(x.split("_")[1])):
    m = mean(fe_vals[k]); s = std(fe_vals[k])
    flag = " <<< ZERO VARIANCE" if s == 0 else ""
    print("  {:6s}  mean={:.2f}  std={:.2f}{}".format(k, m, s, flag))
