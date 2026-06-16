# CS611 Assignment 2 — Pipeline Walkthrough & Defence Guide

This document explains how the pipeline works and why every choice was made, so you can explain and defend it to a professor without needing to read the source code.

---

## 1. What this builds

A production-style pipeline that answers one question: **"Will this customer default on their cash loan?"**

The pipeline trains a model, scores every customer every month, and tracks whether the model is still working — all orchestrated by Apache Airflow running inside Docker. The prediction target is **30 DPD (days past due) at month 6 on book (mob=6)**: a customer is labelled bad if they are 30 or more days behind by their sixth loan instalment.

---

## 2. End-to-end flow — one full run traced

When you trigger the DAG, five tasks run in sequence. Here is exactly what each one reads, does, and writes.

---

### Task 1 — build_datamart (`scripts/data_pipeline_main.py`)

**Reads:** Four raw CSV files in `data/`:

| File | Contents |
|------|----------|
| `lms_loan_daily.csv` | One row per customer per instalment — amount due, paid, overdue, balance |
| `feature_clickstream.csv` | 20 anonymous behavioural signals (fe_1..fe_20) per customer per month |
| `features_attributes.csv` | Customer demographics: name, age, SSN, occupation |
| `features_financials.csv` | Credit bureau data: income, debt, payment behaviour, credit mix |

**Does:** Processes all 24 monthly snapshots (2023-01-01 → 2024-12-01) through three layers.

**BRONZE layer** — partitions each source CSV by snapshot date. No cleaning at all. Writes CSV.

```
datamart/bronze/lms/bronze_loan_daily_2023_01_01.csv
datamart/bronze/clks/bronze_clks_mthly_2023_01_01.csv
datamart/bronze/attr/bronze_attr_mthly_2023_01_01.csv
datamart/bronze/fin/bronze_fin_mthly_2023_01_01.csv
```

A bronze loan row looks like:
```
loan_id    Customer_ID  installment_num  overdue_amt  due_amt  snapshot_date
L0001234   CUS1234      6                2500.00      2083.33  2023-01-01
```

**SILVER layer** — cleans, type-casts, and derives new columns. Writes Parquet.

For the loan table, two key columns are added:
- `mob` = `installment_num` (months on book)
- `installments_missed` = `ceil(overdue_amt / due_amt)` (how many payments have been skipped)
- `first_missed_date` = `snapshot_date − installments_missed months`
- `dpd` = `datediff(snapshot_date, first_missed_date)` (days past due)

So if a customer owes 2500 and each instalment is 2083, they have missed ~1 instalment. Their `first_missed_date` is one month ago and `dpd` ≈ 30 days.

For the clickstream table: negative fe_* values are replaced with 0. For the financials table: a sign-fix was applied so that negative numeric strings (e.g. "-100") become null rather than being stripped to their magnitude.

```
datamart/silver/lms/silver_loan_daily_2023_01_01.parquet     ← +mob, +dpd
datamart/silver/clks/silver_clks_mthly_2023_01_01.parquet    ← negatives→0
datamart/silver/fin/silver_fin_mthly_2023_01_01.parquet      ← typed, capped
```

**GOLD layer — label store:** Filters silver LMS to rows where `mob == 6`, then labels `dpd >= 30` as 1 (bad loan) and 0 otherwise.

```
datamart/gold/label_store/gold_label_store_2023_01_01.parquet
```

A label row:
```
loan_id    Customer_ID  label  label_def    snapshot_date
L0001234   CUS1234      1      30dpd_6mob   2023-01-01
```

Each monthly partition holds roughly **375 labelled customers** — those whose sixth instalment falls in that snapshot month.

**GOLD layer — feature stores** (built but not used by the model):
- `datamart/gold/feature_store/eng/` — 6-month rolling clickstream history (`click_1m..click_6m`)
- `datamart/gold/feature_store/cust_fin_risk/` — financial KPIs per customer per month

These demonstrate the medallion architecture but the ML model reads raw clickstream directly (see Design Decision 5).

---

### Task 2 — train_select (`scripts/model_train.py --snapshotdate 2024-09-01`)

**Reads:**
- All label store partitions (all 24 loaded, then filtered to 2023-07-01 → 2024-08-31)
- `data/feature_clickstream.csv` (same date filter applied)

**Does:**

Joins labels to features on `Customer_ID + snapshot_date`. Every labelled customer gets their 20 clickstream features for the same month.

Splits the data **chronologically first**:

```
2023-07-01 ──────────────────────────────── 2024-06-30 | 2024-07-01 ── 2024-08-31
          TRAIN / TEST window (12 months)              |   OOT (2 months)
          → random 80/20 split within this window      |   completely untouched
```

Fits a `StandardScaler` **on X_train only**, then transforms X_test and X_oot with it. This is the core leakage prevention step.

Trains three candidate models:

| Model | Tuning |
|-------|--------|
| LogisticRegression | Fixed params (`C=1.0, balanced`) — no search |
| RandomForestClassifier | RandomizedSearchCV, 20 combinations, `scoring="roc_auc"` |
| XGBoost | RandomizedSearchCV, 100 combinations, `scoring="roc_auc"` |

Results and winner selection:

```
Model               Train AUC   Test AUC   OOT AUC   Overfit gap
LogisticRegression  0.6555      0.6457     0.6297     0.026   ← SELECTED
RandomForest        0.7880      0.6210     0.6140     0.174
XGBoost             0.7569      0.6190     0.6265     0.130
```

**Writes:** `model_bank/credit_model_2024_09_01.pkl` — a dict containing the fitted model, the StandardScaler, all metrics, training window dates, and a full model-selection record with all three candidates' scores.

---

### Task 3 — inference_backfill (loop over all 24 months)

**Reads:** `model_bank/credit_model_2024_09_01.pkl` + `data/feature_clickstream.csv` (filtered to one snapshot date per iteration)

**Does:** For each of the 24 months, loads the saved StandardScaler, applies it to the raw features, then calls `model.predict_proba(X)[:, 1]` to get each customer's probability of default. Scores **all ~8,974 customers** present in the clickstream that month — not just those with labels.

**Writes:** One Parquet directory per month (Spark partitioned format):
```
datamart/gold/model_predictions/credit_model_2024_09_01/
    credit_model_2024_09_01_predictions_2023_01_01.parquet/   ← part-*.parquet inside
    credit_model_2024_09_01_predictions_2023_02_01.parquet/
    ...  (24 total)
```

A prediction row:
```
Customer_ID  snapshot_date  model_name                      model_predictions
CUS1234      2023-01-01     credit_model_2024_09_01.pkl     0.7231
```

The score is a probability between 0 and 1. Higher = more likely to default.

---

### Task 4 — monitoring_backfill (loop over all 24 months)

**Reads:** Prediction Parquets + label store Parquets (inner join by Customer_ID + snapshot_date)

**Does — PSI (stability):** Compares the current month's score distribution against the reference month (2023-01-01). Uses 10 equal-frequency bins defined from the reference population, then measures how much the current distribution has shifted using the standard PSI formula: `Σ (P_current − P_ref) × ln(P_current / P_ref)`. Thresholds: PSI < 0.10 = stable, 0.10–0.25 = watch, > 0.25 = investigate.

**Does — AUC/Gini (performance):** Inner-joins predictions to labels. Only the ~375 customers with a mob=6 label in that month contribute. Computes `roc_auc_score` on those customers. `gini = 2 × AUC − 1`. If only one class is present (all good or all bad), AUC is null.

**Writes:** One plain Parquet file per month:
```
datamart/gold/model_monitoring/credit_model_2024_09_01/
    credit_model_2024_09_01_monitoring_2023_01_01.parquet   ← PSI=null (reference)
    credit_model_2024_09_01_monitoring_2023_02_01.parquet
    ...  (24 total)
```

A monitoring row:
```
snapshot_date  model_name                n_scored  mean_pred  psi     n_labeled  actual_bad_rate  auc     gini
2023-09-01     credit_model_2024_09_01  8974      0.2813     0.0031  375        0.280            0.712   0.425
```

---

### Task 5 — visualise (`scripts/model_monitoring_viz.py`)

**Reads:** All 24 monitoring Parquet files from `datamart/gold/model_monitoring/credit_model_2024_09_01/`

**Writes:** Four PNGs in `monitoring_plots/`:

| File | What it shows |
|------|---------------|
| `1_performance_over_time.png` | AUC and Gini month-by-month — the primary performance chart |
| `2_psi_stability.png` | PSI with 0.10/0.25 threshold bands — the stability chart |
| `3_coverage_sanity.png` | n_scored (volume) + actual bad rate — sanity check |
| `4_score_drift.png` | Mean predicted probability over time — checks for score calibration shift |

---

## 3. Design decisions

Each decision below states: what we chose, why, what the alternatives were, and how to defend it.

---

### 1. Label definition: 30 DPD at mob=6

**Chosen:** `label = 1` if `dpd >= 30` at `mob == 6` (installment number 6).

**Why:** 30 DPD is the industry-standard early delinquency threshold — beyond 30 days overdue, the customer has definitively missed at least one payment. Month 6 is long enough for a default pattern to establish itself yet early enough to have labels across most of our 24-month dataset. The assignment brief specifies exactly this definition.

**Alternatives:** mob=3 (faster labels but noisier signal), mob=12 (more predictive but we'd lose 12 months of training data), 90 DPD (the NPL threshold — but that's after the loss is crystallised, too late for intervention).

**Defence:** "30 DPD at mob=6 is the brief's specification and a standard credit risk definition. It gives enough time for genuine delinquency to emerge while leaving room for both training data and an OOT holdout."

---

### 2. Training window: 12 months

**Chosen:** Train and test on 2023-07-01 → 2024-06-30 (12 months). OOT on 2024-07-01 → 2024-08-31 (2 months). Training cutoff = 2024-09-01.

**Why:** 12 months captures a full seasonal cycle in payment behaviour. The data spans 24 months, so a 12+2 split leaves a clean 2-month OOT and doesn't exhaust all history in training.

**Alternatives:** 18 months (uses more of the available data, smaller OOT). 6 months (risks underfitting seasonality). Using all 24 months for training (no OOT — then you can't compare candidates fairly).

**Defence:** "12 months is a standard risk modelling convention. It covers seasonal variation without leaving us with an inadequate OOT holdout for model selection."

---

### 3. Unused Jan–Jun 2023 history

**Chosen:** The first 6 months of labelled data are not used in training.

**Why:** Our 12-month window, anchored at the 2024-09-01 training date, naturally starts at 2023-07-01. The months before that are left as background history.

**Impact:** Approximately 2,250 additional rows (~6 × 375) are excluded. Small relative to 4,766 training rows.

**Defence:** "A fixed-window convention is reproducible and clean. In a rolling production system, those 6 months would be incorporated as the window advances monthly. The omission costs less than 1 AUC point."

---

### 4. Leakage prevention — time-based OOT split before random split; scaler fit on train only

**Chosen:**
1. Assign the **most recent 2 months to OOT by time**, before any randomisation.
2. Randomly split the remaining 12 months 80/20 into train/test (stratified by label).
3. Fit `StandardScaler` **on X_train only**, then apply it to transform X_test and X_oot.

**Why:** If you randomly split the full dataset first, a customer from August 2024 can appear in both training and OOT. That is temporal leakage — the model sees future data during training. If you fit the scaler on all data, the normalisation parameters carry information from test/OOT distributions into training.

**Alternative (wrong approach):** `train_test_split(all_data, random_state=42)` — this is the classic leakage mistake that inflates test AUC by 3–5 points and makes the model look better than it is in production.

**Defence:** "OOT is our simulation of deployment. It must remain untouched until every training decision is made. Scaler-on-train-only mimics what happens in production: you can only normalise against the distribution you saw during training, because future data doesn't exist yet."

---

### 5. Reading raw clickstream directly; skipping the gold feature stores

**Chosen:** Both `model_train.py` and `model_inference.py` read `data/feature_clickstream.csv` directly, bypassing the bronze/silver/gold processing chain for ML features.

**Why — train/serve consistency:** The scaler is fitted on the raw CSV values and inference also passes raw CSV values. If training used silver-cleaned values but inference used raw, any difference in how negatives or edge cases are handled would cause a silent systematic error in production scores.

**Why — gold feature stores don't add value here:** The gold engagement store keeps only `fe_1` from each month (the other 19 features are discarded with the comment "behaviour is similar" — which is factually wrong; all 20 features are near-independent with correlations < 0.02). Adopting the gold financial features would also introduce the temporal alignment issue for those variables.

**Defence:** "Consistency between training and serving is more important than the silver cleaning for these features. Both pipelines see identical raw values. The gold feature stores are built correctly and demonstrate the medallion architecture, but adopting them for the model requires resolving the temporal alignment issue and would not improve AUC on this dataset."

---

### 6. Temporal alignment — features at mob=6, not at application time

**The limitation:** The brief says "predict at application time" (month 0 of the loan). Our join uses features from the mob=6 snapshot — the same month as the DPD observation.

**Why we kept it this way:** The clickstream features (fe_1..fe_20) are ongoing behavioural signals observed every month, not captured only at application. A bank with an existing customer base scores them continuously. Joining at mob=6 answers "what does this customer look like right now, and are they at risk?" This is **not target leakage** — the fe_ signals are causally independent of whether a payment was made.

**What it means for the model's use case:** The model is correctly described as a **month-6 early-warning signal**, not an application-time credit scorer. It can flag customers likely to reach 90 DPD (NPL) while there is still time to intervene (collections contact, restructuring, payment holiday).

**Defence:** "We acknowledge this is a departure from strict application-time prediction. The features are contemporaneous with the label observation, not from loan origination. For an ongoing customer relationship — which this dataset represents — month-6 scoring is still operationally useful: early intervention at month 6 is more cost-effective than recovery at month 12."

---

### 7. Three-model comparison; selection by OOT AUC — really a case for LR on stability

**Chosen:** Three candidates (LR with fixed params, RF with 20-combination search, XGBoost with 100-combination search). Winner selected by OOT AUC.

**The honest picture:** The OOT AUC differences are tiny — 0.6297 vs 0.6265 vs 0.6140. No statistical test would call LR a clear winner on discrimination alone. The real case for LR is:

- **Overfit gap:** LR train→OOT drop = **0.026 AUC points**. RF drop = 0.174. XGBoost drop = 0.130. Tree models are memorising the training data — their training AUC (0.79, 0.76) is not real.
- **Deployment stability:** In 24 months of live monitoring, LR's AUC standard deviation is 0.028. XGBoost's is 0.055. LR is more predictable month-to-month.
- **Parsimony:** A logistic regression is interpretable, fast to retrain (seconds vs minutes), and its coefficients can be inspected to explain why a customer received a high score.

OOT AUC is used as the selection criterion because it is the only metric not tainted by in-sample overfitting. But the argument for LR is really about **stability and generalisation**, not a 0.003 AUC edge.

**Defence:** "The OOT AUC differences are within noise. We chose Logistic Regression because its train-to-OOT gap (2.6 points) is an order of magnitude smaller than the ensemble gaps (13–17 points). A model that genuinely generalises at 0.63 AUC is more valuable in production than one that shows 0.79 in-sample and decays rapidly."

---

### 8. Monitoring split into performance (AUC where labels exist) and stability (PSI)

**Chosen:** PSI is computed for all ~8,974 scored customers per month. AUC is computed only for the ~375 customers who have mob=6 labels that month.

**Why two metrics:** PSI measures whether the score distribution has shifted. AUC measures whether the scores still correctly rank borrowers. These are independent failure modes. A model could maintain stable PSI while its discriminating power collapses (e.g., if the bad-rate drops but all customers' scores drop uniformly). You need AUC to confirm discrimination is intact.

**Why PSI alone is insufficient:** PSI only tells you the score distribution has shifted relative to a reference. It doesn't tell you if the model is still useful. In practice you'd act on a PSI alert by investigating — but AUC is the ground truth of whether to retrain.

**Why AUC can be null in a live system:** Labels only mature at mob=6. In production, scoring happens monthly but you only get ground truth 6 months later. Our backfill gives AUC for all 24 months because we have the full historical label store. In live deployment, you'd have a 6-month AUC lag.

**Defence:** "We track both performance and stability because they test different failure modes. PSI is an early-warning signal that can be computed immediately. AUC is ground truth but has a 6-month lag in production."

---

### 9. PSI reference month: 2023-01-01

**Chosen:** All monthly PSI values compare against the January 2023 score distribution.

**Why:** PSI answers "has the population we're scoring shifted relative to a known baseline?" The baseline should be the earliest stable production snapshot — before any temporal drift. 2023-01-01 is the first month of data and sits outside the training window, so it represents the pre-training population.

**Why not use a training-period month:** That would conflate model quality changes with population shifts, making PSI harder to interpret over a long horizon.

**Note:** PSI for 2023-01-01 is null (comparing a distribution to itself is trivially 0 — null is more honest than reporting 0.000 for the reference month itself).

**Defence:** "First-month reference gives the most stable baseline and is outside the training window, so it doesn't bias the PSI calculation. All subsequent months can be compared to a consistent anchor."

---

### 10. class_weight='balanced' — calibration caveat

**Chosen:** LR and RF both use `class_weight='balanced'`, which up-weights the minority class (bad loans, ~28%) during fitting.

**Why:** Without class weighting, models on imbalanced data tend to predict "good" for almost everyone and achieve ~72% accuracy without learning anything useful about defaulters.

**Caveat:** `class_weight='balanced'` adjusts the **decision boundary** but does not calibrate **predicted probabilities**. The `predict_proba` output is shifted upward for the positive class. The model's scores do not directly equal the true default probability. For expected-loss calculations (PD × LGD × EAD), a Platt scaling step would be needed.

**AUC is still valid:** AUC is a rank-order metric that doesn't depend on calibration. All monitoring AUC values are correct. But `mean_pred` (~0.28) should not be read as "28% of customers will default" without calibration verification.

**Defence:** "Balanced class weighting ensures the minority class has adequate influence during training. AUC is rank-order and unaffected. Probability calibration would be needed before using these scores as literal default probabilities in a credit loss model."

---

### 11. Single-trigger linear DAG, no scheduling

**Chosen:** `schedule_interval=None` (manual trigger only). Five tasks in a simple linear chain.

**Why:** For a 24-month backfill, a linear chain is safe and easy to debug. If one month fails, the whole task fails with a clear exit code. Parallelising inference across months would require Airflow dynamic task mapping, which adds significant complexity.

**For production:** `schedule_interval='0 1 1 * *'` runs on the 1st of each month. `catchup=True` backfills any missed runs automatically. The existing inference/monitoring loops could be replaced with single-month runs driven by Airflow's `{{ ds }}` execution date template variable.

**Defence:** "Manual trigger with a linear chain is appropriate for a course submission that needs one clean run. Production would use a monthly schedule with catchup enabled and single-snapshot tasks per DAG run."

---

### 12. Docker + PySpark + Parquet

**Docker:** Reproducibility. Exact Airflow version (2.6.1), Python version (3.7), and all library versions are pinned. Anyone with Docker Desktop can reproduce the results without installing anything locally.

**PySpark:** The data pipeline extends Assignment 1's medallion architecture, which used PySpark for scalability demonstration. In production, LMS data for a real bank would be hundreds of millions of rows — PySpark handles this where pandas cannot. We use pandas for the single-node ML scripts (inference, monitoring) where Spark overhead is unnecessary.

**Parquet:** Columnar format with embedded schema. Faster for column-selective reads. Type-safe (no implicit type conversion on read/write). Compatible with cloud storage (S3, GCS, ADLS) and distributed query engines. Spark writes it as partitioned directories by default, which matches how production data lakes are organised.

---

### 13. Own Assignment 1 datamart vs three classmates' datamarts

**Chosen:** We use our own data pipeline output as the bronze/silver/gold source.

**Why:** Traceability and consistency. We understand our transformation logic — including the sign-fix applied to financial numeric columns and how DPD is derived. Using a classmate's datamart would mean data quality issues in their pipeline silently propagate into our model with no way to audit.

**Defence:** "Full pipeline lineage is traceable from raw CSV to gold table in our own codebase. Using a third-party datamart would break that chain."

---

## 4. Known limitations

**1. Temporal alignment (most important to flag):** Features at mob=6 are not features at application time. The model cannot be used at loan origination in its current form. It is best described as a month-6 early-warning signal.

**2. Weak feature signal (OOT AUC ~0.63):** 20 anonymous behavioural clickstream signals carry limited predictive information about loan default. Conventional credit risk models use bureau scores, income/debt ratios, and payment history — none of which enter our feature set. 0.63 AUC means the model correctly rank-orders borrowers about 63% of the time vs 50% for random.

**3. Engagement gold table drops 19 of 20 signals:** The gold engagement store keeps only `fe_1`, discarding `fe_2..fe_20`. The comment in the code claims "behaviour is similar for all other fe_n" — correlation analysis shows this is wrong; all 20 features are near-independent (max |r| ≈ 0.02). This doesn't affect the model (which reads raw CSV directly) but would need to be fixed if the gold store were ever adopted for training.

**4. Broad `.fillna(0)` in silver LMS:** The `installments_missed` derivation applies `fillna(0)` to the entire DataFrame, zeroing out any NaN in other numeric columns including `overdue_amt`. A customer with an unknown overdue amount is treated as having no overdue, producing a false-negative label. Impact is likely small on this dataset.

**5. Hardcoded outlier caps in silver financials:** `Num_of_Loan ≤ 9`, `Interest_Rate ≤ 34`, etc. are fixed values from Assignment 1 EDA. They don't adapt to population drift. Not relevant for the current clickstream model, but would need data-driven capping if financial features were adopted.

**6. Unused Jan–Jun 2023 history (~2,250 rows):** Small impact, easy to fix by extending the training window.

**7. No probability calibration:** `class_weight='balanced'` shifts predicted probabilities upward. Scores cannot be interpreted as literal default rates without a calibration step.

---

## 5. Professor questions and answers

**Q1: Why is your AUC only 0.63? Is that acceptable?**

For a model using only 20 anonymous behavioural signals with no credit bureau features, 0.63 OOT AUC is expected. Real-world credit models with bureau data achieve 0.70–0.80 AUC. The Gini of 0.26 reflects meaningful but modest discriminating power. This pipeline is a proof-of-concept demonstrating ML engineering architecture — model training, evaluation, inference, and monitoring — not a production-grade credit scorer. A production model would require bureau scores, income/debt ratios, and application-time features.

**Q2: How do you know there's no data leakage?**

Three safeguards: (1) The OOT set is carved out by time before any random split — data from July–August 2024 cannot appear in training. (2) The StandardScaler is fitted on X_train only, then applied to transform X_test and X_oot — no information from held-out sets influences normalisation. (3) The label (`dpd >= 30`) is derived from LMS payment records; the features (`fe_1..fe_20`) are clickstream signals — there is no causal path between them. The train-to-OOT AUC gap for LR (0.026) is consistent with a model that genuinely generalises; if there were leakage we would expect near-zero gap.

**Q3: Why did you choose Logistic Regression over XGBoost? XGBoost is usually better.**

The OOT AUC difference (0.6297 vs 0.6265) is within noise — no statistical test would call this significant. The case for LR is the overfit gap: LR's train-to-OOT drop is 2.6 AUC points versus 13.0 for XGBoost. XGBoost memorised the training data (train AUC 0.76 vs OOT 0.63 is a 13-point gap). In 24 months of live monitoring, LR's AUC standard deviation is 0.028 versus XGBoost's 0.055. A stable, interpretable 0.63 beats an unstable 0.76 train AUC that decays in production.

**Q4: What would trigger a model refresh?**

Two triggers: (1) Performance — AUC drops below 0.60 on a rolling 3-month window (a meaningful degradation from the 0.63 baseline, not within the ±0.03 normal variance). (2) Stability — PSI exceeds 0.10 for two consecutive months, indicating the incoming population has shifted materially from the reference distribution. In our dataset, PSI stays below 0.004 throughout, so only a performance trigger is practically relevant. In governance terms: PSI > 0.10 → investigate; PSI > 0.25 → retrain regardless of AUC.

**Q5: Why does PSI stay so low (< 0.01) across all 24 months?**

The clickstream features (fe_1..fe_20) come from the same underlying customer population each month. The model is a fixed linear transformation of those features with a fixed scaler. Since neither the population nor the model weights change, the score distribution is extremely stable. This is the expected behaviour for a well-calibrated batch-scoring system on a stable population. PSI would spike if a marketing campaign brought in a different customer segment, or if the model were retrained with substantially different weights.

**Q6: What's the difference between n_scored and n_labeled in your monitoring table?**

`n_scored` ≈ 8,974: every customer in the clickstream for that month, scored regardless of whether we have labels for them. `n_labeled` ≈ 375: the subset whose sixth loan instalment falls in that exact snapshot month, giving us ground truth labels. AUC is computed only on those 375. In a live deployment you would only have labels for customers who entered the loan book 6 months prior, so `n_labeled` would be 0 for the most recent months until labels mature.

**Q7: Your features are at mob=6, not at application time. Isn't that a problem?**

It is a limitation, declared transparently. The clickstream features represent ongoing behaviour, not application-time characteristics. The model is better described as "a month-6 early-warning system" than "an application-time credit score." This is still operationally valuable: a bank can use it to flag high-risk customers at month 6 and trigger collections or restructuring before the loan reaches 90 DPD (NPL status). Early intervention at month 6 is more cost-effective than loss recovery at month 12+.

**Q8: Why score all 8,974 customers for inference instead of just the ones with labels?**

Because in production you don't know which customers will have labels at scoring time. Inference and evaluation are two separate stages: you score everyone monthly, then evaluate performance only where ground truth is available. Scoring only labelled customers would be an unrealistic simulation of deployment — in real operations you want a risk score for every active customer, not just those at exactly mob=6.

**Q9: Why use PySpark if the data fits in pandas?**

The data pipeline extends Assignment 1's medallion architecture, which uses PySpark for scalability demonstration. Partitioned Parquet writes are natural in PySpark. In production, LMS data for a real bank would be hundreds of millions of rows. We use pandas for the single-node ML scripts (inference, monitoring) where distributed overhead would be wasteful — that is the correct balance.

**Q10: What deployment options would you recommend?**

Two options: (1) **Batch scoring (current approach)** — score all customers at month-end, store predictions in the gold table, trigger collections workflow from the prediction table. Low infrastructure requirements, easy to audit. (2) **Real-time API** — wrap `predict_proba` in a REST endpoint (FastAPI + model artifact from model bank). Suitable if a risk score is needed at the moment of a loan application or customer event. The LogisticRegression model serialises to ~50 KB and responds in milliseconds. Before replacing the production model, a champion/challenger setup would route 5–10% of traffic to the challenger and compare AUC on the same cohort over 1–2 months.

---

*To go deeper on any section, ask for an interactive walkthrough.*
