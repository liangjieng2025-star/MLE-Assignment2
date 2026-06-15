import argparse
import os
import glob
import pandas as pd
import pickle
import numpy as np
import pprint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

# to call this script:
# python scripts/model_train.py --snapshotdate "2024-09-01"


# helper functions

def _evaluate_candidate(model, X_tr, y_tr, X_te, y_te, X_ot, y_ot):
    """Return dict with train/test/OOT AUC and Gini for a fitted model."""
    def _ag(X, y):
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        return round(auc, 4), round(2.0 * auc - 1.0, 4)
    tr_a, tr_g = _ag(X_tr, y_tr)
    te_a, te_g = _ag(X_te, y_te)
    ot_a, ot_g = _ag(X_ot, y_ot)
    return {
        "auc_train":  tr_a, "gini_train": tr_g,
        "auc_test":   te_a, "gini_test":  te_g,
        "auc_oot":    ot_a, "gini_oot":   ot_g,
    }


# main code
def main(snapshotdate, modelname):
    print('\n\nstarting job\n\n')
    os.chdir("/opt/airflow")

    # initialise SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")


    # confiq set up
    train_test_period_months = 12
    oot_period_months = 2
    train_test_ratio = 0.8

    config = {}
    config["model_train_date_str"] = snapshotdate
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] = oot_period_months
    config["model_train_date"] = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["oot_end_date"] = config["model_train_date"] - timedelta(days=1)
    config["oot_start_date"] = config["model_train_date"] - relativedelta(months=oot_period_months)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=train_test_period_months)
    config["train_test_ratio"] = train_test_ratio
    config["model_bank_directory"] = "model_bank/"

    if modelname:
        config["model_version"] = modelname
    else:
        config["model_version"] = "credit_model_" + snapshotdate.replace("-", "_")

    pprint.pprint(config)


    # load label store
    folder_path = "datamart/gold/label_store/"
    files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, "*"))]
    label_store_sdf = spark.read.option("header", "true").parquet(*files_list)
    print("label_store row_count:", label_store_sdf.count())

    labels_sdf = label_store_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"]) &
        (col("snapshot_date") <= config["oot_end_date"])
    )
    print("extracted labels_sdf", labels_sdf.count(),
          config["train_test_start_date"], config["oot_end_date"])


    # load feature store
    feature_location = "data/feature_clickstream.csv"
    features_store_sdf = spark.read.csv(feature_location, header=True, inferSchema=True)
    print("feature_store row_count:", features_store_sdf.count())

    features_sdf = features_store_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"]) &
        (col("snapshot_date") <= config["oot_end_date"])
    )
    print("extracted features_sdf", features_sdf.count(),
          config["train_test_start_date"], config["oot_end_date"])


    # joining labels to features
    data_pdf = labels_sdf.join(features_sdf, on=["Customer_ID", "snapshot_date"], how="left").toPandas()
    feature_cols = [c for c in data_pdf.columns if c.startswith("fe_")]


    # time-based split with OOT for most recent window
    oot_pdf = data_pdf[
        (data_pdf["snapshot_date"] >= config["oot_start_date"].date()) &
        (data_pdf["snapshot_date"] <= config["oot_end_date"].date())
    ]
    train_test_pdf = data_pdf[
        (data_pdf["snapshot_date"] >= config["train_test_start_date"].date()) &
        (data_pdf["snapshot_date"] <= config["train_test_end_date"].date())
    ]

    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols],
        train_test_pdf["label"],
        test_size=1 - config["train_test_ratio"],
        random_state=88,
        shuffle=True,
        stratify=train_test_pdf["label"],
    )

    print("X_train", X_train.shape[0])
    print("X_test",  X_test.shape[0])
    print("X_oot",   X_oot.shape[0])
    print("y_train bad rate", round(float(y_train.mean()), 2))
    print("y_test  bad rate", round(float(y_test.mean()),  2))
    print("y_oot   bad rate", round(float(y_oot.mean()),   2))


    # preprocess with fit scaler on X_train only, and transform all splits
    scaler = StandardScaler()
    transformer_stdscaler = scaler.fit(X_train)

    X_train_sc = transformer_stdscaler.transform(X_train)
    X_test_sc  = transformer_stdscaler.transform(X_test)
    X_oot_sc   = transformer_stdscaler.transform(X_oot)

    # Logistic Regression model
    print("\nLogisticRegression ")
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=88,
    )
    lr_model.fit(X_train_sc, y_train)
    lr_metrics = _evaluate_candidate(lr_model, X_train_sc, y_train, X_test_sc, y_test, X_oot_sc, y_oot)
    lr_hp = {k: v for k, v in lr_model.get_params().items()}
    print("  Train AUC {auc_train}  Test AUC {auc_test}  OOT AUC {auc_oot}".format(**lr_metrics))


    # Random Forest
    rf_param_dist = {
        "n_estimators":     [50, 100, 200],
        "max_depth":        [3, 5, 7, None],
        "min_samples_leaf": [1, 5, 10],
        "max_features":     ["sqrt", "log2"],
    }
    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=88, class_weight="balanced"),
        param_distributions=rf_param_dist,
        scoring="roc_auc",
        n_iter=20,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    rf_search.fit(X_train_sc, y_train)
    rf_model = rf_search.best_estimator_
    rf_metrics = _evaluate_candidate(rf_model, X_train_sc, y_train, X_test_sc, y_test, X_oot_sc, y_oot)
    rf_hp = rf_search.best_params_
    print("  Best params: {}".format(rf_hp))
    print("  Train AUC {auc_train}  Test AUC {auc_test}  OOT AUC {auc_oot}".format(**rf_metrics))


    # XGBoost
    print("\nXGBoost with RandomizedSearchCV ")
    xgb_clf = xgb.XGBClassifier(eval_metric="logloss", random_state=88)
    xgb_param_dist = {
        "n_estimators":     [25, 50],
        "max_depth":        [2, 3],
        "learning_rate":    [0.01, 0.1],
        "subsample":        [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "gamma":            [0, 0.1],
        "min_child_weight": [1, 3, 5],
        "reg_alpha":        [0, 0.1, 1],
        "reg_lambda":       [1, 1.5, 2],
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=xgb_param_dist,
        scoring="roc_auc",
        n_iter=100,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    xgb_search.fit(X_train_sc, y_train)
    xgb_model = xgb_search.best_estimator_
    xgb_metrics = _evaluate_candidate(xgb_model, X_train_sc, y_train, X_test_sc, y_test, X_oot_sc, y_oot)
    xgb_hp = xgb_search.best_params_
    print("  Best params: {}".format(xgb_hp))
    print("  Train AUC {auc_train}  Test AUC {auc_test}  OOT AUC {auc_oot}".format(**xgb_metrics))


    # Model Selection based on OOT AUC, with comparison table
    candidates = [
        {"name": "LogisticRegression", "model": lr_model, "hp": lr_hp, "metrics": lr_metrics},
        {"name": "RandomForest",        "model": rf_model, "hp": rf_hp, "metrics": rf_metrics},
        {"name": "XGBoost",             "model": xgb_model,"hp": xgb_hp,"metrics": xgb_metrics},
    ]

    winner = max(candidates, key=lambda c: c["metrics"]["auc_oot"])

    # Comparison table
    header = "{:<22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>9}  {:>9}".format(
        "Model", "Train AUC", "Train Gini", "Test AUC", "Test Gini", "OOT AUC", "OOT Gini"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("Model Selection — evaluation on train / test / OOT splits")
    print(sep)
    print(header)
    print(sep)
    for c in candidates:
        m = c["metrics"]
        sel = " <-- SELECTED" if c["name"] == winner["name"] else ""
        print("{:<22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>9}  {:>9}{}".format(
            c["name"],
            m["auc_train"], m["gini_train"],
            m["auc_test"],  m["gini_test"],
            m["auc_oot"],   m["gini_oot"],
            sel,
        ))
    print(sep)
    print("Winner: {}  (OOT AUC = {})".format(winner["name"], winner["metrics"]["auc_oot"]))
    print(sep + "\n")


    # Build model artefact with metadata, and save to model bank
    best_model  = winner["model"]
    best_metrics = winner["metrics"]

    model_selection_record = []
    for c in candidates:
        rec = {"name": c["name"]}
        rec.update(c["metrics"])
        model_selection_record.append(rec)

    model_artefact = {}
    model_artefact["model"]                      = best_model
    model_artefact["model_version"]              = config["model_version"]
    model_artefact["preprocessing_transformers"] = {"stdscaler": transformer_stdscaler}
    model_artefact["data_dates"]                 = config
    model_artefact["data_stats"] = {
        "X_train": X_train.shape[0],
        "X_test":  X_test.shape[0],
        "X_oot":   X_oot.shape[0],
        "y_train": round(float(y_train.mean()), 2),
        "y_test":  round(float(y_test.mean()),  2),
        "y_oot":   round(float(y_oot.mean()),   2),
    }
    model_artefact["results"] = {
        "auc_train":  best_metrics["auc_train"],
        "auc_test":   best_metrics["auc_test"],
        "auc_oot":    best_metrics["auc_oot"],
        "gini_train": best_metrics["gini_train"],
        "gini_test":  best_metrics["gini_test"],
        "gini_oot":   best_metrics["gini_oot"],
    }
    model_artefact["hp_params"] = winner["hp"]
    model_artefact["model_selection"] = {
        "candidates":           model_selection_record,
        "selected_name":        winner["name"],
        "selection_criterion":  "OOT AUC (highest)",
    }

    pprint.pprint({k: v for k, v in model_artefact.items() if k != "model"})


    # save artefact to model bank with version name
    if not os.path.exists(config["model_bank_directory"]):
        os.makedirs(config["model_bank_directory"])

    file_path = os.path.join(config["model_bank_directory"], config["model_version"] + ".pkl")
    with open(file_path, "wb") as f:
        pickle.dump(model_artefact, f)
    print("Model saved to {}".format(file_path))


    # end
    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True,
                        help="YYYY-MM-DD — sets the training cutoff; window dates are derived from this")
    parser.add_argument("--modelname",    type=str, required=False, default=None,
                        help="model version name to save (default: credit_model_<snapshotdate>)")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
