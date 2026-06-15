from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

# pipeline constants 

# All 24 monthly snapshots the pipeline covers (2023-01-01 → 2024-12-01)
_DATES = (
    "2023-01-01 2023-02-01 2023-03-01 2023-04-01 2023-05-01 2023-06-01 "
    "2023-07-01 2023-08-01 2023-09-01 2023-10-01 2023-11-01 2023-12-01 "
    "2024-01-01 2024-02-01 2024-03-01 2024-04-01 2024-05-01 2024-06-01 "
    "2024-07-01 2024-08-01 2024-09-01 2024-10-01 2024-11-01 2024-12-01"
)

_MODEL      = "credit_model_2024_09_01.pkl"
_TRAIN_DATE = "2024-09-01"

# bash command templates 
# $D is intentionally NOT a Python placeholder — it is a bash variable and
# must survive .format() substitution unchanged.  Only {dates} and {model}
# are replaced by Python.

_INFERENCE_CMD = """\
set -e
cd /opt/airflow
export PYTHONPATH=/opt/airflow
for D in {dates}; do
    echo "=== inference $D ==="
    python scripts/model_inference.py --snapshotdate "$D" --modelname "{model}"
done
""".format(dates=_DATES, model=_MODEL)

_MONITORING_CMD = """\
set -e
cd /opt/airflow
export PYTHONPATH=/opt/airflow
for D in {dates}; do
    echo "=== monitoring $D ==="
    python scripts/model_monitoring.py --snapshotdate "$D" --modelname "{model}"
done
""".format(dates=_DATES, model=_MODEL)

# DAG definition

with DAG(
    dag_id="loan_default_ml_pipeline",
    description=(
        "End-to-end loan default ML pipeline: "
        "datamart build, model training + selection, "
        "inference backfill, monitoring backfill, visualisation."
    ),
    schedule_interval=None,    # manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["cs611", "ml"],
    default_args={"owner": "airflow", "retries": 0},
) as dag:

    # Task 1: bronze → silver → gold datamart (all 24 months)
    # Also regenerates sign-fixed silver_fin + gold_cust_fin_risk partitions.
    build_datamart = BashOperator(
        task_id="build_datamart",
        bash_command=(
            "cd /opt/airflow && "
            "PYTHONPATH=/opt/airflow "
            "python scripts/data_pipeline_main.py"
        ),
    )

    # Task 2: train 3 candidates, select best by OOT AUC, write model bank
    train_select = BashOperator(
        task_id="train_select",
        bash_command=(
            "cd /opt/airflow && "
            "PYTHONPATH=/opt/airflow "
            "python scripts/model_train.py --snapshotdate {td}"
        ).format(td=_TRAIN_DATE),
    )

    # Task 3: score all 24 monthly snapshots → gold predictions table 
    inference_backfill = BashOperator(
        task_id="inference_backfill",
        bash_command=_INFERENCE_CMD,
    )

    # Task 4: AUC/Gini + PSI for all 24 months → gold monitoring table 
    monitoring_backfill = BashOperator(
        task_id="monitoring_backfill",
        bash_command=_MONITORING_CMD,
    )

    # Task 5: 4 monitoring PNGs → monitoring_plots/ 
    visualise = BashOperator(
        task_id="visualise",
        bash_command=(
            "cd /opt/airflow && "
            "PYTHONPATH=/opt/airflow "
            "python scripts/model_monitoring_viz.py --modelname {model}"
        ).format(model=_MODEL),
    )

    #  Linear chain 
    build_datamart >> train_select >> inference_backfill >> monitoring_backfill >> visualise
