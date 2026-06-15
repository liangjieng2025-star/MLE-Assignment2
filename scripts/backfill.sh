#!/bin/bash
set -e
MODELNAME="credit_model_2024_09_01.pkl"
DATES="2023-01-01 2023-02-01 2023-03-01 2023-04-01 2023-05-01 2023-06-01
       2023-07-01 2023-08-01 2023-09-01 2023-10-01 2023-11-01 2023-12-01
       2024-01-01 2024-02-01 2024-03-01 2024-04-01 2024-05-01 2024-06-01
       2024-07-01 2024-08-01 2024-09-01 2024-10-01 2024-11-01 2024-12-01"

echo "=== INFERENCE BACKFILL ==="
for D in $DATES; do
    echo "--- inference $D ---"
    python scripts/model_inference.py --snapshotdate "$D" --modelname "$MODELNAME" 2>/dev/null | tail -3
done

echo "=== MONITORING BACKFILL ==="
for D in $DATES; do
    echo "--- monitoring $D ---"
    python scripts/model_monitoring.py --snapshotdate "$D" --modelname "$MODELNAME" 2>/dev/null \
        | grep -E "Summary|Saved|WARNING|AUC|PSI"
done

echo "=== ALL DONE ==="
