#!/usr/bin/env bash
# The whole reported workflow, in order.  Settings mirror configs/experiment.json.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
python scripts/verify_flash.py   | tee logs/verify_flash.log
python tests/run_tests.py        | tee logs/tests.log
python scripts/make_dataset.py   > logs/make_dataset.log 2>&1
for s in 0 1 2; do
  for p in 0 1; do
    python scripts/train.py --physics $p --seed $s --epochs 200 --hidden 128 128 128 >> logs/train.log 2>&1
  done
done
python scripts/learning_curve.py >> logs/learning_curve.log 2>&1
python scripts/evaluate.py        > logs/evaluate.log 2>&1
echo DONE > logs/run_all.done
