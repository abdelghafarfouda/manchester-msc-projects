#!/usr/bin/env bash
# The whole reported workflow, in order.  Settings mirror configs/experiment.json
# (original study) and configs/capacity.json (capacity comparison, 2026-10-04).
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

# --- October 2026 revision -------------------------------------------------
python scripts/derive_domain.py   > logs/derive_domain.log 2>&1
python scripts/guarded_demo.py    > logs/guarded_demo.log 2>&1
python scripts/analyse_errors.py  > logs/analyse_errors.log 2>&1
python scripts/capacity.py provenance > logs/capacity.log 2>&1
for s in 0 1 2; do
  for w in 64 192; do
    python scripts/train.py --physics 0 --seed $s --epochs 200 --batch 64 --lr 0.001 \
      --hidden $w $w $w --threads 2 --out-dir results/capacity --tag ffn_h${w}_s${s} >> logs/capacity.log 2>&1
  done
done
python scripts/capacity.py epoch-timing >> logs/capacity.log 2>&1
python scripts/capacity.py score        >> logs/capacity.log 2>&1
echo DONE > logs/run_all.done
