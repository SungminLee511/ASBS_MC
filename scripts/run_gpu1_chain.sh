#!/usr/bin/env bash
# GPU 1: Batch 1 only (10 runs in parallel)
set -euo pipefail
cd /home/sky/SML/ASBS_MC

echo "=== GPU 1 CHAIN START: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="

bash scripts/run_batch1_gpu1.sh 2>&1

echo "=== GPU 1 BATCH 1 COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="
