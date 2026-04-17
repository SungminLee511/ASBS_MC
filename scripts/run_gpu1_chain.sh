#!/usr/bin/env bash
# GPU 1: Batch 1 (10 runs) → Batch 2 (10 runs)
set -euo pipefail
cd /home/sky/SML/ASBS_MC

echo "=== GPU 1 CHAIN START: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="

bash scripts/run_batch1_gpu1.sh 2>&1
echo "--- GPU 1 batch 1 done, starting batch 2 ---"
bash scripts/run_batch2_gpu1.sh 2>&1

echo "=== GPU 1 CHAIN COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="
