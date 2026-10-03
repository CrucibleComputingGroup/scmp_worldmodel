#!/usr/bin/env bash
# End-to-end sensitivity probes + evaluation for evaluate/e2e_sensitivity.py.
#   MODE=probe SHARD=i NUM_SHARDS=n bash scripts/run_e2e_probe.sh   # fp + base + probes[i::n]
#   MODE=eval  TARGETS="96 48 40"   bash scripts/run_e2e_probe.sh   # e2e_<T> on KEYS (diverse_10)
# Same protocol as run_mp_vit_protocol.sh (halve, sc_prec 8, SmoothQuant, recipe skip, seeded noise).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python}
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}
export SC_OWEN_MODE=${SC_OWEN_MODE:-bitrev}
export SC_SMOOTH_SCALES=${SC_SMOOTH_SCALES:-$PWD/results/smoothquant_scales.pt}
CONFIG=${CONFIG:-configs/evaluation/bridge/frame_ada_sc_full.yaml}
FP_CONFIG=${FP_CONFIG:-configs/evaluation/bridge/frame_ada.yaml}
E2E=${E2E:-results/e2e}
STEPS=${STEPS:-50}; SCHED=${SCHED:-PNDM}
SKIP=${SKIP:-$($PY -c "import json;print(json.load(open('results/final_sc_recipe.json'))['skip'])")}
MODE=${MODE:?probe|eval}
run () { local tag=$1 cfg=$2 keys=$3 n=$4; shift 4
  echo "=== [$tag] $(date +%T)"
  env SC_HALVE=1 SC_PREC=8 "$@" $PY -u evaluate/eval_local_n_samples.py \
    --config "$cfg" --skip "$SKIP" --tag "$tag" --keys_file "$keys" --num_samples "$n" \
    --shard 0 --num_shards 1 --inference_steps "$STEPS" --scheduler "$SCHED" --save_latents; }
if [ "$MODE" = probe ]; then
  L0=$($PY -c "import json;print(json.load(open('$E2E/plan.json'))['L0'])")
  CK=$E2E/calib_keys.json; NK=$($PY -c "import json;print(len(json.load(open('$CK'))))")
  SHARD=${SHARD:-0}; NUM_SHARDS=${NUM_SHARDS:-1}
  if [ "$SHARD" = 0 ]; then
    run e2e_fp "$FP_CONFIG" "$CK" "$NK"
    run "e2e_base_$L0" "$CONFIG" "$CK" "$NK" SC_UNIFORM_STOC_LEN="$L0"
  fi
  i=0
  for p in $($PY -c "import json;print(' '.join(json.load(open('$E2E/plan.json'))['probes']))"); do
    [ $((i % NUM_SHARDS)) = "$SHARD" ] && run "e2e_$p" "$CONFIG" "$CK" "$NK" \
      SC_UNIFORM_STOC_LEN="$L0" SC_GROUP_UNIFORM_TABLE="$E2E/probes/$p.json"
    i=$((i + 1))
  done
else
  KEYS=${KEYS:-results/diverse_10.json}; N=${N:-10}
  for T in ${TARGETS:?}; do
    run "vit_e2e_$T" "$CONFIG" "$KEYS" "$N" SC_UNIFORM_STOC_LEN="$T" SC_GROUP_UNIFORM_TABLE="$E2E/e2e_$T.json"
  done
fi
echo "=== run_e2e_probe $MODE COMPLETE $(date +%T)"
