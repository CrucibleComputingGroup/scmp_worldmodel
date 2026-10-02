#!/usr/bin/env bash
# SC uniform vs mixed precision on IRASim under the scmp_vit FORMAL protocol
# (scmp_vit cls/run_l7_10k.sh), with the IRASim deployment choices kept:
# SmoothQuant scales + the final_sc_recipe skip set.
#
#   sc_prec 8, halve, linears per_row + chunk_d 128, qk/av per_row, all six
#   ops on SC; MP = calibrated threshold table over all six ops, metric amax,
#   objective sigma, W_g = macs, argmin pricing macs, 2% protected channels
#   (activation amax) at the ladder ceiling.
#
#   CELLS="uni_32 mp_32" N=300 bash scripts/run_mp_vit_protocol.sh
#   CELLS="fp int8 uni_128 uni_96 mp_96" LADDER=128,96,64,48,32,24,16 ...
#
# Cells: fp (FP model) | int<b> (naive symmetric WbAb, same ops + skip) |
#        uni_<L> (uniform SC, L halved cycles) | mp_<L> (calibrate + MP at budget L).
# LADDER defaults to scmp_vit's L5 (64..16, the formal mp_32 ladder); budgets
# >= 64 need L7 (128,96,64,48,32,24,16).
#   SHARD=0 NUM_SHARDS=4 CUDA_VISIBLE_DEVICES=0 CELLS=mp_32 bash scripts/run_mp_vit_protocol.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python}
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}
export SC_OWEN_MODE=${SC_OWEN_MODE:-bitrev}
export SC_SMOOTH_SCALES=${SC_SMOOTH_SCALES:-$PWD/results/smoothquant_scales.pt}
CONFIG=${CONFIG:-configs/evaluation/bridge/frame_ada_sc_full.yaml}
FP_CONFIG=${FP_CONFIG:-configs/evaluation/bridge/frame_ada.yaml}
TAG_PREFIX=${TAG_PREFIX:-vit_}
CAL_KEYS=${CAL_KEYS:-results/diverse_300.json}   # calibration pool excludes these
KEYS=${KEYS:-results/diverse_300.json}
N=${N:-300}; NCAL=${NCAL:-4}; SEED=${SEED:-1234}
STEPS=${STEPS:-50}; SCHED=${SCHED:-PNDM}
SHARD=${SHARD:-0}; NUM_SHARDS=${NUM_SHARDS:-1}
CAL=${CAL:-results/mp_calib}
CELLS=${CELLS:-"uni_32 mp_32"}
SKIP=${SKIP:-$($PY -c "import json;print(json.load(open('results/final_sc_recipe.json'))['skip'])")}
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av
LADDER=${LADDER:-64,48,32,24,16}
mkdir -p "$CAL/logs"

ev () { local tag=$1 cfg=$2; shift 2
  local extra=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do extra+=("$1"); shift; done
  [ "${1:-}" = "--" ] && shift
  echo "=== eval [$tag] n=$N shard=$SHARD/$NUM_SHARDS $(date +%T)"
  env SC_HALVE=1 SC_PREC=8 "${extra[@]}" $PY -u evaluate/eval_local_n_samples.py \
    --config "$cfg" --skip "$SKIP" --tag "$tag" --keys_file "$KEYS" \
    --num_samples "$N" --shard "$SHARD" --num_shards "$NUM_SHARDS" \
    --inference_steps "$STEPS" --scheduler "$SCHED" "$@"; }

calib () {   # tag ladder budget [extra...]
  local tag=$1 ladder=$2 budget=$3; shift 3
  [ -f "$CAL/${tag}_wrapper.json" ] && { echo "[mp] calib $tag cached"; return; }
  local top=${ladder%%,*}; local ratio; ratio=$($PY -c "print(f'{$budget/$top:.6f}')")
  echo "=== calib [$tag] ladder=$ladder budget=$budget $(date +%T)"
  $PY -u evaluate/calibrate_mp_thresholds.py --config "$CONFIG" --skip "$SKIP" \
    --inference_steps "$STEPS" --scheduler "$SCHED" \
    --mp_levels "$ladder" --budget_ratio "$ratio" --budget_ref_stoc_len "$top" \
    --sc_prec 8 --halve 1 --chunk_d 128 --operators "$ALLOPS" --attn_stoc_len "$budget" \
    --metric-select amax --objective sigma --cross-layer-weight macs --argmin-pricing macs \
    --qk_granularity per_row \
    --protect-channel-frac 0.02 --protect-channel-metric act --protect-channel-stoc-len 0 \
    --calib_seed "$SEED" --eval_keys_file "$CAL_KEYS" --n_calib_samples "$NCAL" \
    "$@" --output_json "$CAL/$tag.json" 2>&1 | tee "$CAL/logs/calib_$tag.log" \
    | grep -E "^\[calib\] (protected channels|expected)|Error|Traceback|SystemExit"
  [ -f "$CAL/${tag}_wrapper.json" ] || { echo "[mp] FAILED calib $tag"; return 1; }; }

for c in $CELLS; do
  case "$c" in
    fp)    ev "${TAG_PREFIX}fp" "$FP_CONFIG" ;;
    int*)  b=${c#int}; ev "${TAG_PREFIX}$c" "$CONFIG" -- --naive_int8 --naive_bits "$b" ;;
    uni_*) b=${c#uni_}; ev "${TAG_PREFIX}$c" "$CONFIG" SC_UNIFORM_STOC_LEN="$b" ;;
    mp_*)  b=${c#mp_}; [ "$b" -lt "${LADDER%%,*}" ] || { echo "[mp] $c: budget must be below the ladder top ${LADDER%%,*}"; continue; }
           calib "$c" "$LADDER" "$b" \
             && ev "${TAG_PREFIX}$c" "$CONFIG" SC_UNIFORM_STOC_LEN="$b" SC_MP_TABLE="$CAL/${c}_wrapper.json" ;;
    *) echo "[mp] unknown cell $c (uni_<L> | mp_<L>)" ;;
  esac
done
