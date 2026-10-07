#!/usr/bin/env bash
# Diagnostics for "row-level MP loses to uniform at matched budget" (IRASim):
#   rand_<L>  same mp_<L> table, row ranking randomly permuted per call
#             (SC_MP_SHUFFLE=1): same rung fractions, no ranking information.
#   blk_<L>   no row dispatch: each (op, block) uniform at its mp_<L> bucket
#             avg_stoc_len (SC_GROUP_UNIFORM_TABLE): cross-layer allocation only.
# Same eval protocol / keys / seed / skip / SmoothQuant as run_mp_vit_protocol.sh.
#   CELLS="rand_48 blk_48" CAL=results/mp_calib_vit KEYS=results/diverse_10.json N=10 bash scripts/run_mp_diag.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python}
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}
export SC_OWEN_MODE=${SC_OWEN_MODE:-bitrev}
export SC_SMOOTH_SCALES=${SC_SMOOTH_SCALES:-$PWD/results/smoothquant_scales.pt}
CONFIG=${CONFIG:-configs/evaluation/bridge/frame_ada_sc_full.yaml}
TAG_PREFIX=${TAG_PREFIX:-vit_}
KEYS=${KEYS:-results/diverse_300.json}; N=${N:-300}
STEPS=${STEPS:-50}; SCHED=${SCHED:-PNDM}
CAL=${CAL:-results/mp_calib}
SKIP=${SKIP:-$($PY -c "import json;print(json.load(open('results/final_sc_recipe.json'))['skip'])")}
ev () { local tag=$1; shift
  echo "=== eval [$tag] n=$N $(date +%T)"
  env SC_HALVE=1 SC_PREC=8 "$@" $PY -u evaluate/eval_local_n_samples.py \
    --config "$CONFIG" --skip "$SKIP" --tag "$tag" --keys_file "$KEYS" \
    --num_samples "$N" --shard 0 --num_shards 1 --inference_steps "$STEPS" --scheduler "$SCHED"; }
for c in ${CELLS:?}; do
  case "$c" in   # PR #4 PaYN block-Gamma allocation (one level per module) in the aligned pipeline
    g758) ev "${TAG_PREFIX}payn_96" SC_UNIFORM_STOC_LEN=96 SC_GROUP_UNIFORM_TABLE=results/mp_fractions_sc_g758.json; continue ;;
    g658) ev "${TAG_PREFIX}payn_48" SC_UNIFORM_STOC_LEN=48 SC_GROUP_UNIFORM_TABLE=results/mp_fractions_sc_g658.json; continue ;;
    g632) ev "${TAG_PREFIX}payn_40" SC_UNIFORM_STOC_LEN=40 SC_GROUP_UNIFORM_TABLE=results/mp_fractions_sc_g632.json; continue ;;
  esac
  b=${c#*_}
  [ -f "$CAL/mp_$b.json" ] || { echo "[diag] $c: $CAL/mp_$b.json missing"; continue; }
  case "$c" in
    rand_*) ev "${TAG_PREFIX}$c" SC_UNIFORM_STOC_LEN="$b" SC_MP_TABLE="$CAL/mp_${b}_wrapper.json" SC_MP_SHUFFLE=1 ;;
    blk_*)  ev "${TAG_PREFIX}$c" SC_UNIFORM_STOC_LEN="$b" SC_GROUP_UNIFORM_TABLE="$CAL/mp_$b.json" ;;
    *) echo "[diag] unknown cell $c" ;;
  esac
done
