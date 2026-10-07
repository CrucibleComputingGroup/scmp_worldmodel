#!/bin/bash
# Per-group precision traces (evaluate/group_trace.py, scmp-group-trace-v1) for the IRASim
# Table V SC cells (vit-protocol alignment, results/mp_calib_vit tables): uniform
# 128/96/64/48/40/32, per-row MP 96/48/40, per-(row, chunk) 96/48/40, mp_48_esc. One video
# (the first of diverse_10, seeded as the evals), every GTRACE_STRIDE-th DiT forward,
# GTRACE_WINDOWS of them, exact SC path, all cells on ONE GPU. Per cell: capture, then the
# check (identity vs the capture's own scmp-trace-summary, context vs the archived n=10
# trace when there is one). Cells whose check.json is ok are skipped. Port of scmp_llm
# benchmark/ppl/kbands/run_group_trace_all_20260929.sbatch.
#   sbatch scripts/group_trace_all.sbatch       (CELLS="uni_48 pair_48" to subset)
#SBATCH --job-name=wm_gtrace_all
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/coe-nbleier/zhkangqi/sc_traces/wm_group_trace/logs/gtrace_all_%j.out
set -uo pipefail
R=/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_worldmodel
P=/nfs/turbo/coe-nbleier/zhkangqi/irasim_data
cd $R || exit 1
PY=$HOME/miniconda3/envs/qdit/bin/python
export PYTHONPATH=$R/kernels:$R:$P/pydeps:$P/stubs
export BRIDGE_ROOT=$R/robotdata/opensource_robotdata/bridge
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64 SC_SMOOTH_SCALES=$R/results/smoothquant_scales.pt
unset SC_MP_TRACE
export GTRACE_WINDOWS=${GTRACE_WINDOWS:-12} GTRACE_STRIDE=${GTRACE_STRIDE:-5}
ROOT=${ROOT:-/nfs/turbo/coe-nbleier/zhkangqi/sc_traces/wm_group_trace}
ARCH=$R/results/sc_traces_vit
C=$R/results/mp_calib_vit
SKIP=$($PY -c "import json;print(json.load(open('results/final_sc_recipe.json'))['skip'])")
CELLS=${CELLS:-"uni_128 uni_96 uni_64 uni_48 uni_40 uni_32 mp_96 mp_48 mp_40 pair_96 pair_48 pair_40 mp_48_esc"}
for cell in $CELLS; do
  OUTD=$ROOT/$cell
  if $PY -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') else 1)" \
      "$OUTD/check.json" 2>/dev/null; then echo "[gtrace-all] $cell already done, skipped"; continue; fi
  case "$cell" in
    uni_*) ENVS=(SC_UNIFORM_STOC_LEN=${cell#uni_}) ;;
    mp_48_esc) ENVS=(SC_UNIFORM_STOC_LEN=48 SC_MP_TABLE=$C/mp_48_esc_wrapper.json) ;;
    mp_*|pair_*) ENVS=(SC_UNIFORM_STOC_LEN=${cell#*_} SC_MP_TABLE=$C/${cell}_wrapper.json) ;;
    *) echo "[gtrace-all] unknown cell $cell"; continue ;;
  esac
  mkdir -p "$OUTD"; rm -rf "$OUTD/eval"
  ARCHF=$ARCH/vit_${cell}_trace.json; AARG=(); [ -f "$ARCHF" ] && AARG=(--archived-trace "$ARCHF")
  echo "[gtrace-all] $cell: ${ENVS[*]}"
  t0=$SECONDS
  if env SC_HALVE=1 SC_PREC=8 "${ENVS[@]}" GTRACE_CELL=$cell GTRACE_OUT="$OUTD/exact" \
       GTRACE_CHECK_TRACE="$OUTD/exact/sc_trace_summary.json" EVAL_OUT_ROOT="$OUTD/eval" \
       $PY -u evaluate/group_trace.py --config configs/evaluation/bridge/frame_ada_sc_full.yaml \
         --skip "$SKIP" --tag gtrace_$cell --keys_file results/diverse_10.json --num_samples 1 \
         --inference_steps 50 --scheduler PNDM > "$OUTD/capture.log" 2>&1 \
     && $PY -u evaluate/group_trace_check.py --exact "$OUTD/exact" \
          --sc-trace "$OUTD/exact/sc_trace_summary.json" "${AARG[@]}" \
          --out "$OUTD/check.json" > "$OUTD/check.log" 2>&1; then
    echo "[gtrace-all] $cell OK in $((SECONDS - t0)) s: $(grep -E '^\[check\] (identity|tile)' "$OUTD/check.log" | tr '\n' ' ')"
  else
    echo "[gtrace-all] $cell FAILED after $((SECONDS - t0)) s"; tail -3 "$OUTD/capture.log" "$OUTD/check.log" 2>/dev/null
  fi
done
echo "[gtrace-all] done"
