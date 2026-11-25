#!/bin/sh -e
# Full CV of one queue plan on the LAAS platform (pfcalcul): one array task per fold (#45).
# Run from the clone, with the env active (it is exported to the jobs):
#   source envs/lstm/lstm_jcp_env/bin/activate
#   PLAN=slurm/laas/full_cv_patch_transformer_v8.json sh slurm/laas/submit_full_cv.sh --gpu 1
# DATA and RESULTS override the dataset and results dirs below. See slurm/laas/README.md.

# The helper cds to the script's dir: run under an absolute path, since we leave the clone below.
case "$0" in
    /*) ;;
    *) exec sh "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" "$@" ;;
esac

echo "example: PLAN=slurm/laas/full_cv_patch_transformer_v8.json $0 --gpu 1"

REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${DATA:-dataset_v3.7.1}
RESULTS=${RESULTS:-results/full_cv_v3.7.1}
[ -n "$PLAN" ] || { echo "PLAN is not set"; exit 1; }
PLAN=$(cd "$(dirname "$PLAN")" && pwd)/$(basename "$PLAN")
NAME=$(basename "$PLAN" .json)
[ -f "$PLAN" ] || { echo "no plan $PLAN"; exit 1; }
[ -n "$VIRTUAL_ENV" ] || { echo "activate the env first: source envs/lstm/lstm_jcp_env/bin/activate"; exit 1; }
[ -f "$DATA/metadata.json" ] || { echo "no dataset at $DATA (set DATA)"; exit 1; }
FOLDS=$(python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))['folds']) - 1)" "$PLAN")
mkdir -p "$RESULTS/logs"

EXEC="python3 -u full_cv_fold.py $PLAN $DATA --results-root $RESULTS --fold-index \$SLURM_ARRAY_TASK_ID --device cuda"

# The helper parses --gpu and needs EXEC set before it is sourced. It builds and uses a
# venv when the submit dir holds a requirements.txt (the clone does), so submit from RESULTS:
# the job then uses the active env, exported to it.
cd "$RESULTS"
GPU_REQUIRED=true

. /pfcalcul/tools/sbatchHelpers5.sh

jobMessage=$(
######################### SBATCH launcher
#########################     #SBATCH --option        # <= this is an enabled paramater
#########################     ##SBATCH --option       # <= this is a disabled parameter
sbatch ${sbatchopt} << eof
#!/bin/bash
#SBATCH --job-name="$NAME"
#SBATCH --array=0-$FOLDS
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --output=$RESULTS/logs/${NAME}_%A_%a.out
#SBATCH --error=$RESULTS/logs/${NAME}_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=50000
#SBATCH --time=2-00:00:00

. /pfcalcul/tools/sbatchHelpers5.sh
ulimit -n 65536
envs/lstm/datasynch_perso /datasets//GRF2Kine/insole2grfm

cd $REPO
echo "python: \$(which python3), commit: \$(git rev-parse --short HEAD)"

time $EXEC

eof
)

showSubmitted
