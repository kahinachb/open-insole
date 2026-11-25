#!/bin/sh -e
# Full CV evaluation of the MLP and Transformer v8 on the LAAS platform (pfcalcul): one GPU job
# running slurm/laas/evaluate_full_cv.sh once the 30 folds of submit_full_cv.sh are trained (#45).
# Run from the clone, with the env active (it is exported to the job):
#   source envs/lstm/lstm_jcp_env/bin/activate
#   sh slurm/laas/submit_evaluation.sh --gpu 1
# DATA and RESULTS override the dataset and results dirs below. See slurm/laas/README.md.

# The helper cds to the script's dir: run under an absolute path, since we leave the clone below.
case "$0" in
    /*) ;;
    *) exec sh "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" "$@" ;;
esac

echo "example: $0 --gpu 1"

REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${DATA:-dataset_v3.7.1}
RESULTS=${RESULTS:-results/full_cv_v3.7.1}
NAME=full_cv_evaluation
[ -n "$VIRTUAL_ENV" ] || { echo "activate the env first: source envs/lstm/lstm_jcp_env/bin/activate"; exit 1; }
[ -f "$DATA/metadata.json" ] || { echo "no dataset at $DATA (set DATA)"; exit 1; }
TRAINED=$(ls "$RESULTS"/*/all/fold_*/metadata.json 2>/dev/null | wc -l)
[ "$TRAINED" -eq 30 ] || { echo "$TRAINED folds trained in $RESULTS, 30 expected"; exit 1; }
mkdir -p "$RESULTS/logs"

EXEC="sh slurm/laas/evaluate_full_cv.sh $RESULTS $DATA"

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
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --output=$RESULTS/logs/${NAME}_%j.out
#SBATCH --error=$RESULTS/logs/${NAME}_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=50000
#SBATCH --time=12:00:00

. /pfcalcul/tools/sbatchHelpers5.sh
ulimit -n 65536
envs/lstm/datasynch_perso /datasets//GRF2Kine/insole2grfm

cd $REPO
echo "python: \$(which python3), commit: \$(git rev-parse --short HEAD)"

time $EXEC

eof
)

showSubmitted
