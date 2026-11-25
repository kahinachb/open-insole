#!/bin/sh -e
# Full CV evaluation of the MLP and the Patch Transformer v8 (ADR 0003), inference only:
# the paper metrics with the full configuration, then each model's test scores per
# Sensor configuration (minimal and best per Patch count of the #42 ranking, a frozen copy).
# Usage, from the clone: sh slurm/laas/evaluate_full_cv.sh RESULTS DATA

R=$1
DATA=$2
OUT=$R/evaluation
PARETO=slurm/laas/pareto_mlp_sensor_dropout_w21.json

python3 -u evaluate_grfm.py --run mlp=$R/mlp_sensor_dropout_w21/all --run v8=$R/patch_transformer_v8/all \
    --dataset-dir $DATA --output-dir $OUT/full_configuration
for MODEL in mlp_sensor_dropout_w21 patch_transformer_v8; do
    python3 -u evaluate_sensor_configurations.py $R/$MODEL/all --output-dir $OUT/configurations/$MODEL \
        --pareto $PARETO --data-dir $DATA --device cuda
done
