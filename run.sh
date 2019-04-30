#!/bin/bash

NUM_WORKERS="4"
MEM="16G"
EXCLUDE="jagupard4,jagupard5,jagupard6,jagupard7,jagupard8,jagupard13"

BASE_CMD="python m2vae/train.py --cuda --pin_memory --batch_size 128 --epochs 500 --activation relu"

for n_tracks in 1 5; do
    for annealing_epochs in 50 100; do
        exp_name="exp/$n_tracks""track_$annealing_epochs""anneal"
        mkdir -p "$exp_name"
        cmd="$BASE_CMD --n_tracks $n_tracks --annealing_epochs $annealing_epochs --exp_dir $exp_name"
        nlprun -a py36-muj -g 1 -c "$NUM_WORKERS" -r "$MEM" -x "$EXCLUDE" "$cmd" -o "$exp_name/stdout"
    done
done
