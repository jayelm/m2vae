#!/bin/bash

NUM_WORKERS="4"
MEM="16G"
EXCLUDE="jagupard4,jagupard5,jagupard6,jagupard7,jagupard8,jagupard13"

EPOCHS="1000"

BASE_CMD="python m2vae/train.py --resume --cuda --pin_memory --batch_size 128 --epochs $EPOCHS --activation relu"

for n_tracks in 1 5; do
    for annealing_epochs in 100 500; do
        exp_name="exp/small_$n_tracks""track_$annealing_epochs""anneal"
        mkdir -p "$exp_name"
        cmd="$BASE_CMD --n_tracks $n_tracks --annealing_epochs $annealing_epochs --exp_dir $exp_name"
        nlprun -a py36-muj -g 1 -c "$NUM_WORKERS" -r "$MEM" -x "$EXCLUDE" "$cmd" -o "$exp_name/stdout"
    done
done
