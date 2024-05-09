#!/bin/bash

#WARNING! starts everything in parallel

source ~/.env/bin/activate

LOGDIR=logs/

if [ $# -eq 1 ]; then
    LOGDIR="$1/"
fi
mkdir ${LOGDIR}
export LOGDIR

./grid_scripts/gridsearch_batch_size_32.sh &
./grid_scripts/gridsearch_batch_size_128.sh &
./grid_scripts/gridsearch_embed_dim_64.sh &
./grid_scripts/gridsearch_embed_dim_256.sh  &
./grid_scripts/gridsearch_learn_rate_2e-4.sh &
./grid_scripts/gridsearch_learn_rate_5e-5.sh &
./grid_scripts/gridsearch_num_heads_2.sh &
./grid_scripts/gridsearch_num_heads_4.sh &
./grid_scripts/gridsearch_num_layers_4.sh &
./grid_scripts/gridsearch_num_layers_5.sh &
./grid_scripts/gridsearch_prime.sh &
./grid_scripts/gridsearch_seq_len_10.sh &
./grid_scripts/gridsearch_seq_len_30.sh &

#camp out here until everything returns
wait