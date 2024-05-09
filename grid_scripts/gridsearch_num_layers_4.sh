#!/bin/bash

export PYTHONUNBUFFERED="true"
export CUDA_VISIBLE_DEVICES=0

# generate results

#target values
# num layers [3, 4, 5]
# num heads [1, 3, 5]
# sequence len [10, 20, 30]
# batch size [32, 64, 128]
# embedding dim [64, 128, 256]
# learning rate [2e-4, 1e-4, 5e-5]

#default values
# SEQ_LEN="20"
# NUM_LAYERS="3"
# NUM_HEADS="1"
# BATCH_SIZE="64"
# EMBED_DIM="128"
# LEARN_RATE="1e-4"

#this run values
SEQ_LEN="20"
NUM_LAYERS="4"
NUM_HEADS="1"
BATCH_SIZE="64"
EMBED_DIM="128"
LEARN_RATE="1e-4"


for GYM_ENV in "hopper" "halfcheetah" "walker2d"; do 
    for DATASET in "expert"; do  # "medium" "medium-replay" 
        echo python experiment.py \
            --env ${GYM_ENV} \
            --dataset ${DATASET} \
            --K ${SEQ_LEN} \
            --n_layer ${NUM_LAYERS} \
            --n_head ${NUM_HEADS} \
            --batch_size ${BATCH_SIZE} \
            --embed_dim ${EMBED_DIM} \
            --learning_rate ${LEARN_RATE} > \
        ${LOGDIR}${GYM_ENV}-${DATASET}-${SEQ_LEN}-${NUM_LAYERS}-${NUM_HEADS}-${BATCH_SIZE}-${EMBED_DIM}-${LEARN_RATE}.log
        python experiment.py \
            --env ${GYM_ENV} \
            --dataset ${DATASET} \
            --K ${SEQ_LEN} \
            --n_layer ${NUM_LAYERS} \
            --n_head ${NUM_HEADS} \
            --batch_size ${BATCH_SIZE} \
            --embed_dim ${EMBED_DIM} \
            --learning_rate ${LEARN_RATE} \
        | tee -a "${LOGDIR}${GYM_ENV}-${DATASET}-${SEQ_LEN}-${NUM_LAYERS}-${NUM_HEADS}-${BATCH_SIZE}-${EMBED_DIM}-${LEARN_RATE}.log"
    done
done
