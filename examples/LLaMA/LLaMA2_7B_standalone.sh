#!/bin/bash

DATASET_1="./"
DATASET_2="./"
DATASET_3="./"
DATASET="0.2 ${DATASET_1} 0.3 ${DATASET_2} 0.5 ${DATASET_3}"

TP_SIZE=2
PP_SIZE=2
WORLD_SIZE=8
MICRO_BATCH_SIZE=8
# The int is the number of micro steps of gradient accumulation
GLOBAL_BATCH_SIZE=$((($WORLD_SIZE * $MICRO_BATCH_SIZE) / ($TP_SIZE * $PP_SIZE) * 8))
# GLOBAL_BATCH_SIZE=128

JOB_NAME="LLaMA_tp${TP_SIZE}_pp${PP_SIZE}_mbs${MICRO_BATCH_SIZE}_gpus${WORLD_SIZE}"

LOAD_CHECKPOINT_PATH="./test"
SAVE_CHECKPOINT_PATH="./test"
TOKENIZER_PATH="~/jiashu/llama/tokenizer.model"
TENSORBOARD_DIR="./"

TRAIN_ITERS=400
EVAL_ITERS=10
EVAL_INTERVAL=100
SAVE_INTERVAL=100
LOG_INTERVAL=1

# Setting --tensorboard-queue-size to 1 significantly slows down the training
options=" \
    --finetune \
    --sequence-parallel \
        --tensor-model-parallel-size ${TP_SIZE} \
        --pipeline-model-parallel-size ${PP_SIZE} \
    --num-layers 32 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --swiglu \
        --ffn-hidden-size 11008\
        --disable-bias-linear \
        --normalization RMSNorm \
    --tokenizer-type Llama2Tokenizer \
        --make-vocab-size-divisible-by 1 \
    --init-method-std 0.01 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --lr 6.0e-5
        --lr-decay-iters 10 \
        --lr-warmup-iters 5 \
        --min-lr 6.0e-6 \
        --override-opt_param-scheduler \
        --lr-decay-style cosine \
    --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --no-gradient-accumulation-fusion \
    --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
    --save-interval ${SAVE_INTERVAL} \
        --save ${SAVE_CHECKPOINT_PATH} \
    --log-interval ${LOG_INTERVAL} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
        --tensorboard-queue-size 1000 \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
    --bf16 \
    --recompute-activations \
        --recompute-granularity selective
    "

torchrun --nproc_per_node=32 --master_port=29500 pretrain_llama.py ${options}
