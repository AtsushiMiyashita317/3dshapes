#!/bin/bash
#PBS -N pl_ddp
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oed
#PBS -q rt_HF
#PBS -P gah51684


cd $PBS_O_WORKDIR

ckpt_predictor="./checkpoints/predictor/best-epoch=732-val_loss=0.9436.ckpt"
batch_size=10000
num_workers=8
strategy="ddp"
backend="nccl"
gpus_per_node=8
lr=0.0025

source venv/bin/activate

module load hpcx/2.20

# ---- PyTorch distributed 実行 ----
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_port=12345 \
    --nproc_per_node=${gpus_per_node} \
    clnf.py \
        ${ckpt_predictor} \
        --batch_size ${batch_size} \
        --num_workers ${num_workers} \
        --strategy ${strategy} \
        --backend ${backend} \
        --devices ${gpus_per_node} \
        --lr ${lr} \
