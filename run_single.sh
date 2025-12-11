#!/bin/bash
#PBS -N pl_single
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -q rt_HG
#PBS -P gah51684


cd $PBS_O_WORKDIR

ckpt_predictor="./checkpoints/predictor/best-epoch=732-val_loss=0.9436.ckpt"
batch_size=5000
num_workers=4
lr=0.002
latent_dim=192
num_bases=192

source venv/bin/activate

python clnf.py \
    ${ckpt_predictor} \
    --batch_size ${batch_size} \
    --num_workers ${num_workers} \
    --lr ${lr} \
    --latent_dim ${latent_dim} \
    --num_bases ${num_bases}
