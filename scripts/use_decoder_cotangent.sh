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
ckpt_autoencoder="./checkpoints/autoencoder/best-epoch=599-val_loss=0.0020.ckpt"
batch_size=16000
num_workers=4
lr=0.004

num_bases_sym=276
num_bases_null=276
eps_p_sym=0.001
eps_q_sym=0.1
eps_p_null=0.001
eps_q_null=0.1

run_name="use_decoder_cotangent"

source venv/bin/activate

python clnf.py \
    ${ckpt_predictor} \
    ${ckpt_autoencoder} \
    --run_name ${run_name} \
    --batch_size ${batch_size} \
    --num_workers ${num_workers} \
    --lr ${lr} \
    --num_bases_null ${num_bases_null} \
    --eps_p_null ${eps_p_null} \
    --eps_q_null ${eps_q_null} \
