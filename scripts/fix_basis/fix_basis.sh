#!/bin/bash
#SBATCH --job-name=fix_basis
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

export PYENV_ROOT="/home/miyashita21/mrnas04home/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
export WANDB_API_KEY="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"
eval "$(pyenv init -)"
source venv/bin/activate

ckpt_predictor="./checkpoints/predictor/best-epoch=732-val_loss=0.9436.ckpt"
ckpt_autoencoder="./checkpoints/autoencoder/best-epoch=599-val_loss=0.0020.ckpt"
batch_size=4000
num_workers=4
lr=0.002

num_bases_sym=4
eps_p_sym=0.001
eps_q_sym=0.1
hom_layers=24

run_name="fix_basis"

python clnf.py \
    ${ckpt_predictor} \
    ${ckpt_autoencoder} \
    --run_name ${run_name} \
    --batch_size ${batch_size} \
    --num_workers ${num_workers} \
    --lr ${lr} \
    --num_bases_sym ${num_bases_sym} \
    --eps_p_sym ${eps_p_sym} \
    --eps_q_sym ${eps_q_sym} \
    --hom_layers ${hom_layers} \
    --fix_w_sym_to_commutative_rotation_basis
