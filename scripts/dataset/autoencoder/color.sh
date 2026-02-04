#!/bin/bash
#PBS -N pl_single
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -q rt_HG
#PBS -P gah51684


cd $PBS_O_WORKDIR

batch_size=1600
max_steps=12000
num_workers=4
removed_factors="floor_hue wall_hue object_hue"
run_name="color"

source venv/bin/activate

python predictor.py \
    --removed_factors ${removed_factors} \
    --batch_size ${batch_size} \
    --max_steps ${max_steps} \
    --num_workers ${num_workers} \
    --run_name ${run_name}

python autoencoder.py \
    --removed_factors ${removed_factors} \
    --batch_size ${batch_size} \
    --max_steps ${max_steps} \
    --num_post_layers 2 \
    --latent_dim 24 \
    --num_workers ${num_workers} \
    --run_name ${run_name}

