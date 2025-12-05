#!/bin/bash

nproc_per_node=$1
master_addr=$2

nvidia-smi

WANDB_API_KEY=709b10653760c2e1064058dd3a3bef3251589e95

echo OMPI_COMM_WORLD_RANK: $OMPI_COMM_WORLD_RANK
echo OMPI_COMM_WORLD_SIZE: $OMPI_COMM_WORLD_SIZE
torchrun \
    --nnodes=${OMPI_COMM_WORLD_SIZE} \
    --node_rank=${OMPI_COMM_WORLD_RANK} \
    --master_port=12345 \
    --nproc_per_node=${nproc_per_node} \
    --master_addr=${master_addr} \
    ${@:3}
