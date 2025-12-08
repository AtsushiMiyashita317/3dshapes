#!/bin/bash
#PBS -N pl_ddp
#PBS -l select=5
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -q rt_HF
#PBS -P gah51684


cd $PBS_O_WORKDIR

ckpt_predictor="./checkpoints/predictor/best-epoch=732-val_loss=0.9436.ckpt"
batch_size=10000
num_workers=1
strategy="ddp"
backend="nccl"
gpus_per_node=8
lr=0.01
resume="./wandb/run-20251206_112844-mlazidl7/files/best-epoch=49-val_loss=97.8120.ckpt"

source venv/bin/activate

module load hpcx/2.20

# ---- ホストリストの取得 ----
# PBS_NODEFILE に割り当てノードが列挙されている
nodes=$(sort -u $PBS_NODEFILE)
master_addr=$(head -n 1 <<< "$nodes")

# ランク数 (world size)
nnodes=$(wc -l <<< "$nodes")
world_size=$((nnodes * gpus_per_node))

echo "MASTER: $master_addr"
echo "NNODES: $nnodes"
echo "WORLD_SIZE: $world_size"

# ---- PyTorch distributed 実行 ----
mpirun --hostfile $PBS_NODEFILE -np $nnodes \
    ./dist_run.sh ${gpus_per_node} ${master_addr} \
        clnf.py \
            ${ckpt_predictor} \
            --batch_size ${batch_size} \
            --num_workers ${num_workers} \
            --strategy ${strategy} \
            --backend ${backend} \
            --num_nodes ${nnodes} \
            --devices ${gpus_per_node} \
            --lr ${lr} \
            --resume ${resume}
