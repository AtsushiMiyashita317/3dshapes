#!/bin/bash
#PBS -N pl_ddp
#PBS -l select=4:ncpus=192:ngpus=8:mpiprocs=8
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oed
#PBS -o job.log
#PBS -q rt_HF
#PBS -P gah51684


cd $PBS_O_WORKDIR

CKPT_PREDICTOR="./checkpoints/predictor/best-epoch=732-val_loss=0.9436.ckpt"
BATCH_SIZE=32000
NUM_WORKERS=32
STRATEGY="ddp"
NUM_NODES=4
DEVICES_PER_NODE=8

source venv/bin/activate

cat $SGE_JOB_HOSTLIST > ./hostfile
HOST=${HOSTNAME:0:5}

module load openmpi
module load nccl/2.5/2.5.6-1

# ---- ホストリストの取得 ----
# PBS_NODEFILE に割り当てノードが列挙されている
nodes=$(sort -u $PBS_NODEFILE)
master_addr=$(head -n 1 <<< "$nodes")

# ランク数 (world size)
nnodes=$(wc -l <<< "$nodes")
gpus_per_node=8
world_size=$((nnodes * gpus_per_node))

echo "MASTER: $master_addr"
echo "NNODES: $nnodes"
echo "WORLD_SIZE: $world_size"

# ---- PyTorch distributed 実行 ----
mpirun --hostfile ./hostfile -np $NHOSTS \
    clnf.py \
        $CKPT_PREDICTOR \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --strategy $STRATEGY \
        --num_nodes $NUM_NODES \
        --devices $DEVICES_PER_NODE \
