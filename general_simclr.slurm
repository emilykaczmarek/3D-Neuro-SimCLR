#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=10
#SBATCH --error=/scratch/e/ekacz/SSL/SimCLR/full_res_fixed_transforms/job_%j/%j_0_log.err
#SBATCH --job-name=simclr
#SBATCH --mem-per-gpu=78G
#SBATCH --gpus-per-node=4
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --open-mode=append
#SBATCH --output=/scratch/e/ekacz/SSL/SimCLR/full_res_fixed_transforms/job_%j/%j_0_log.out
#SBATCH --time=8:00:00

# command
#module --quiet load python/3.10
source /home/e/ekacz/SSL/IJEPA_MRI/ijepa-main/.venv/bin/activate

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=29601

srun --ntasks-per-node=1 --nodes=3 bash -c "
torchrun \
  --nproc_per_node=4 \
  --nnodes=3 \
  --node_rank=\$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  main.py
"



#MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

#torchrun --nproc_per_node=4 --nnodes=3 --node_rank=$SLURM_NODEID --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n 1) --master_port=29600 main.py


#torchrun --nproc_per_node=4 --master_port=29600 main.py

# command
#module --quiet load python/3.10
#source simclr-env/bin/activate

# Set environment variables
#export MASTER_ADDR=$(hostname)
#export MASTER_PORT=29511

# Use torchrun to launch DDP
#torchrun --nproc-per-node=4 main.py
#srun python main.py 
