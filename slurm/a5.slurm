#!/bin/bash

# Set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=6-00:00:00

# set name of job
#SBATCH --job-name=ne5

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=mpainter@robots.ox.ac.uk

# Use a small partition
#SBATCH --partition=small

# Variables
CP_LOCK_FILE="/raid/local_scratch/mmp10-ihp03-"$SLURM_JOB_ID"/.lock"
CP_FIN_FILE="/raid/local_scratch/mmp10-ihp03-"$SLURM_JOB_ID"/.fin"
CODE_DIR="/jmain01/home/JAD020/ihp03/mmp10-ihp03/Deep-Neuroevolution-With-SharedWeights---Ensembling-A-Better-Solution/src/"
SSD_DIR="/raid/local_scratch/mmp10-ihp03-"$SLURM_JOB_ID"/"
IMAGENET_TAR=$CODE_DIR"dataset/data/imagenet.tar"

# Setup compute environment
module load python3/anaconda
module load cuda/9.0
source activate mpainter_ne

# Copy imagenet and unpack it. Also use files to for locking (the dataset copying)
echo "Copying Imagenet to local SSD"
if [ ! -f $CP_FIN_FILE ] && [ ! -f $CP_LOCK_FILE ]; then
    > $CP_LOCK_FILE
    cp $IMAGENET_TAR $SSD_DIR
    cd $SSD_DIR
    tar -xvf imagenet.tar
    rm $CP_LOCK_FILE
    > $CP_FIN_FILE
fi

# If the fin file doesn't exist, then another process is copying Imagenet, and we can wait for a while
while [ ! -f $CP_FIN_FILE ]; do
    echo "Sleeping for 10 seconds to wait for another process to finish copying and extracting imagenet"
    sleep 10
done

# Run
echo "Running experiment"
cd $CODE_DIR
python main.py f5aster_r2r_adagrad

# Clean compute environment
source deactivate
module purge
