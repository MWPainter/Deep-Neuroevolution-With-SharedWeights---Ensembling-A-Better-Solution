#!/bin/bash

# Set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=1-00:00:00

# set name of job
#SBATCH --job-name=ne_r2r

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=mpainter@robots.ox.ac.uk

# Use a small partition
#SBATCH --partition=small

# run the application
module load python3/anaconda
module load cuda/9.0
source activate mpainter_ne
cd /jmain01/home/JAD020/ihp03/mmp10-ihp03/Deep-Neuroevolution-With-SharedWeights---Ensembling-A-Better-Solution/src
python main.py r2wr
python main.py r2dr
source deactivate
module purge
