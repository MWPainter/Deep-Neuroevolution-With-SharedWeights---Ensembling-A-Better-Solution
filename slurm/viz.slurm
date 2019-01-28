#!/bin/bash

# Set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=1-00:00:00

# set name of job
#SBATCH --job-name=ne_viz

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

#python main.py mnist_weight_viz_r2r
#python main.py cifar_weight_viz_r2r

#python main.py mnist_weight_viz_r2r_conv
python main.py cifar_weight_viz_r2r_conv

#python main.py mnist_weight_viz_net2net
#python main.py cifar_weight_viz_net2net

#python main.py mnist_weight_viz_net2net_conv
python main.py cifar_weight_viz_net2net_conv

#python main.py mnist_weight_viz_netmorph
#python main.py cifar_weight_viz_netmorph

#python main.py mnist_weight_viz_netmorph_conv
python main.py cifar_weight_viz_netmorph_conv

#python main.py mnist_weight_viz
#python main.py cifar_weight_viz

#python main.py mnist_weight_viz_conv
python main.py cifar_weight_viz_conv

source deactivate
module purge
