# Deep Neuroevolution With Shared Weights: Ensembling A Better Solution
## Course project for cs231n in 2018

TODO: Description of the project.
TODO: Include diagrams.


## Setup

#### Anaconda environment

The setup that we use for this code is as follows:
1. Download andinstall [Anaconda](https://www.anaconda.com/download/#linux),
2. Create a new conda environment `conda create -n <your_env_name> python=3.6 anaconda`.
3. Use `source activate <your_env_name>` to start using the environment, and `source deactivate <your_env_name>` to stop.
4. When in your environment (after running `source activate <your_env_name>`) install the following dependencies, ensuring that you install using `conda` and the appropriate versions for your system, including if you want to use a GPU or not:
    1. [PyTorch](https://pytorch.org/),
    3. [TensorboardX](https://anaconda.org/conda-forge/tensorboardx), (Here's the [GitHub](https://github.com/lanpa/tensorboardX) repo for anyone interested),
    4. [Tqdm](https://anaconda.org/conda-forge/tqdm) for terminal progress bar.
    
Note that we will use jupyter notebooks, which are part of the default anaconda environment.

#### Getting the datasets

- MNist: No work required, we use the PyTorch built in dataset for MNist.
- Cifar: In `src/dataset/data` run the shell script by typing `./get_cifar10.sh`.
- Imagenet: Download from the portal site [here](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads), and unzip the data in `src/dataset/data/imagenet` as described [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset). For imagenet, we use 'standard normalization', as seen in this [code snippet](https://github.com/pytorch/examples/blob/e0d33a69bec3eb4096c265451dbb85975eb961ea/imagenet/main.py#L113-L126).


## Acknowledgements

Code for Inception architectures is adapted from [PyTorch's Model Zoo](https://github.com/Cadene/pretrained-models.pytorch). 

Code for counting FLOPs is addapted from [TODO](http://todo.com).
