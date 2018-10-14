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
4. When in your environment (after running `source activate <your_env_name>`) install the following dependencies, ensuring that you install using `conda` and the appropriate versions for your system:
    1. [PyTorch](https://pytorch.org/)
    2. [TensorboardX](https://anaconda.org/conda-forge/tensorboardx). (Here's the [GitHub](https://github.com/lanpa/tensorboardX) repo for anyone interested).
    
Note that we will use jupyter notebooks, which are part of the default anaconda environment.

#### Getting the datasets

TODO: just direct them to the appropriate scripts

## Schedule


- R2R finished + tests: 20th Oct
- Imagnet + inception: 27th Oct
- Net2Net reproduced + encorporated in tests: 10th Nov
- Flops counting and visualizations: 24th Nov
- Draft: Dec 1st
- Finished: Dec 15th
- Deadline: Jan 8th



## Todo list:
#### Cleaning up the tutorial.py file
- At: "### Training a small netowork, with 3 resnet modules, 3 max pools, and a fc output" markdown
- Update the r2r block 1 to actually add noise, as a ratio of the average weight in the filter
    - A function to compute the stddev value in the filter
    - Add noise with a stddev *= noise ratio
- Make sure all of the "assumptions" made in R2R_block_v1 are updated in the later versions

#### Admin
- Remove /dataset directory
- ~rename "Deriving resnet2resnet.py" to "tutorial.py"~
- Write a proper description + diagrams for the readme
- write a description of how to use the repo 
    - downloading the datasets
    - working through the tutorial/deriving the code
    - how the library is structured
    - how to use the librarys
    - how to visualize (i.e. the training scripts will save 
- Clean up Deriving resnet2resnt/tutorial
    - Update the forword
    - Add a link to the paper describing the whole process
    - Have a high level, wordy description
    - Level 1 headers for title. Level 2 headers for section. Level 3 headers for each bit of code
    - Clean all of the code snippits
- Clean up the library code
    - Properly split into files
    - Have a main.py/train.py which is the entry point to training
    - Clean up the plotting code (jupyter notebook)
- Test that everything works


#### (General) Coding Todo
- Saving model/training state to be able to recover state
- Use of the generic training loop from MSFT internship?
- Fix seeds for reproducibility
- Use data loaders
    - That's probably the bottle neck...?
- Add tensorboardX for plotting
    - And plot lots of things!
    - Use it to debug better


#### Network Transforms Todo
- Alter the network transforms to actually be a function
    - newLayerBefore, newLayer, newLayerAfter = R2R(layerBefore, layer, layerAfter)
    - apply iteratively down for a widen
    - have a suite of network transforms?
    - get rid of the use of specific modules
- Implement R2R for fully connected layers
- Fix the R2DeeperR test
- Implement Net2Net in PyTorch (find another repo that does this?)
    - Run Net2Net through our tests
- Reproduce Net2Net results, withe exactly the same tests as Net2Net did
    - Run our network transforms through the same tests
- At expanding the network time, try freezing the old networks weights and seeing if they 
- Looking into multiple applications of the transforms
    - How to optimally apply them through training

TODO: clean this next bullet up into more concise ideas
- Better understanding of the learning rate problem
    - Consider something like the variance of the gradient, and how this changes w.r.t the magnitude of the initializations of the new varialbes
    - Maybe some ideas from [here](https://arxiv.org/pdf/1310.6343.pdf) are useful
    - Potentially use a taylor expansion somehow
        - Something to try evaluate the gradients about a point (a single example) and see how it varies the update in general
        - Want the updates for the existing weights to be around the same as 
        - So scale the new weights accordingly, so as not cause new large gradients
        - Gradients for the existing network should be approximately the same as if the new layers were not there
            - The difficulty is it being for ANY example
        - Will involve lots of chain rule?
        - And how the gradients propogate
        - I tried some scheme emperically evaluating the gradients to scale new weights appropraitely, but it was v.v. noisy.
    - I did have some math that seemed like it would work
        - Taylor expansions of the gradient, w.r.t new weights. Variance of the gradients, taylor expanded the gradient expression. 
    - Compare the different ideas with some test? Compare lots of curves for different schemes
    
    
#### Imagenet work
- Dataset work
    - Downloading the data
    - Dataset object
    - Definitely req's data loaders
- Implement Inception-Resnetv2 architecture 
    - ANd a smaller shallower version as a starting point
- Repeate all experiments on imagenet using Inception-Resnetv2
- Data augmentation? 
    - Hve this mostly from MSFT internship
  
  
#### Evalutation/Visualizations
- Implement Net2Net in PyTorch + compare
- Cleanly repeat tests for Net2Net (i.e. every test we have for R2R, we should be adding a N2N curve)
- Working diagrams of R2R tests done (flops)
- Properly computing the FLOPs in the forward and backward passes
    - Open source this code
    - Message Emad about it and adding it into PyTorch at some point?
- Taking nn.Modules and printing an architecture diagram from it
    - TODO: find the repo that we had for this (there is a github repo taking PyTorch modules and printing graphs)
- Saliency maps / Class visualizations
    - Run before and after widening
    - Something that shows the difference
    - I.e. what does each layer represent?
        - New layers after widening should be random
        - New layers after training should learn something
- Weight visualizatioons
    - Show that the weights in the networks before and after widening
    - E.g. show that new layers in the network are random
    - E.g. show that new layers in the first layer of the network
        - Visualize new layers before and after some training
        - Maybe have 4 layers -> 8 layers, and visualize all layers at all points
            - Should see that we learn 4 NEW and DIFFERENT types of edge in the second set of 4 layers for example
            - Do the old layers change?
    - See the weight visualizations in cs231n assignments (and do something similar)
        - How they're like the edges and the different filters for edges/colours
- Run all of these visualizations on Net2Net and compare
    - Maybe/hopefully we see that our's learn the new edges etc faster


#### Paper/Writing Todo
- Plan out headings and bullets
    - Copy relevant parts from the project report
- Related Work/Finish writing literature survey for the paper
    - Classification state of the art
        - (Important)
    - Network transforms (Net2Net)
        - (Important)
    - Transfer learning
    - Incremental learning
        - (Is this a thing?)
        - (Related to transfer learning?)
    - Some data augmentation stuff
    - Neural archiutecture search 
        - (SHORT, as this is related but not direct here)
    - Efficient Neural architecture search
        - Short again
- Write paper! 
    - Aim to be complete about a month before deadline?
- In the introduction
    - Look at all the papers about R2R and how people are using it
    - Mention the many uses that it has
    


### Neuroevolution Todo
#### Coding Todo
- Make it actually work again (larger population? change algo a bit?)
- Fine tuning
- Fix ensemble idea
- Hyper networks
    - (Read a paper about neural nets that output the weights of other networks/architectures)
    - Some vague idea about learning some embedding for neural network encodings, and optimizing in some latent space
        - I.E. Some more clever search than just neuroevolution in the architecture space
- Checking say 10 small perterbations from Inception, and check that inception isn't necessarily the best of the 10
    - A check that inception isn't a max point in the architecture space
    - And it's worth continuing the optimization in the architecture space


#### Neuroevolution Paper/Writing Todo
- Plan out headings and bullets
    - Copy relevant parts from the project report
- Related work/Finish writing literature survey for the paper
    - Similar to the network transform
    - Reference self
    - Heavy literature review of transfer learning
    - Heavy literature review of nerual architecture search
    - Heavy literature review of efficient Neural architecture search