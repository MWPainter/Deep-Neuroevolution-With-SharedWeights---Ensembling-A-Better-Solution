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
    2. [TensorFlow](https://anaconda.org/conda-forge/tensorflow),
    3. [TensorboardX](https://anaconda.org/conda-forge/tensorboardx), (Here's the [GitHub](https://github.com/lanpa/tensorboardX) repo for anyone interested),
    4. [Tqdm](https://anaconda.org/conda-forge/tqdm) for terminal progress bar.
    
Note that we will use jupyter notebooks, which are part of the default anaconda environment.

#### Getting the datasets

TODO: just direct them to the appropriate scripts

## Schedule


- Imagnet + inception: 27th Oct
- R2R finished + tests: 10th Nov
- Net2Net reproduced + encorporated in tests: 17th Nov
- Flops counting and visualizations: 1st Dec
- Draft: Dec 8st
- Finished: Dec 22nd
- Deadline: Jan 23rd



## Todo list:

#### Admin
- ~Remove /dataset directory~
- ~rename "Deriving resnet2resnet.py" to "tutorial.py"~
- Write a proper description + diagrams for the readme
- Write a description of the repo overview (what librarys are where)
- write a description of how to use the repo 
    - downloading the datasets
    - working through the tutorial/deriving the code
    - how the library is structured
    - how to use the librarys
    - how to visualize (i.e. the training scripts will save arrays, and run them in jupyter notebook)
- Clean up Deriving resnet2resnt/tutorial
    - Update the forword/high level wordy description at the beginning
    - Add a link to the paper describing the whole process
    - ~Level 1 headers for title. Level 2 headers for section. Level 3 headers for each bit of code~
    - ~Clean all of the code snippits~
- Clean up the library code
    - ~Properly split into files. Have difference packages:~
        - ~Dataset~
        - ~R2R~
        - ~Neuroevolution (or ne?)~
        - ~utils~
    - ~Have a main.py/train.py which is the entry point to training~ 
    - **Move everything from R2R into `utils`/`r2r`/`ne` appropraitely 
        - **Go through all of the files in utils, moving the things it says
        - **Go through all of the files in r2r, moving things ass appropriate (start with r2r.py then resblock then resnet and so on)
        - **Go through all of the files in ne, moving things as appropriate
    - **Make sure all \_\_init\_\_.py's are correct 
    - **Move everything from `batch_cifar_tests.py` to `main.py`, and make sure that the
        - **Test that the main.py scripts still run (fix broken imports...)
    - **Clean up the plotting code (jupyter notebook)
        - **Test that the plotting code works (train some small networks one time for say 200 iter)
    - **Add to this readme: description of the high level overview - i.e. the folders + plotting code description (like MSFT work)
- Test that everything works
- Update the docs directory, to include the historical docs from the class, and the up to date docs when they're written


#### (General) Coding Todo
- Saving model/training state to be able to recover state
- Use of the generic training loop from MSFT internship?
- Fix seeds for reproducibility
- Use data loaders
    - That's probably the bottle neck...?
- Add tensorboardX for plotting
    - And plot lots of things!
    - Use it to debug better
- ~Update extending_out_channels to re-use code from extending_in_channels~


#### Network Transforms Todo
- ~Alter the network transforms to actually be a function~
    - ~newLayer, newLayerAfter = R2R(layer, layerAfter)~
    - ~apply iteratively down for a widen~
    - ~have a suite of network transforms?~
    - ~get rid of the use of specific modules~
- ~Implement R2R for fully connected layers~
    - ~Prototype in tutorial.pynb first~
    - ~The transform can be implemented as a special case of a conv (where the spatial dimensions are 1x1. So just need to expand dims and squeeze dims around the general widen transform!!)~
- Find a better way to deal with residual connections than masking?
- Fix the R2DeeperR test
- Implement Net2Net in PyTorch (find another repo that does this?)
    - Run Net2Net through our tests
- Reproduce Net2Net results, withe exactly the same tests as Net2Net did
    - Run our network transforms through the same tests
- At expanding the network time, try freezing the old networks weights and seeing if they 
- Looking into multiple applications of the transforms
    - How to optimally apply them through training
- For inception networks, we actually will need \[newLayers], \[newLayersAfters] = R2R(\[layers], \[layersAfter])
    - I.e. we need arrays of layers, which are concatenated together

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
- Try merging R2R and Net2Net? (I.e. duplicated filters, with the negation as we have in R2R. Use this to show ours is a bit more general?)
    
    
#### Imagenet work
- Dataset work
    - Downloading the data
    - Dataset object
    - Definitely req's data loaders
- **Implement Inception-Resnetv2 architecture 
    - **ANd a smaller shallower version as a starting point
    - **Adapt implementation from [here](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionresnetv2.py) and [here](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py) and reference them correctly
- Repeate all experiments on imagenet using Inception-Resnetv2
    - Use the pretrained models from [here](https://github.com/Cadene/pretrained-models.pytorch) and reference correctly
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
    - ~Prototype this in tutorial.py first~
        - From prototyping we conclude:
        - To learn in the new layers, we actually need a strong reward signal (if we're at 99% training acc, then we won't learn anything in the new weights)
        - MNist is too easy to really see anything, we need to re-run on Cifar-10. That is, you can basically "solve" Mnist using a fully connected network, and the conv's can be very random still
        - we can demonstrate these things in the paper, as it may be interesting
        - maybe something is broken and the weights for Mnist should be much cleaner?
        - Try fixing the weights from before?
        - **Try using a two fc network to provide some visualizations for MNist instead??**
    - Show that the weights in the networks before and after widening
    - E.g. show that new layers in the network are random
    - E.g. show that new layers in the first layer of the network
        - Visualize new layers before and after some training
        - Maybe have 4 layers -> 8 layers, and visualize all layers at all points
            - Should see that we learn 4 NEW and DIFFERENT types of edge in the second set of 4 layers for example
            - Do the old layers change?
    - See the weight visualizations in cs231n assignments (and do something similar)
        - How they're like the edges and the different filters for edges/colours
    - Maybe consider the weights for the following situations:
        - Net2DeeperNet by adding a new FIRST layer
        - R2DeeperR by adding a new FIRST layer
- Run all of these visualizations on Net2Net and compare
    - Maybe/hopefully we see that our's learn the new edges etc faster
- Tensorboard summaries -> pretty seaboarn plots
    - OpenSource this too
    


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
- Our contributions:
    - A new set of function preserving transforms
        - Deepen works with residual connections, whereas net2deepernet works without residual connections
        - New weights are arbitrary (this is a blessing and a curse, as we have to consider inits, however, the freedom to initialize the weights however you like could allow a lot of flexibility, forseeably in meta learning or transfer learning)
        - (we believe?) that our schema is simpler to implement and use
        - We don't alter any weights in the function transformation. Therefore, if on widening you decide to keep the weights fixed, then you could still run the old network as before. That is, you can train YOLO mini, then widen, and finish training to YOLO full, and then keep both networks.
    


### Neuroevolution Todo
#### Coding Todo
- Uncomment and re-format to work with how r2r is now implemented
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