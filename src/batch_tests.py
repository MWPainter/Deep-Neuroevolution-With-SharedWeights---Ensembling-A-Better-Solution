import torch as t
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset import MnistDataset, CifarDataset

from utils import cudafy
from utils import train_loop
from utils import parameter_magnitude, gradient_magnitude, update_magnitude, update_ratio
from utils import model_flops

from r2r import widen_network_, make_deeper_network_, Mnist_Resnet, Cifar_Resnet





"""
This file contains all of the logic for all tests that need to be run on batch systems. That is, anything that trains
a network on Mnist or Cifar (which really are just small scale tests for R2R before inception).
"""





"""
Defining the training loop for the mnist and cifar tests.
"""





def _make_optimizer_fn(model, lr, weight_decay):
    """
    Mnist Tests + Cifar Tests.

    The make optimizer function, as part of the interface for the "train_loop" function in utils.train_utils.

    :param model: The model to make an optimizer for.
    :param lr: The learning rate to use
    :param weight_decay: THe weight decay to use
    :return: The optimizer for the network (discr_optimizer, gen_optimizer), which is passed into the remaining
        training loop functions
    """
    return t.optim.Adam(model.discr.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)





def _load_fn(model, optimizer, load_file):
    """
    Mnist Tests + Cifar Tests.

    The make load (model) function, as part of the interface for the "train_loop" function in utils.training_utils.

    :param model: The model to restore the state for
    :param optimizer: The optimizer to restore the state for (same as the return value from 'make_optimizer_fn')
    :param load_file: The filename for a checkpoint dict, saved using 'checkpoint_fn' below.
    :return: The restored model, optimizer and the current epoch with the best validation loss seen so far.
    """
    # Load state dict, and update the model and
    checkpoint = t.load(load_file)
    cur_epoch = checkpoint['next_epoch']
    best_val_loss = checkpoint['best_val_loss']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return the model and optimizer with restored state
    return model, optimizer, cur_epoch, best_val_loss





def _checkpoint_fn(model, optimizer, epoch, best_val_loss, checkpoint_dir, is_best_so_far):
    """
    Mnist Tests + Cifar Tests.

    The checkpoint function, as part of the interface for the "train_loop" function in utils.training_utils.
    This function will take the current state of training (i.e. the tuple (model, optimizer, epoch, best_val_loss)
    save it in the appropriate checkpoint file(s), in

    :param model: The model to take the checkpoint for
    :param optimizer: The optimizer to take the checkpoint for
    :param epoch: The current epoch in training
    :param best_val_loss: The best validation loss seen so far
    :param checkpoint_dir: The directory for which to save checkpoints in
    :param is_best_so_far: If the checkpoint is the best so far (with respect to the validation loss)
    :return: Nothing.
    """
    # Make the checkpoint
    checkpoint = {}
    checkpoint['next_epoch'] = epoch + 1
    checkpoint['best_val_loss'] = best_val_loss
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    # Save it as the most up to date checkpoint
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth.tar'.format(epoch=epoch))
    t.save(checkpoint, filename)

    # Save it as the "best" checkpoint if we are the best
    if is_best_so_far:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        t.save(checkpoint, best_filename)





def _update_op(model, optimizer, minibatch, iter, args):
    """
    Mnist Tests + Cifar Tests.

    The update op function, as part of the interface for the "train_loop" function in utils.training_utils.
    This function directly *performs* the update on model parameters. A number of losses will be computed,
    and are returned in a dictionary of PyTorch scalar Variables. The dictionary is used to log TensorBoard
    summaries, by the main training loop.

    The WGAN update will alternate between updating the discriminator and generator. Parameter clipping and
    gradient penalties are applied to the discriminator, ontop of a non-saturating Gan loss.

    :param model: A nn.Module object to take a gradient step
    :param optimizer: The optimizer object (as defined by make_optimizer_fn)
    :param minibatch: A minibatch of data to use for this update
    :param iter: The iteration index
    :param args: The command line arguments (opt parser) passed in through the command line.
    :return: An updated reference to the model and optimizer, and a dictionary from strings to PyTorch scalar Variables
            used for TensorBoard summaries.
    """
    # If we have expended the number of flops for this test, then we should stop any updates
    if hasattr(args, "total_flops") and hasattr(args, "flops_budget") and args.total_flops >= args.flops_budget:
        return model, optimizer, {}

    # Switch model to training mode, and cudafy minibatch
    model.train()
    xs, ys = minibatch
    xs, ys = cudafy(xs), cudafy(ys)

    # Widen or deepen the network at the correct times
    if iter in args.widen_times or iter in args.deepen_times:
        if iter in args.widen_times:
            model = widen_network_() # TODO
        elif iter in args.deepen_times:
            model = make_deeper_network_() # TODO
        optimizer = _make_optimizer_fn(model, args.lr, args.weight_decay)

    # Forward pass - compute a loss
    loss_fn = _make_loss_fn()
    ys_pred = model(xs)
    loss = loss_fn(ys_pred, ys)

    # Backward pass - make an update step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute the generator or discriminator loss, and make an appropriate step, clipping discriminator weights
    losses = {}
    losses['loss'] = loss
    losses['accuracy'] = _accuracy(ys_pred, ys)

    # Hackily keep track of model flops using the args/options dictionary
    if not hasattr(args, "total_flops"):
        args.total_flops = 0
    if not hasattr(args, "cur_model_flops_per_update") or iter in args.widen_times or iter in args.deepen_times:
        args.cur_model_flops_per_update = model_flops(model, minibatch)
    args.total_flops += args.cur_model_flops_per_update
    losses['iter_flops'] = args.cur_model_flops
    losses['total_flops'] = args.total_flops

    # Losses about weight norms etc. Only compute occasionally because this is heavyweight
    if iter % args.tb_log_freq == 0:
        weight_mag = parameter_magnitude(model)
        grad_mag = gradient_magnitude(model)
        param_update_mag = update_magnitude(model, args.lr, grad_mag)
        param_update_ratio = update_ratio(model, args.lr, weight_mag, grad_mag)
        losses['weight_mag'] = weight_mag
        losses['grad_mag'] = grad_mag
        losses['update_mag'] = param_update_mag
        losses['update_ratio'] = param_update_ratio

    # Return the model, optimizer and dictionary of 'losses'
    return model, optimizer, losses





def _accuracy(prediction, target):
    """
    Helper to compute the accuracy of predictions in a minibatch.
    :param prediction: Prediction probabilities
    :param target: Ground truth prediction probabilities (one-hot)
    :returns: Accuracy
    """
    _, pred_classes = t.max(prediction, 1)
    _, actual_classes = t.max(target, 1)
    return t.mean((pred_classes == actual_classes).type(t.float))





def _make_loss_fn():
    """
    Helper to keep the definition of the loss function in a single place
    :return:
    """
    return nn.BCEWithLogitsLoss()





def _validation_loss(model, minibatch):
    """
    Mnist Tests + Cifar Tests.

    Computes the loss on a minibatch that has not been seen during training at all.

    :param model: The model to compute the validation loss for.
    :param minibatch: A PyTorch Varialbe of shape (N,D) to compute the validation loss over.
    :return: A PyTorch scalar Variable with value of the validation loss.
        Returns the discriminator and generator losses.
    """
    # Put in eval mode
    model.eval()

    # Unpack minibatch
    xs, ys = cudafy(minibatch[0]), cudafy(minibatch[1])

    # Compute loss and accuracy
    loss_fn = _make_loss_fn()
    ys_pred = model(xs)
    loss = loss_fn(ys_pred, ys)
    accuracy = _accuracy(ys_pred, ys)

    # Return the dictionary of losses
    return {'loss': loss,
            'accuracy': accuracy}






def _mnist_test(args, model=None, widen_times=[], deepen_times=[], identity_init_network=False, use_random_padding=False):
    """
    Train a mnist resnet, widening and deepening at some points

    :param args: Arguments from an ArgParser specifying how to run the trianing
    :param model: Optionally pass in a model (that may have been trained already)
    :param widen_times: The timesteps to widen at
    :param deepen_times: The timesteps to deepen at
    :param use_random_padding: If any paddings should be random initialiations or identity/zero initialized
    :returns: The trained model
    """
    # Add widening times and deepening times into the args object
    args.widen_times = widen_times
    args.deepen_times = deepen_times
    args.use_random_padding = use_random_padding

    # Make the data loader objects
    train_dataset = MnistDataset(train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    val_dataset = MnistDataset(train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # Make the model, and run the training loop
    if model is None:
        model = Mnist_Resnet(identity_initialize=identity_init_network)
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # return the model in case we want to do anything else with it
    return model






def _cifar_test(args, model=None, widen_times=[], deepen_times=[], identity_init_network=False, use_random_padding=False):
    """
    Train a mnist resnet, widening and deepening at some points

    :param args: Arguments from an ArgParser specifying how to run the trianing
    :param model: Optionally pass in a model (that may have been trained already)
    :param widen_times: The timesteps to widen at
    :param deepen_times: The timesteps to deepen at
    :param use_random_padding: If any paddings should be random initialiations or identity/zero initialized
    :returns: The trained model
    """
    # Add widening times and deepening times into the args object
    args.widen_times = widen_times
    args.deepen_times = deepen_times
    args.use_random_padding = use_random_padding

    # Make the data loader objects
    train_dataset = CifarDataset(mode="train")
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    val_dataset = CifarDataset(mode="val")
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # Make the model, and run the training loop
    if model is None:
        model = Cifar_Resnet(identity_initialize=identity_init_network)
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # return the model in case we want to do anything else with it
    return model







"""
Defining instances of the mnist and cifar tests.
"""




# 2 stage widening, under the fixed number of epochs + flops budget
# copying tests from net2net, and net2deepernet





def mnist_identity_init_test(args):
    """
    Training runs to confirm that a network initialized so that it's trainable despite being initialized to a constant
    zero/identity function.
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.epochs = 6000/len(MnistDataset()) # TODO: move this to the defaults, and set defaults properly
    print(args.epochs)
    raise Exception("Remove this once checked")
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Add subdirectories to our tb logging and checkpoint
    tb_dir = os.path.join(args.tb_dir, "{shard}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, "{shard}")

    # identity initialize loop
    args.tb_dir = tb_dir.format(shard="identity_initialized")
    args.checkpoint_dir = checkpoint_dir.format(shard="identity_initialized")
    _mnist_test(args, identity_init_network=True)

    # randomly initialized loop
    args.tb_dir = tb_dir.format(shard="randomly_initialized")
    args.checkpoint_dir = checkpoint_dir.format(shard="randomly_initialized")
    _mnist_test(args, identity_init_network=False)





def cifar_identity_init_test(args):
    """
    Training runs to confirm that a network initialized so that it's trainable despite being initialized to a constant
    zero/identity function.
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Add subdirectories to our tb logging and checkpoint
    tb_dir = os.path.join(args.tb_dir, "{shard}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, "{shard}")

    # identity initialize loop
    args.tb_dir = tb_dir.format(shard="identity_initialized")
    args.checkpoint_dir = checkpoint_dir.format(shard="identity_initialized")
    _cifar_test(args, identity_init_network=True)

    # randomly initialized loop
    args.tb_dir = tb_dir.format(shard="randomly_initialized")
    args.checkpoint_dir = checkpoint_dir.format(shard="randomly_initialized")
    _cifar_test(args, identity_init_network=False)





def mnist_widen_test(args):
    """
    Training runs which widen the network, with R2WiderR or random padding

    Use different args for args.widen_times to perform a multi_stage widening test
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Add subdirectories to our tb logging and checkpoint
    tb_dir = os.path.join(args.tb_dir, "{shard}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, "{shard}")

    # identity initialize loop
    args.tb_dir = tb_dir.format(shard="R2WiderR")
    args.checkpoint_dir = checkpoint_dir.format(shard="R2WiderR")
    _mnist_test(args, widen_times=args.widen_times, use_random_padding=False)

    # randomly initialized loop
    args.tb_dir = tb_dir.format(shard="random_padding")
    args.checkpoint_dir = checkpoint_dir.format(shard="random_padding")
    _mnist_test(args, widen_times=args.widen_times, use_random_padding=True)





def cifar_widen_test(args):
    """
    Training runs which widen the network, with R2WiderR or random padding

    Use different args for args.widen_times to perform a multi_stage widening test
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Add subdirectories to our tb logging and checkpoint
    tb_dir = os.path.join(args.tb_dir, "{shard}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, "{shard}")

    # identity initialize loop
    args.tb_dir = tb_dir.format(shard="R2WiderR")
    args.checkpoint_dir = checkpoint_dir.format(shard="R2WiderR")
    _cifar_test(args, widen_times=args.widen_times, use_random_padding=False)

    # randomly initialized loop
    args.tb_dir = tb_dir.format(shard="random_padding")
    args.checkpoint_dir = checkpoint_dir.format(shard="random_padding")
    _cifar_test(args, widen_times=args.widen_times, use_random_padding=True)





def mnist_widen_with_budget_test(args):
    """
    Training runs which widen the network, with R2WiderR or random padding, with a budget on the number of flops that
    can be used in training

    Use different args for args.widen_times to perform a multi_stage widening test
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    raise Exception("Add a flops budget + make sure epochs default is enough to hit it")

    # Add subdirectories to our tb logging and checkpoint
    tb_dir = os.path.join(args.tb_dir, "{shard}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, "{shard}")

    # identity initialize loop
    args.tb_dir = tb_dir.format(shard="R2WiderR")
    args.checkpoint_dir = checkpoint_dir.format(shard="R2WiderR")
    _mnist_test(args, widen_times=args.widen_times, use_random_padding=False)

    # randomly initialized loop
    args.tb_dir = tb_dir.format(shard="random_padding")
    args.checkpoint_dir = checkpoint_dir.format(shard="random_padding")
    _mnist_test(args, widen_times=args.widen_times, use_random_padding=True)





def cifar_widen_with_budget_test(args):
    """
    Training runs which widen the network, with R2WiderR or random padding, with a budget on the number of flops that
    can be used in training

    Use different args for args.widen_times to perform a multi_stage widening test
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    raise Exception("Add a flops budget + make sure epochs default is enough to hit it")

    # Add subdirectories to our tb logging and checkpoint
    tb_dir = os.path.join(args.tb_dir, "{shard}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, "{shard}")

    # identity initialize loop
    args.tb_dir = tb_dir.format(shard="R2WiderR")
    args.checkpoint_dir = checkpoint_dir.format(shard="R2WiderR")
    _cifar_test(args, widen_times=args.widen_times, use_random_padding=False)

    # randomly initialized loop
    args.tb_dir = tb_dir.format(shard="random_padding")
    args.checkpoint_dir = checkpoint_dir.format(shard="random_padding")
    _cifar_test(args, widen_times=args.widen_times, use_random_padding=True)





def mnist_deepen_test(args):
    """
    Training runs which deepen the network, with R2DeeperR or random padding

    Use different args for args.deepen_times to perform a multi_stage deepening test
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Add subdirectories to our tb logging and checkpoint
    tb_dir = os.path.join(args.tb_dir, "{shard}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, "{shard}")

    # identity initialize loop
    args.tb_dir = tb_dir.format(shard="R2DeeperR")
    args.checkpoint_dir = checkpoint_dir.format(shard="R2DeeperR")
    _mnist_test(args, deepen_times=args.deepen_times, use_random_padding=False)

    # randomly initialized loop
    args.tb_dir = tb_dir.format(shard="random_padding")
    args.checkpoint_dir = checkpoint_dir.format(shard="random_padding")
    _mnist_test(args, deepen_times=args.deepen_times, use_random_padding=True)





def cifar_deepen_test(args):
    """
    Training runs which deepen the network, with R2DeeperR or random padding

    Use different args for args.deepen_times to perform a multi_stage deepening test
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Add subdirectories to our tb logging and checkpoint
    tb_dir = os.path.join(args.tb_dir, "{shard}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, "{shard}")

    # identity initialize loop
    args.tb_dir = tb_dir.format(shard="R2DeeperR")
    args.checkpoint_dir = checkpoint_dir.format(shard="R2DeeperR")
    _cifar_test(args, deepen_times=args.deepen_times, use_random_padding=False)

    # randomly initialized loop
    args.tb_dir = tb_dir.format(shard="random_padding")
    args.checkpoint_dir = checkpoint_dir.format(shard="random_padding")
    _cifar_test(args, deepen_times=args.deepen_times, use_random_padding=True)





def mnist_deepen_with_budget_test(args):
    """
    Training runs which deepen the network, with R2DeeperR or random padding, with a budget on the number of flops that
    can be used in training

    Use different args for args.deepen_times to perform a multi_stage deepening test
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    raise Exception("Add a flops budget + make sure epochs default is enough to hit it")

    # Add subdirectories to our tb logging and checkpoint
    tb_dir = os.path.join(args.tb_dir, "{shard}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, "{shard}")

    # identity initialize loop
    args.tb_dir = tb_dir.format(shard="R2DeeperR")
    args.checkpoint_dir = checkpoint_dir.format(shard="R2DeeperR")
    _mnist_test(args, deepen_times=args.deepen_times, use_random_padding=False)

    # randomly initialized loop
    args.tb_dir = tb_dir.format(shard="random_padding")
    args.checkpoint_dir = checkpoint_dir.format(shard="random_padding")
    _mnist_test(args, deepen_times=args.deepen_times, use_random_padding=True)





def cifar_deepen_with_budget_test(args):
    """
    Training runs which deepen the network, with R2DeeperR or random padding, with a budget on the number of flops that
    can be used in training

    Use different args for args.deepen_times to perform a multi_stage deepening test
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    raise Exception("Add a flops budget + make sure epochs default is enough to hit it")

    # Add subdirectories to our tb logging and checkpoint
    tb_dir = os.path.join(args.tb_dir, "{shard}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, "{shard}")

    # identity initialize loop
    args.tb_dir = tb_dir.format(shard="RDeeperR")
    args.checkpoint_dir = checkpoint_dir.format(shard="R2DeeperR")
    _cifar_test(args, deepen_times=args.deepen_times, use_random_padding=False)

    # randomly initialized loop
    args.tb_dir = tb_dir.format(shard="random_padding")
    args.checkpoint_dir = checkpoint_dir.format(shard="random_padding")
    _cifar_test(args, deepen_times=args.deepen_times, use_random_padding=True)





def mnist_net_to_net_style_test(args, widen=True):
    """
    Training runs which are used to duplicate the tests from the Net2Net paper

    :param widen: If we are testing R2WiderR rather than R2DeeperR
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Add subdirectories to our tb logging and checkpoint
    tb_dir = os.path.join(args.tb_dir, "{shard}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, "{shard}")

    # Teacher network training loop
    args.tb_dir = tb_dir.format(shard="Teacher")
    args.checkpoint_dir = checkpoint_dir.format(shard="Teacher")
    teacher_model = _mnist_test(args)

    # Make an R2R transformed model
    model = copy.deepcopy(teacher_model)
    raise Exception("Fix the todos here first")
    if widen:
        model = widen_network_()  # TODO
    else:
        model = make_deeper_network_()  # TODO

    # R2R transformed model training
    args.tb_dir = tb_dir.format(shard="R2R_student")
    args.checkpoint_dir = checkpoint_dir.format(shard="R2R_student")
    _mnist_test(args, model=model)

    # Make an randomly padded model
    model = copy.deepcopy(teacher_model)
    raise Exception("Fix the todos here first")
    if widen:
        model = widen_network_()  # TODO
    else:
        model = make_deeper_network_()  # TODO

    # R2R transformed model training
    args.tb_dir = tb_dir.format(shard="random_padding_student")
    args.checkpoint_dir = checkpoint_dir.format(shard="random_padding_student")
    _mnist_test(args, model=model)





def cifar_net_to_net_style_test(args, widen=True):
    """
    Training runs which are used to duplicate the tests from the Net2Net paper

    :param widen: If we are testing R2WiderR rather than R2DeeperR
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Add subdirectories to our tb logging and checkpoint
    tb_dir = os.path.join(args.tb_dir, "{shard}")
    checkpoint_dir = os.path.join(args.checkpoint_dir, "{shard}")

    # Teacher network training loop
    args.tb_dir = tb_dir.format(shard="Teacher")
    args.checkpoint_dir = checkpoint_dir.format(shard="Teacher")
    teacher_model = _mnist_test(args)

    # Make an R2R transformed model
    model = copy.deepcopy(teacher_model)
    raise Exception("Fix the todos here first")
    if widen:
        model = widen_network_()  # TODO
    else:
        model = make_deeper_network_()  # TODO

    # R2R transformed model training
    args.tb_dir = tb_dir.format(shard="R2R_student")
    args.checkpoint_dir = checkpoint_dir.format(shard="R2R_student")
    _mnist_test(args, model=model)

    # Make an randomly padded model
    model = copy.deepcopy(teacher_model)
    raise Exception("Fix the todos here first")
    if widen:
        model = widen_network_()  # TODO
    else:
        model = make_deeper_network_()  # TODO

    # R2R transformed model training
    args.tb_dir = tb_dir.format(shard="random_padding_student")
    args.checkpoint_dir = checkpoint_dir.format(shard="random_padding_student")
    _mnist_test(args, model=model)





