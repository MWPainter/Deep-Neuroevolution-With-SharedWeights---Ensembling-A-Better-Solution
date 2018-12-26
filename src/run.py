import os
import torch as t
import torch.nn as nn

from dataset import get_imagenet_dataloader

from utils import cudafy
from utils import train_loop
from utils import parameter_magnitude, gradient_magnitude, update_magnitude, update_ratio
from utils import model_flops

from r2r import widen_network_, make_deeper_network_, inceptionv4, inceptionresnetv2





"""
File containing all of the tests that relate to Imagenet Tests.
"""





"""
Defining the training loop.
"""





def _make_optimizer_fn(model, lr, weight_decay):
    """
    The make optimizer function, as part of the interface for the "train_loop" function in utils.train_utils.

    :param model: The model to make an optimizer for.
    :param lr: The learning rate to use
    :param weight_decay: THe weight decay to use
    :return: The optimizer for the network optimizer, which is passed into the remaining
        training loop functions
    """
    return t.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)





def _load_fn(model, optimizer, load_file):
    """
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
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    t.save(checkpoint, filename)

    # Save it as the "best" checkpoint if we are the best
    if is_best_so_far:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        best_dirname = os.path.dirname(filename)
        if not os.path.exists(best_dirname):
            os.makedirs(best_dirname)
        t.save(checkpoint, best_filename)





def _update_op(model, optimizer, minibatch, iter, args):
    """
    Mnist Tests + Cifar Tests.

    The update op function, as part of the interface for the "train_loop" function in utils.training_utils.
    This function directly *performs* the update on model parameters. A number of losses will be computed,
    and are returned in a dictionary of PyTorch scalar Variables. The dictionary is used to log TensorBoard
    summaries, by the main training loop.

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
    xs, ys = cudafy(minibatch[0]), cudafy(minibatch[1])

    # Widen or deepen the network at the correct times
    if iter in args.widen_times or iter in args.deepen_times:
        # TODO: implement widen and deepen so that it can be Net2Net/R2R/NetMorph
        raise Exception("Not implemented yet")
        if iter in args.widen_times:
            # TODO: implement widen for InceptionV4 and InceptionResnetV2, such that it can be R2WiderR or Net2WiderNet
            pass
        elif iter in args.deepen_times:
            # TODO: implement deepen for InceptionV4 and InceptionResnetV2, such that it can be R2DeeperR or Net2DeeperNet
            pass
        optimizer = _make_optimizer_fn(model, args.lr, args.weight_decay)

    # Forward pass - compute a loss
    loss_fn = _make_loss_fn()
    ys_pred = model(xs)
    loss = loss_fn(ys_pred, ys)

    # Backward pass - make an update step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute loss and accuracy
    losses = {}
    losses['loss'] = loss
    losses['accuracy@1'], losses['accuracy@5'] = accuracy(ys_pred, ys, topk=(1,5))

    # Hackily keep track of model flops using the args/options dictionary
    if not hasattr(args, "total_flops"):
        args.total_flops = 0
    if not hasattr(args, "cur_model_flops_per_update") or iter in args.widen_times or iter in args.deepen_times:
        args.cur_model_flops_per_update = model_flops(model, xs)
    args.total_flops += args.cur_model_flops_per_update
    losses['iter_flops'] = args.cur_model_flops_per_update
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





def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k. I.e. if topk=(1,5) it will compute the precision@1 and the
    precision@k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res





def _make_loss_fn():
    """
    Helper to keep the definition of the loss function in a single place
    :return:
    """
    return nn.CrossEntropyLoss()





def _validation_loss(model, minibatch, args):
    """
    Computes the loss on a minibatch that has not been seen during training at all.

    :param model: The model to compute the validation loss for.
    :param minibatch: A PyTorch Varialbe of shape (N,D) to compute the validation loss over.
    :param args: The command line arguments (opt parser) passed in through the command line.
    :return: A PyTorch scalar Variable with value of the validation loss.
        Returns validation loss and validation accuracy.
    """

    return {'loss': 1.0,
            'accuracy@1': 1.0,
            'accuracy@5': 1.0}

    # If we have expended the number of flops for this test, then we should stop any updates
    if hasattr(args, "total_flops") and hasattr(args, "flops_budget") and args.total_flops >= args.flops_budget:
        return {}

    with t.no_grad():
        # Put in eval mode
        model.eval()

        # Unpack minibatch
        xs, ys = cudafy(minibatch[0]), minibatch[1]

        # Compute loss and accuracy
        loss_fn = _make_loss_fn()
        ys_pred = model(xs).cpu()
        loss = loss_fn(ys_pred, ys)
        accuracy1, accuracy5 = accuracy(ys_pred, ys, topk=(1,5))

        # Return the dictionary of losses
        return {'loss': loss,
                'accuracy@1': accuracy1,
                'accuracy@5': accuracy5}






"""
Net2Net duplicate tests
"""




def _net_2_wider_net_inception_test(args):
    """
    Duplicate the Net2WiderNet tests, on InceptionV4.

    :param args: Arguments from an ArgParser specifying how to run the trianing
    """
    # Force some of the command line args, so can't ruin the experimaent
    args.load = ""
    args.widen_times = []
    args.deepen_times = []
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Make the data loaders for imagenet
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # Create a pre-trained inception network
    model = inceptionv4()

    # Train for zero epochs, to run a single validation epoch on the base network
    args.shard = "teacher"
    epochs_cache = args.epochs
    args.epochs = 0
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)
    args.epochs = epochs_cache

    # Widen the network with R2R and train
    args.shard = "student_R2R"
    model = cudafy(widen_network_(model, new_channels=1.4, new_hidden_nodes=0, init_type='match_std',
                           function_preserving=True, multiplicative_widen=True))
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with Net2Net and train
    args.shard = "student_net2net"
    del model
    model = inceptionv4()
    # TODO: widen the network with Net2Net
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with NetMorph and train
    args.shard = "student_netmorph"
    del model
    model = inceptionv4()
    # TODO: widen the network with NetMorph
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen with random padding and train
    args.shard = "student_random_padding"
    del model
    model = inceptionv4()
    model = cudafy(widen_network_(model, new_channels=2, new_hidden_nodes=0, init_type='match_std',
                           function_preserving=False, multiplicative_widen=True))
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Make an inception network without any pretraining
    args.shard = "randomly_initialized"
    del model
    model = inceptionv4(pretrained=False)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)






def _net_2_wider_net_inception_resnet_test(args):
    """
    Duplicate the Net2WiderNet tests on InceptionResnetV2

    :param args: Arguments from an ArgParser specifying how to run the trianing
    """
    # Force some of the command line args, so can't ruin the experimaent
    args.load = ""
    args.widen_times = []
    args.deepen_times = []
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Make the data loaders for imagenet
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # Create a pre-trained inception network
    model = inceptionresnetv2()

    # Train for zero epochs, to run a single validation epoch on the base network
    args.shard = "teacher"
    epochs_cache = args.epochs
    args.epochs = 0
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)
    args.epochs = epochs_cache

    # Widen the network with R2R and train
    args.shard = "student_R2R"
    model = cudafy(widen_network_(model, new_channels=2, new_hidden_nodes=0, init_type='match_std',
                           function_preserving=True, multiplicative_widen=True))
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with NetMorph and train
    args.shard = "student_netmorph"
    del model
    model = inceptionresnetv2()
    # TODO: widen the network with NetMorph
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen with random padding and train
    args.shard = "student_random_padding"
    del model
    model = inceptionresnetv2()
    model = cudafy(widen_network_(model, new_channels=2, new_hidden_nodes=0, init_type='match_std',
                           function_preserving=False, multiplicative_widen=True))
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Make an inception network without any pretraining
    args.shard = "randomly_initialized"
    del model
    model = inceptionresnetv2(pretrained=False)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)






def _net_2_deeper_net_inception_test(args):
    """
    Duplicate the Net2DeeperNet tests, on InceptionV4.

    :param args: Arguments from an ArgParser specifying how to run the trianing
    """
    raise Exception("Not implemented yet")

    # Force some of the command line args, so can't ruin the experimaent
    args.load = ""
    args.widen_times = []
    args.deepen_times = []
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Make the data loaders for imagenet
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # Create a pre-trained inception network
    model = inceptionv4()

    # Train for zero epochs, to run a single validation epoch on the base network
    args.shard = "teacher"
    epochs_cache = args.epochs
    args.epochs = 0
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)
    args.epochs = epochs_cache

    # Widen the network with R2R and train
    args.shard = "student_R2R"
    # TODO: deepen network with R2DeeperR
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with Net2Net and train
    args.shard = "student_net2net"
    del model
    model = inceptionv4()
    # TODO: deepen the network with Net2Net
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen with random padding and train
    args.shard = "student_random_padding"
    del model
    model = inceptionv4()
    # TODO: deepen the network with a random padding
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Make an inception network without any pretraining
    args.shard = "randomly_initialized"
    del model
    model = inceptionv4(pretrained=False)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)






def _net_2_deeper_net_inception_resnet_test(args):
    """
    Duplicate the Net2DeeperNet tests, on InceptionV4.

    :param args: Arguments from an ArgParser specifying how to run the trianing
    """
    raise Exception("Not implemented yet")

    # Force some of the command line args, so can't ruin the experimaent
    args.load = ""
    args.widen_times = []
    args.deepen_times = []
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Make the data loaders for imagenet
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # Create a pre-trained inception network
    model = inceptionresnetv2()

    # Train for zero epochs, to run a single validation epoch on the base network
    args.shard = "teacher"
    epochs_cache = args.epochs
    args.epochs = 0
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)
    args.epochs = epochs_cache

    # Widen the network with R2R and train
    args.shard = "student_R2R"
    # TODO: deepen network with R2DeeperR
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with Net2Net and train
    args.shard = "student_net2net"
    del model
    model = inceptionresnetv2()
    # TODO: deepen the network with Net2Net
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen with random padding and train
    args.shard = "student_random_padding"
    del model
    model = inceptionresnetv2()
    # TODO: deepen the network with a random padding
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Make an inception network without any pretraining
    args.shard = "randomly_initialized"
    del model
    model = inceptionresnetv2(pretrained=False)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)




"""
R2R tests
"""






def _r_2_wider_r_inception_test(args):
    """
    Duplicate the Net2WiderNet tests, on InceptionV4.

    :param args: Arguments from an ArgParser specifying how to run the trianing
    """
    raise Exception("Not implemented yet")

    # Force some of the command line args, so can't ruin the experimaent
    args.load = ""
    args.deepen_times = []
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Make the data loaders for imagenet
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # Create a pre-trained inception network
    model = inceptionv4()

    # Train for zero epochs, to run a single validation epoch on the base network
    args.shard = "teacher"
    epochs_cache = args.epochs
    args.epochs = 0
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)
    args.epochs = epochs_cache

    # Widen the network with R2R and train
    args.shard = "student_R2R"
    # TODO: specify to widen the network with R2R (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with Net2Net and train
    args.shard = "student_net2net"
    del model
    model = inceptionv4()
    # TODO: specify to widen the network with Net2Net (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with NetMorph and train
    args.shard = "student_netmorph"
    del model
    model = inceptionv4()
    # TODO: specify to widen the network with NetMorph (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen with random padding and train
    args.shard = "student_random_padding"
    del model
    model = inceptionv4()
    # TODO: specify to widen the network with random padding (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Make an inception network without any pretraining
    args.shard = "randomly_initialized"
    del model
    model = inceptionv4(pretrained=False)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)






def _r_2_wider_r_inception_resnet_test(args):
    """
    Duplicate the Net2WiderNet tests on InceptionResnetV2

    :param args: Arguments from an ArgParser specifying how to run the trianing
    """
    raise Exception("Not implemented yet")

    # Force some of the command line args, so can't ruin the experimaent
    args.load = ""
    args.deepen_times = []
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Make the data loaders for imagenet
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # Create a pre-trained inception network
    model = inceptionresnetv2()

    # Train for zero epochs, to run a single validation epoch on the base network
    args.shard = "teacher"
    epochs_cache = args.epochs
    args.epochs = 0
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)
    args.epochs = epochs_cache

    # Widen the network with R2R and train
    args.shard = "student_R2R"
    # TODO: specify to widen the network with R2R (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with NetMorph and train
    args.shard = "student_netmorph"
    del model
    model = inceptionresnetv2()
    # TODO: specify to widen the network with NetMorph (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen with random padding and train
    args.shard = "student_random_padding"
    del model
    model = inceptionresnetv2()
    # TODO: specify to widen the network with random padding (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Make an inception network without any pretraining
    args.shard = "randomly_initialized"
    del model
    model = inceptionresnetv2(pretrained=False)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)






def _r_2_deeper_r_inception_test(args):
    """
    Duplicate the Net2DeeperNet tests, on InceptionV4.

    :param args: Arguments from an ArgParser specifying how to run the training
    """
    raise Exception("Not implemented yet")

    # Force some of the command line args, so can't ruin the experimaent
    args.load = ""
    args.widen_times = []
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Make the data loaders for imagenet
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # Create a pre-trained inception network
    model = inceptionv4()

    # Train for zero epochs, to run a single validation epoch on the base network
    args.shard = "teacher"
    epochs_cache = args.epochs
    args.epochs = 0
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)
    args.epochs = epochs_cache

    # Widen the network with R2R and train
    args.shard = "student_R2R"
    # TODO: specify to deepen the network with R2R (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with Net2Net and train
    args.shard = "student_net2net"
    del model
    model = inceptionv4()
    # TODO: specify to deepen the network with Net2Net (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen with random padding and train
    args.shard = "student_random_padding"
    del model
    model = inceptionv4()
    # TODO: specify to deepen the network with random padding (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Make an inception network without any pretraining
    args.shard = "randomly_initialized"
    del model
    model = inceptionv4(pretrained=False)
    args.deepen_times = []
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)






def _r_2_deeper_r_inception_resnet_test(args):
    """
    Duplicate the Net2DeeperNet tests, on InceptionV4.

    :param args: Arguments from an ArgParser specifying how to run the trianing
    """
    raise Exception("Not implemented yet")

    # Force some of the command line args, so can't ruin the experimaent
    args.load = ""
    args.widen_times = []
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Make the data loaders for imagenet
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # Create a pre-trained inception network
    model = inceptionresnetv2()

    # Train for zero epochs, to run a single validation epoch on the base network
    args.shard = "teacher"
    epochs_cache = args.epochs
    args.epochs = 0
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)
    args.epochs = epochs_cache

    # Widen the network with R2R and train
    args.shard = "student_R2R"
    # TODO: specify to deepen the network with R2R (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with Net2Net and train
    args.shard = "student_net2net"
    del model
    model = inceptionresnetv2()
    # TODO: specify to deepen the network with Net2Net (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen with random padding and train
    args.shard = "student_random_padding"
    del model
    model = inceptionresnetv2()
    # TODO: specify to deepen the network with random padding (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Make an inception network without any pretraining
    args.shard = "randomly_initialized"
    del model
    model = inceptionresnetv2(pretrained=False)
    args.deepen_times = []
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)





