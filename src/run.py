import os
import copy
import torch as t
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset import get_imagenet_dataloader, CifarDataset

from utils import cudafy
from utils import train_loop
from utils import parameter_magnitude, gradient_magnitude, update_magnitude, update_ratio
from utils import model_flops
from utils import count_parameters

from r2r import widen_network_, make_deeper_network_, inceptionv4, resnet10, resnet18, resnet26





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
    for widen in model.load_with_widens:
        if widen:
            model.widen(1.414)
        else:
            if len(model.deepen_indidces_list) == 0:
                raise Exception("Too many deepen times for this test.")
            deepen_indices = model.deepen_indidces_list.pop(0)
            model.deepen(deepen_indices)
        model = cudafy(model)

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
        if iter in args.widen_times:
            model.widen(1.414)
        if iter in args.deepen_times:
            if len(args.deepen_indidces_list) == 0:
                raise Exception("Too many deepen times for this test.")
            deepen_indices = args.deepen_indidces_list.pop(0)
            model.deepen(deepen_indices)
        model = cudafy(model)
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
    args.total_flops = 0
    epochs_cache = args.epochs
    args.epochs = 0
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)
    args.epochs = epochs_cache

    # Widen the network with R2R and train
    args.shard = "student_R2R"
    args.total_flops = 0
    model = cudafy(widen_network_(model, new_channels=1.4, new_hidden_nodes=0, init_type='match_std',
                           function_preserving=True, multiplicative_widen=True))
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with Net2Net and train
    args.shard = "student_net2net"
    args.total_flops = 0
    del model
    model = inceptionv4()
    # TODO: widen the network with Net2Net
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with NetMorph and train
    args.shard = "student_netmorph"
    args.total_flops = 0
    del model
    model = inceptionv4()
    # TODO: widen the network with NetMorph
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen with random padding and train
    args.shard = "student_random_padding"
    args.total_flops = 0
    del model
    model = inceptionv4()
    model = cudafy(widen_network_(model, new_channels=2, new_hidden_nodes=0, init_type='match_std',
                           function_preserving=False, multiplicative_widen=True))
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Make an inception network without any pretraining
    args.shard = "randomly_initialized"
    args.total_flops = 0
    del model
    model = inceptionv4(pretrained=False)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)





def net_2_wider_net_resnet(args):
    """
    Duplicates of the Net2WiderNet tests, on cifar.
    :param args:
    :return:
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    args.widen_times = []
    args.deepen_times = []

    # Make the data loader objects
    train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # Teacher network training loop
    args.shard = "teacher_w_residual"
    args.total_flops = 0
    initial_model = resnet18(thin=True, thinning_ratio=4*1.414)
    teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                               _validation_loss, args)

    # R2R
    model = copy.deepcopy(teacher_model)
    model.widen(1.414)
    args.shard = "R2R_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # # NetMorph
    model = copy.deepcopy(teacher_model)
    model.morphism_scheme="netmorph"
    model.widen(1.414)
    args.shard = "NetMorph_student"
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # RandomPadding
    model = copy.deepcopy(teacher_model)
    model.function_preserving = False
    model.widen(1.414)
    args.shard = "RandomPadding_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Random init start
    model_= resnet18(thin=True, thinning_ratio=4*1.414)
    model.widen(1.414)
    args.shard = "Completely_Random_Init"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Net2Net teacher
    initial_model = resnet18(thin=True, thinning_ratio=4*1.414, use_residual=False, morphism_scheme="net2net")
    args.shard = "teacher_w_out_residual"
    args.total_flops = 0
    teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                               _validation_loss, args)

    # Net2Net
    model = copy.deepcopy(teacher_model)
    model.widen(1.414)
    args.shard = "Net2Net_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)





def net_2_deeper_net_resnet(args):
    """
    Duplicates of the Net2WiderNet tests, on cifar.
    :param args:
    :return:
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    args.widen_times = []
    args.deepen_times = []

    # Make the data loader objects
    train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # Teacher network training loop
    args.shard = "teacher_w_residual"
    args.total_flops = 0
    initial_model = resnet10(thin=True, thinning_ratio=4)
    teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                               _validation_loss, args)

    # R2R
    model = copy.deepcopy(teacher_model)
    model.deepen([1,1,1,1])
    model = cudafy(model)
    args.shard = "R2R_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # RandomPadding
    model = copy.deepcopy(teacher_model)
    model.function_preserving = False
    model.deepen([1,1,1,1])
    model = cudafy(model)
    args.shard = "RandomPadding_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Random init start
    model_= resnet10(thin=True, thinning_ratio=4)
    model.deepen([1,1,1,1])
    model = cudafy(model)
    args.shard = "Completely_Random_Init"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Net2Net teacher
    initial_model = resnet10(thin=True, thinning_ratio=4, use_residual=False, morphism_scheme="net2net")
    args.shard = "teacher_w_out_residual"
    args.total_flops = 0
    teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                               _validation_loss, args)

    # Net2Net
    model = copy.deepcopy(teacher_model)
    model.deepen([1,1,1,1])
    model = cudafy(model)
    args.shard = "Net2Net_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
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
    args.total_flops = 0
    epochs_cache = args.epochs
    args.epochs = 0
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)
    args.epochs = epochs_cache

    # Widen the network with R2R and train
    args.shard = "student_R2R"
    args.total_flops = 0
    # TODO: specify to widen the network with R2R (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with Net2Net and train
    args.shard = "student_net2net"
    args.total_flops = 0
    del model
    model = inceptionv4()
    # TODO: specify to widen the network with Net2Net (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen the network with NetMorph and train
    args.shard = "student_netmorph"
    args.total_flops = 0
    del model
    model = inceptionv4()
    # TODO: specify to widen the network with NetMorph (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Widen with random padding and train
    args.shard = "student_random_padding"
    args.total_flops = 0
    del model
    model = inceptionv4()
    # TODO: specify to widen the network with random padding (in the training loop)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # Make an inception network without any pretraining
    args.shard = "randomly_initialized"
    args.total_flops = 0
    del model
    model = inceptionv4(pretrained=False)
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)





def r_2_wider_r_resnet(args):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Make the data loader objects
    train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # R2R
    model = resnet18(thin=True, thinning_ratio=4*1.414)
    args.shard = "R2R_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Net2Net
    model = resnet18(thin=True, thinning_ratio=4*1.414, morphism_scheme="net2net")
    args.shard = "Net2Net_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # RandomPadding
    model_= resnet18(thin=True, thinning_ratio=4*1.414, function_preserving=False)
    args.shard = "RandomPadding_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # # NetMorph
    model = resnet18(thin=True, thinning_ratio=4*1.414, morphism_scheme="netmorph")
    args.shard = "NetMorph_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Random init start
    model = resnet18(thin=True, thinning_ratio=4*1.414)
    model.widen(1.414)
    args.shard = "Completely_Random_Init"
    args.total_flops = 0
    args.widen_times = []
    args.deepen_times = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Teacher network training loop
    model = resnet18(thin=True, thinning_ratio=4*1.414)
    args.shard = "teacher_w_residual"
    args.total_flops = 0
    args.widen_times = []
    args.deepen_times = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                               _validation_loss, args)

    # Net2Net teacher
    model = resnet18(thin=True, thinning_ratio=4*1.414, use_residual=False)
    args.shard = "teacher_w_out_residual"
    args.total_flops = 0
    args.widen_times = []
    args.deepen_times = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                               _validation_loss, args)





def r_2_deeper_r_resnet(args):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    # Make the data loader objects
    train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # R2R
    model = resnet10(thin=True, thinning_ratio=4)
    args.deepen_indidces_list = [[1,1,1,1]]
    args.shard = "R2R_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Net2Net
    model = resnet10(thin=True, thinning_ratio=4, use_residual=False, morphism_scheme="net2net")
    args.deepen_indidces_list = [[1,1,1,1]]
    args.shard = "Net2Net_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # RandomPadding
    model = resnet10(thin=True, thinning_ratio=4, function_preserving=False)
    args.deepen_indidces_list = [[1,1,1,1]]
    args.shard = "RandomPadding_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Random init start
    model = resnet10(thin=True, thinning_ratio=4)
    model.deepen([1,1,1,1])
    args.shard = "Completely_Random_Init"
    args.total_flops = 0
    args.widen_times = []
    args.deepen_times = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Teacher network training loop
    model = resnet10(thin=True, thinning_ratio=4)
    args.shard = "teacher_w_residual"
    args.total_flops = 0
    args.widen_times = []
    args.deepen_times = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                               _validation_loss, args)

    # Net2Net teacher
    model = resnet10(thin=True, thinning_ratio=4, use_residual=False)
    args.shard = "teacher_w_out_residual"
    args.total_flops = 0
    args.widen_times = []
    args.deepen_times = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                               _validation_loss, args)






"""
Learning rate and weight decay adaption tests
"""





def quadruple_widen_run(args):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    args.deepen_times = []
    if len(args.widen_times) != 4:
        raise Exception("Widening times needs to be a list of length 4 for this test")

    # Make the data loader objects
    train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # R2R
    model = resnet18(thin=True, thinning_ratio=4*4)
    args.shard = "R2R"
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Net2Net
    model = resnet18(thin=True, thinning_ratio=4*4, morphism_scheme="net2net")
    args.shard = "Net2Net_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)





def double_deepen_run(args):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    args.widen_times = []
    if len(args.deepen_times) != 2:
        raise Exception("Deepening times needs to be a list of length 2 for this test")
    args.deepen_indidces_list = [[1,1,1,1], [1,1,1,1]]

    # Make the data loader objects
    train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # R2R
    model = resnet10(thin=True, thinning_ratio=4)
    args.shard = "R2R"
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Net2Net
    model = resnet10(thin=True, thinning_ratio=4, morphism_scheme="net2net")
    args.shard = "Net2Net_student"
    args.total_flops = 0
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)





def double_widen_and_deepen_run(args):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    if len(args.widen_times) != 2:
        raise Exception("Widening times needs to be a list of length 2 for this test")
    if len(args.deepen_times) != 2:
        raise Exception("Deepening times needs to be a list of length 2 for this test")
    args.deepen_indidces_list = [[1,1,1,1], [1,1,1,1]]

    # Make the data loader objects
    train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # R2R
    model = resnet10(thin=True, thinning_ratio=4*2)
    args.shard = "R2R"
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Net2Net
    model = resnet10(thin=True, thinning_ratio=4*2, morphism_scheme="net2net")
    args.shard = "Net2Net_student"
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)






"""
Showing that R2R == faster training tests
"""








def r2r_faster_test_part_1(args):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    if len(args.widen_times) != 2:
        raise Exception("Widening times needs to be a list of length 2 for this test")
    if len(args.deepen_times) != 2:
        raise Exception("Deepening times needs to be a list of length 2 for this test")
    args.deepen_indidces_list = [[1,1,1,1], [0,1,2,1]]

    # Make the data loaders for imagenet
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # R2R
    model = resnet26(thin=True, thinning_ratio=2)
    # HACK
    model.load_with_widens = [False]
    model.deepen_indidces_list = [[1,1,1,1], [0,1,2,1]]
    # HACK
    args.shard = "R2R_Then_Widened"
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)








def r2r_faster_test_part_2(args):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    if len(args.widen_times) != 0:
        raise Exception("Widening times needs to be a list of length 0 for this test")
    if len(args.deepen_times) != 0:
        raise Exception("Deepening times needs to be a list of length 0 for this test")

    # Make the data loaders for imagenet
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # Larger model trained staight up
    model = resnet26(thin=True, thinning_ratio=2)
    model.deepen([1,1,1,1])
    model.widen(1.414)
    model.deepen([0,1,2,1])
    model.widen(1.414)
    # HACK
    model.load_with_widens = []
    model.deepen_indidces_list = []
    # HACK
    model = cudafy(model)
    args.shard = "Full_Model"
    args.widen_times = []
    args.deepen_times = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)








def r2r_faster_test_part_3(args):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    if len(args.widen_times) != 0:
        raise Exception("Widening times needs to be a list of length 2 for this test")
    if len(args.deepen_times) != 0:
        raise Exception("Deepening times needs to be a list of length 2 for this test")
    args.deepen_indidces_list = [[1,1,1,1], [0,1,2,1]]

    # Make the data loaders for imagenet
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # R2R
    model = resnet26(thin=True, thinning_ratio=1.414)
    # HACK
    model.load_with_widens = []
    model.deepen_indidces_list = []
    # HACK
    args.shard = "R2R_Teacher"
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)








def r2r_faster_test_part_4(args):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    if len(args.widen_times) != 1:
        raise Exception("Widening times needs to be a list of length 2 for this test")
    if len(args.deepen_times) != 2:
        raise Exception("Deepening times needs to be a list of length 2 for this test")
    args.deepen_indidces_list = [[1,1,1,1], [0,1,2,1]]

    # Make the data loaders for imagenet
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # R2R
    model = resnet26(thin=True, thinning_ratio=1.414)
    # HACK
    model.load_with_widens = []
    model.deepen_indidces_list = []
    # HACK
    args.shard = "R2R_One_Widen"
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)