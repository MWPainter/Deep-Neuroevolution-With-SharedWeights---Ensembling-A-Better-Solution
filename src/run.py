import os
import copy
import torch as t
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset import get_imagenet_dataloader, CifarDataset, Cifar100Dataset, ProductDataset, SvhnDataset

from utils import cudafy
from utils import train_loop
from utils import count_parameters, parameter_magnitude, gradient_magnitude, gradient_l2_norm, update_magnitude, update_ratio
from utils import model_flops

from r2r import * # widen_network_, make_deeper_network_, inceptionv4, resnet10, resnet18, resnet26, resnet10_cifar, resnet18_cifar





"""
File containing all of the tests that relate to Imagenet Tests.
"""





"""
Defining the training loop.
"""





def _make_optimizer_fn(model, lr, weight_decay, args, momentum=None):
    """
    The make optimizer function, as part of the interface for the "train_loop" function in utils.train_utils.

    :param model: The model to make an optimizer for.
    :param lr: The learning rate to use
    :param weight_decay: THe weight decay to use
    :param args: Unused
    :param momentum: Unused
    :return: The optimizer for the network optimizer, which is passed into the remaining
        training loop functions
    """
    return t.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)





def _make_optimizer_fn_adagrad(model, lr, weight_decay, args, momentum=None):
    """
    The make optimizer function, as part of the interface for the "train_loop" function in utils.train_utils.

    :param model: The model to make an optimizer for.
    :param lr: The learning rate to use
    :param weight_decay: THe weight decay to use
    :param args: Unused
    :param momentum: Unused
    :return: The optimizer for the network optimizer, which is passed into the remaining
        training loop functions
    """
    return t.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)





def _make_optimizer_fn_rms(model, lr, weight_decay, args, momentum=None):
    """
    The make optimizer function, as part of the interface for the "train_loop" function in utils.train_utils.

    :param model: The model to make an optimizer for.
    :param lr: The learning rate to use
    :param weight_decay: THe weight decay to use
    :param args: Unused
    :param momentum: Unused
    :return: The optimizer for the network optimizer, which is passed into the remaining
        training loop functions
    """
    return t.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=args.momentum)





def _make_optimizer_fn_sgd(model, lr, weight_decay, args, momentum=None):
    """
    The make optimizer function, as part of the interface for the "train_loop" function in utils.train_utils.

    :param model: The model to make an optimizer for.
    :param lr: The learning rate to use
    :param weight_decay: THe weight decay to use
    :param args: Command line args
    :param momentum: Override for momentum if we want
    :return: The optimizer for the network optimizer, which is passed into the remaining
        training loop functions
    """
    if momentum is None:
        momentum = args.momentum
    return t.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)





def _load_fn(model, optimizer, load_file):
    """
    The make load (model) function, as part of the interface for the "train_loop" function in utils.training_utils.

    :param model: The model to restore the state for
    :param optimizer: The optimizer to restore the state for (same as the return value from 'make_optimizer_fn')
    :param load_file: The filename for a checkpoint dict, saved using 'checkpoint_fn' below.
    :return: The restored model, optimizer and the current epoch with the best validation loss seen so far.
    """
    # for widen in model.load_with_widens:
    #     if widen:
    #         model.widen(1.414)
    #     else:
    #         if len(model.deepen_indidces_list) == 0:
    #             raise Exception("Too many deepen times for this test.")
    #         deepen_indices = model.deepen_indidces_list.pop(0)
    #         model.deepen(deepen_indices)
    #     model = cudafy(model)

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




def _adjust_learning_rate(args, iter, optimizer):
    """
    Helper to adjust learning rate dynamically, and updates it in the optimizer
    """
    if iter in args.lr_drops:
        idx = min(args.lr_drops.index(iter), len(args.lr_drop_mag)-1) # bounds checked access of drop magnitude
        args.lr /= args.lr_drop_mag[idx]
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    # Momentum (if any) is turned off for a while to avoid exploding gradients for the first epoch
    # momentum_switch_on_times = [w + 5005 for w in args.widen_times] + [d + 5005 for d in args.deepen_times]
    # if hasattr(optimizer, 'momentum') and iter in momentum_switch_on_times:
    #     optimizer.momentum = args.momentum




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

    # Adjust the learning rate if need be
    _adjust_learning_rate(args, iter, optimizer)

    # Widen or deepen the network at the correct times
    # Also adjust weight decay if args are set to 
    if iter in args.widen_times or iter in args.deepen_times:
        weight_mag_before = parameter_magnitude(model)
        if iter in args.widen_times:
            print("Widening!")
            model.widen(1.5)
        if iter in args.deepen_times:
            print("Deepening!")
            if len(args.deepen_indidces_list) == 0:
                raise Exception("Too many deepen times for this test.")
            deepen_indices = args.deepen_indidces_list.pop(0)
            model.deepen(deepen_indices, minibatch=xs)
        weight_mag_after = parameter_magnitude(model)
        model = cudafy(model)
        if args.adjust_weight_decay:
            weight_decay_ratio = weight_mag_before / weight_mag_after
            args.weight_decay *= weight_decay_ratio
        optimizer = _make_optimizer_fn(model, args.lr, args.weight_decay, args) #, momentum=0.0)

    # Forward pass - compute a loss
    loss_fn = _make_loss_fn()
    ys_pred = model(xs)
    loss = loss_fn(ys_pred, ys)

    # Backward pass - make an update step, clipping parameters appropriately
    optimizer.zero_grad()
    loss.backward()
    if args.grad_clip > 0.0:
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    # Compute loss and accuracy
    losses = {}
    losses['loss'] = loss
    losses['accuracy@1'], losses['accuracy@5'] = accuracy(ys_pred, ys, topk=(1,5))

    # Hackily keep track of model flops using the args/options dictionary
    if iter == 0 or not hasattr(args, "total_flops"):
        args.total_flops = 0
    if iter == 0 or not hasattr(args, "cur_model_flops_per_update") or iter in args.widen_times or iter in args.deepen_times:
        args.cur_model_flops_per_update = model_flops(model, xs)
    args.total_flops += args.cur_model_flops_per_update
    losses['iter_flops'] = args.cur_model_flops_per_update
    losses['total_flops'] = args.total_flops

    # Losses about weight norms etc. Only compute occasionally because this is heavyweight
    if iter % args.tb_log_freq == 0:
        weight_mag = parameter_magnitude(model)
        grad_mag = gradient_magnitude(model)
        grad_l2 = gradient_l2_norm(model)
        param_update_mag = update_magnitude(model, args.lr, grad_mag)
        param_update_ratio = update_ratio(model, args.lr, weight_mag, grad_mag)
        losses['weight_mag'] = weight_mag
        losses['grad_mag'] = grad_mag
        losses['grad_l2'] = grad_l2
        losses['update_mag'] = param_update_mag
        losses['update_ratio'] = param_update_ratio

    # Return the model, optimizer and dictionary of 'losses'
    return model, optimizer, losses




def _update_op_cts_eval(model, optimizer, minibatch, iter, args):
    """
    Same as _update_op, but assumes that minibatch is a tuple (train_x, train_y, val_x, val_y), allowing for 
    'continuous evaluation'.
    """
    # If we have expended the number of flops for this test, then we should stop any updates
    if hasattr(args, "total_flops") and hasattr(args, "flops_budget") and args.total_flops >= args.flops_budget:
        return model, optimizer, {}

    # Switch model to training mode, and cudafy minibatch
    model.train()
    xs, ys = cudafy(minibatch[0]), cudafy(minibatch[1])
    val_xs, val_ys = cudafy(minibatch[2]), cudafy(minibatch[3])

    # Adjust the learning rate if need be
    _adjust_learning_rate(args, iter, optimizer)

    # Widen or deepen the network at the correct times
    if iter in args.widen_times or iter in args.deepen_times:
        weight_mag_before = parameter_magnitude(model)
        if iter in args.widen_times:
            print("Widening!")
            model.widen(1.5)
        if iter in args.deepen_times:
            print("Deepening!")
            if len(args.deepen_indidces_list) == 0:
                raise Exception("Too many deepen times for this test.")
            deepen_indices = args.deepen_indidces_list.pop(0)
            model.deepen(deepen_indices, minibatch=xs)
        weight_mag_after = parameter_magnitude(model)
        model = cudafy(model)
        if args.adjust_weight_decay:
            weight_decay_ratio = weight_mag_before / weight_mag_after
            args.weight_decay *= weight_decay_ratio
        optimizer = _make_optimizer_fn(model, args.lr, args.weight_decay, args) #, momentum=0.0)

    # Forward pass - compute a loss
    loss_fn = _make_loss_fn()
    ys_pred = model(xs)
    loss = loss_fn(ys_pred, ys)

    # Backward pass - make an update step, clipping parameters appropriately
    optimizer.zero_grad()
    loss.backward()
    if args.grad_clip > 0.0:
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    # Validation forward pass
    val_ys_pred = model(val_xs)

    # Compute loss and accuracy
    losses = {}
    losses['loss'] = loss
    losses['accuracy@1'], losses['accuracy@5'] = accuracy(ys_pred, ys, topk=(1,5))
    losses['valacc@1'], losses['valacc@5'] = accuracy(val_ys_pred, val_ys, topk=(1,5))

    # Hackily keep track of model flops using the args/options dictionary
    if iter == 0 or not hasattr(args, "total_flops"):
        args.total_flops = 0
    if iter == 0 or not hasattr(args, "cur_model_flops_per_update") or iter in args.widen_times or iter in args.deepen_times:
        args.cur_model_flops_per_update = model_flops(model, xs)
    args.total_flops += args.cur_model_flops_per_update
    losses['iter_flops'] = args.cur_model_flops_per_update
    losses['total_flops'] = args.total_flops

    # Losses about weight norms etc. Only compute occasionally because this is heavyweight
    if iter % args.tb_log_freq == 0:
        weight_mag = parameter_magnitude(model)
        grad_mag = gradient_magnitude(model)
        grad_l2 = gradient_l2_norm(model)
        param_update_mag = update_magnitude(model, args.lr, grad_mag)
        param_update_ratio = update_ratio(model, args.lr, weight_mag, grad_mag)
        losses['weight_mag'] = weight_mag
        losses['grad_mag'] = grad_mag
        losses['grad_l2'] = grad_l2
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
Anomolies tests
"""









def net_2_net_overfit_example(args):
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
    train_loader, val_loader = _make_cifar_data_loaders(args)

    orig_lr = args.lr

    # # Teacher network training loop
    # args.shard = "deepen_teacher"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.weight_decay = 1.0e-6 # remove weight decay mostly
    # initial_model = resnet10_cifar(thin=True, thinning_ratio=16)
    # teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn,
    #                            _update_op, _validation_loss, args)
    #
    # # R2R
    # model = copy.deepcopy(teacher_model)
    # model.deepen([2, 2, 0, 0])
    # model = cudafy(model)
    # args.shard = "deepen_student"
    # args.total_flops = 0
    # args.lr = orig_lr / 5.0
    # # args.weight_decay = 3.0e-3
    # args.weight_decay = 1.0e-6
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # Teacher network training loop
    args.shard = "widen_teacher"
    args.total_flops = 0
    args.lr = 0.0
    args.weight_decay = 0.0 # less weight decay mostly
    initial_model = orig_resnet18_cifar(thin=True, thinning_ratio=4, num_classes=10)
    teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn,
                               _update_op, _validation_loss, args)

    # R2R
    model = copy.deepcopy(teacher_model)
    model.widen(1.5)
    model = cudafy(model)
    args.shard = "widen_student"
    args.total_flops = 0
    lr_drop = 0.0
    args.lr = orig_lr / lr_drop
    args.weight_decay = 10000
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Random init comparison
    args.shard = "random_init"
    args.total_flops = 0
    args.lr = 0.0
    args.weight_decay = 10000 # less weight decay mostly
    initial_model = orig_resnet18_cifar(thin=True, thinning_ratio=4, num_classes=10)
    initial_model.widen(1.5)
    teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn,
                               _update_op, _validation_loss, args)









def r_2_r_weight_init_example(args):
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
    train_loader, val_loader = _make_cifar_data_loaders(args)

    orig_lr = args.lr

    # He init widen
    model = orig_resnet18_cifar(thin=True, thinning_ratio=8, num_classes=10)
    model.init_scheme = 'He'
    args.shard = "widen_student_he"
    args.total_flops = 0
    args.lr = orig_lr
    args.widen_times = [1532*5]
    args.deepen_times = []
    args.lr_drops = args.widen_times
    args.lr_drop_mag = [1.0]
    args.weight_decay = 1.0e-3
    args.adjust_weight_decay = False
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # He init deepen
    # model = resnet10_cifar(thin=True, thinning_ratio=8)
    # model.init_scheme = 'He'
    # args.deepen_indidces_list = [[2,2,0,0]]
    # args.shard = "deepen_student_he"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.widen_times = []
    # args.deepen_times = [1532*5]
    # args.lr_drops = args.deepen_times
    # args.lr_drop_mag = [1.0]
    # args.weight_decay = 1.0e-3
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # Scaled init widen
    model = orig_resnet18_cifar(thin=True, thinning_ratio=8, num_classes=10)
    args.shard = "widen_student_std_match"
    args.total_flops = 0
    args.lr = orig_lr
    args.widen_times = [1532*5]
    args.deepen_times = []
    args.lr_drops = args.widen_times
    args.lr_drop_mag = [1.0]
    args.weight_decay = 1.0e-3
    args.adjust_weight_decay = True
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)

    # Scaled init deepen
    # model = resnet10_cifar(thin=True, thinning_ratio=8)
    # args.deepen_indidces_list = [[2,2,0,0]]
    # args.shard = "deepen_student_std_match"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.widen_times = []
    # args.deepen_times = [1532*5]
    # args.lr_drops = args.deepen_times
    # args.lr_drop_mag = [1.0]
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)






"""
Net2Net duplicate tests
"""




def _net_2_wider_net_inception_test(args):
    """
    Duplicate the Net2WiderNet tests, on InceptionV4.

    :param args: Arguments from an ArgParser specifying how to run the trianing
    """
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Force some of the command line args, so can't ruin the experimaent
    # args.load = ""
    # args.widen_times = []
    # args.deepen_times = []
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget

    # # Make the data loaders for imagenet
    # train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    # val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # # Create a pre-trained inception network
    # model = inceptionv4()

    # # Train for zero epochs, to run a single validation epoch on the base network
    # args.shard = "teacher"
    # args.total_flops = 0
    # epochs_cache = args.epochs
    # args.epochs = 0
    # model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)
    # args.epochs = epochs_cache

    # # Widen the network with R2R and train
    # args.shard = "student_R2R"
    # args.total_flops = 0
    # model = cudafy(widen_network_(model, new_channels=1.4, new_hidden_nodes=0, init_type='match_std',
    #                        function_preserving=True, multiplicative_widen=True))
    # model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)

    # # Widen the network with Net2Net and train
    # args.shard = "student_net2net"
    # args.total_flops = 0
    # del model
    # model = inceptionv4()
    # # TODO: widen the network with Net2Net
    # model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)

    # # Widen the network with NetMorph and train
    # args.shard = "student_netmorph"
    # args.total_flops = 0
    # del model
    # model = inceptionv4()
    # # TODO: widen the network with NetMorph
    # model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)

    # # Widen with random padding and train
    # args.shard = "student_random_padding"
    # args.total_flops = 0
    # del model
    # model = inceptionv4()
    # model = cudafy(widen_network_(model, new_channels=2, new_hidden_nodes=0, init_type='match_std',
    #                        function_preserving=False, multiplicative_widen=True))
    # model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)

    # # Make an inception network without any pretraining
    # args.shard = "randomly_initialized"
    # args.total_flops = 0
    # del model
    # model = inceptionv4(pretrained=False)
    # model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)





def net_2_wider_net_resnet(args):
    """
    Duplicates of the Net2WiderNet tests, on cifar.
    :param args:
    :return:
    """
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # args.load = ""
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # args.widen_times = []
    # args.deepen_times = []

    # # Make the data loader objects
    # train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # orig_lr = args.lr
    # orig_wd = args.weight_decay
    # scaling_factor = 1.5

    # # Teacher network training loop
    # args.shard = "teacher_w_residual"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.weight_decay = 5.0e-3
    # initial_model = resnet18_cifar(thin=True, thinning_ratio=8)
    # teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                            _validation_loss, args)

    # # R2R
    # model = copy.deepcopy(teacher_model)
    # model.widen(scaling_factor)
    # args.shard = "R2R_student"
    # args.total_flops = 0
    # # args.lr = orig_lr / 5.0
    # # args.weight_decay = 2.0e-3
    # args.lr = orig_lr / 5.0
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)


    # # # NetMorph
    # model = copy.deepcopy(teacher_model)
    # model.morphism_scheme="netmorph"
    # model.widen(scaling_factor)
    # args.shard = "NetMorph_student"
    # args.total_flops = 0
    # # args.lr = orig_lr
    # # args.weight_decay = 1.0e-5
    # args.lr = orig_lr / 5.0
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)



    # # RandomPadding
    # model = copy.deepcopy(teacher_model)
    # model.function_preserving = False
    # model.init_scheme = 'He'
    # model.widen(scaling_factor)
    # args.shard = "RandomPadding_student"
    # args.total_flops = 0
    # # args.lr = orig_lr / 10.0
    # # args.weight_decay = 3.0e-3
    # args.lr = orig_lr / 5.0
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)


    # # Random init start
    # model = resnet18_cifar(thin=True, thinning_ratio=8)
    # model.widen(scaling_factor)
    # args.shard = "Completely_Random_Init"
    # args.total_flops = 0
    # # args.lr = orig_lr / 2.0
    # # args.weight_decay = 1.0e-3
    # args.lr = orig_lr / 2.0
    # args.weight_decay = 1.0e-4
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)


    # # Net2Net teacher
    # initial_model = resnet18_cifar(thin=True, thinning_ratio=8, use_residual=False, morphism_scheme="net2net")
    # args.shard = "teacher_w_out_residual"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.weight_decay = 5.0e-3
    # teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                            _validation_loss, args)

    # # Net2Net
    # model = copy.deepcopy(teacher_model)
    # model.widen(scaling_factor)
    # args.shard = "Net2Net_student"
    # args.total_flops = 0
    # # args.lr = orig_lr / 5.0
    # # args.weight_decay = 1.0e-3
    # args.lr = orig_lr / 5.0
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)


    # # Random init start v2
    # model = resnet18_cifar(thin=True, thinning_ratio=8, use_residual=False)
    # model.widen(scaling_factor)
    # args.shard = "Completely_Random_Init_Net2Net"
    # args.total_flops = 0
    # # args.lr = orig_lr / 2.0
    # # args.weight_decay = 1.0e-3
    # args.lr = orig_lr
    # args.weight_decay = 1.0e-4
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)





def net_2_wider_net_resnet_hyper_search(args):
    """
    Duplicates of the Net2WiderNet tests, on cifar.
    :param args:
    :return:
    """
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # args.load = ""
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # args.widen_times = []
    # args.deepen_times = []

    # # Make the data loader objects
    # train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)


    # scaling_factor = 1.5
    # orig_lr = args.lr

    # # Teacher network training loop
    # args.shard = "teacher_w_residual"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.weight_decay = 5.0e-3
    # initial_model = resnet18_cifar(thin=True, thinning_ratio=8)
    # r2r_teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                            _validation_loss, args)


    # # Net2Net teacher
    # initial_model = resnet18_cifar(thin=True, thinning_ratio=8, use_residual=False, morphism_scheme="net2net")
    # args.shard = "teacher_w_out_residual"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.weight_decay = 5.0e-3
    # n2n_teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                            _validation_loss, args)

    # # R2R
    # model = copy.deepcopy(r2r_teacher_model)
    # model.widen(scaling_factor)
    # args.shard = "R2R_student"
    # args.total_flops = 0
    # args.lr = orig_lr / 5
    # args.weight_decay = 5.0e-3
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)



    # # RandomPadding
    # model = copy.deepcopy(r2r_teacher_model)
    # model.init_scheme = 'He'
    # model.function_preserving = False
    # model.widen(scaling_factor)
    # args.shard = "RandomPadding_student"
    # args.total_flops = 0
    # args.weight_decay = 5.0e-3
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Lets be unfair to ourselves, and only hyperparam search for the completely randomly initialized network, and the
    # # other function preserving transforms
    # for lr_drop in [1.0, 2.0, 5.0, 10.0]:
    #     for weight_decay in [0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2]:

    #         # Set args
    #         args.lr = orig_lr / lr_drop
    #         args.weight_decay = weight_decay

    #         # NetMorph
    #         model = copy.deepcopy(r2r_teacher_model)
    #         model.morphism_scheme="netmorph"
    #         model.widen(scaling_factor)
    #         args.shard = "NetMorph_student_lr={lr}_wd={wd}".format(lr=args.lr, wd=args.weight_decay)
    #         args.total_flops = 0
    #         train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)


    #         # Random init start
    #         model = resnet18_cifar(thin=True, thinning_ratio=8)
    #         model.widen(scaling_factor)
    #         args.shard = "Completely_Random_Init_lr={lr}_wd={wd}".format(lr=args.lr, wd=args.weight_decay)
    #         args.total_flops = 0
    #         train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)

    #         # Net2Net
    #         model = copy.deepcopy(n2n_teacher_model)
    #         model.widen(scaling_factor)
    #         args.shard = "Net2Net_student_lr={lr}_wd={wd}".format(lr=args.lr, wd=args.weight_decay)
    #         args.total_flops = 0
    #         train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)




def net_2_deeper_net_resnet(args):
    """
    Duplicates of the Net2DeeperNet tests, on cifar.
    :param args:
    :return:
    """
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # args.load = ""
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # args.widen_times = []
    # args.deepen_times = []

    # # Make the data loader objects
    # train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
    #                         num_workers=args.workers, pin_memory=True)

    # orig_lr = args.lr
    # orig_wd = args.weight_decay

    # # Teacher network training loop
    # args.shard = "teacher_w_residual"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.weight_decay = 5.0e-3
    # initial_model = resnet10_cifar(thin=True, thinning_ratio=6)
    # teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn,
    #                            _update_op,
    #                            _validation_loss, args)

    # # R2R
    # model = copy.deepcopy(teacher_model)
    # model.deepen([2,2,0,0])
    # model = cudafy(model)
    # args.shard = "R2R_student"
    # args.total_flops = 0
    # # args.lr = orig_lr / 5.0
    # # args.weight_decay = 3.0e-3
    # args.lr = orig_lr / 5.0
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # RandomPadding
    # model = copy.deepcopy(teacher_model)
    # model.init_scheme = 'He'
    # model.function_preserving = False
    # model.deepen([2,2,0,0])
    # model = cudafy(model)
    # args.shard = "RandomPadding_student"
    # args.total_flops = 0
    # # args.lr = orig_lr / 10.0
    # # args.weight_decay = 3.0e-3
    # args.lr = orig_lr / 5.0
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Random init start
    # model = resnet10_cifar(thin=True, thinning_ratio=6)
    # model.deepen([2,2,0,0])
    # model = cudafy(model)
    # args.shard = "Completely_Random_Init"
    # args.total_flops = 0
    # # args.lr = orig_lr
    # # args.weight_decay = 1.0e-3
    # args.lr = orig_lr / 2.0
    # args.weight_decay = 1.0e-4
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Net2Net teacher
    # initial_model = resnet10_cifar(thin=True, thinning_ratio=6, use_residual=False, morphism_scheme="net2net")
    # args.shard = "teacher_w_out_residual"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.weight_decay = 5.0e-3
    # teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn,
    #                            _update_op,
    #                            _validation_loss, args)

    # # Net2Net
    # model = copy.deepcopy(teacher_model)
    # model = cudafy(model)
    # model.deepen([2,2,0,0], minibatch=next(iter(train_loader))[0].to('cuda'))
    # model = cudafy(model)
    # args.shard = "Net2Net_student"
    # args.total_flops = 0
    # # args.lr = orig_lr / 5.0
    # # args.weight_decay = 1.0e-3
    # args.lr = orig_lr / 5.0
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)


    # # Random init start v2
    # model = resnet10_cifar(thin=True, thinning_ratio=6, use_residual=False)
    # model.deepen([2,2,0,0])
    # args.shard = "Completely_Random_Init_Net2Net"
    # args.total_flops = 0
    # # args.lr = orig_lr / 2.0
    # # args.weight_decay = 1.0e-3
    # args.lr = orig_lr
    # args.weight_decay = 1.0e-4
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)










def net_2_deeper_net_resnet_hyper_search(args):
    """
    Duplicates of the Net2WiderNet tests, on cifar.
    :param args:
    :return:
    """
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # args.load = ""
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # args.widen_times = []
    # args.deepen_times = []

    # # Make the data loader objects
    # train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)


    # orig_lr = args.lr

    # # Teacher network training loop
    # args.shard = "teacher_w_residual"
    # args.total_flops = 0
    # initial_model = resnet10_cifar(thin=True, thinning_ratio=6)
    # r2r_teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                            _validation_loss, args)

    # # Net2Net teacher
    # initial_model = resnet10_cifar(thin=True, thinning_ratio=6, use_residual=False, morphism_scheme="net2net")
    # args.shard = "teacher_w_out_residual"
    # args.total_flops = 0
    # args.weight_decay = 5.0e-3
    # n2n_teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                            _validation_loss, args)

    # # R2R
    # model = copy.deepcopy(r2r_teacher_model)
    # model.deepen([2,2,0,0])
    # model = cudafy(model)
    # args.shard = "R2R_student"
    # args.total_flops = 0
    # args.lr = orig_lr / 10
    # args.weight_decay = 5.0e-3
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Lets be unfair to ourselves, and only hyperparam search for the completely randomly initialized network, and the
    # # other function preserving transforms
    # for lr_drop in [1.0, 2.0, 5.0, 10.0]:
    #     for weight_decay in [0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2]:

    #         args.lr = orig_lr / lr_drop
    #         args.weight_decay = weight_decay

    #         # RandomPadding
    #         model = copy.deepcopy(r2r_teacher_model)
    #         model.init_scheme = 'He'
    #         model.function_preserving = False
    #         model.deepen([2,2,0,0])
    #         model = cudafy(model)
    #         args.shard = "RandomPadding_student_lr={lr}_wd={wd}".format(lr=args.lr, wd=args.weight_decay)
    #         args.total_flops = 0
    #         train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)


    #         # Random init start
    #         model = resnet10_cifar(thin=True, thinning_ratio=6)
    #         model.deepen([2,2,0,0])
    #         model = cudafy(model)
    #         args.shard = "Completely_Random_Init_lr={lr}_wd={wd}".format(lr=args.lr, wd=args.weight_decay)
    #         args.total_flops = 0
    #         train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)

    #         # Net2Net
    #         model = copy.deepcopy(n2n_teacher_model)
    #         model = cudafy(model)
    #         model.deepen([2,2,0,0], minibatch=cudafy(next(iter(train_loader))[0]))
    #         model = cudafy(model)
    #         args.shard = "Net2Net_student_lr={lr}_wd={wd}".format(lr=args.lr, wd=args.weight_decay)
    #         args.total_flops = 0
    #         train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)






"""
R2R tests
"""






def _r_2_wider_r_inception_test(args):
    """
    Duplicate the Net2WiderNet tests, on InceptionV4.

    :param args: Arguments from an ArgParser specifying how to run the trianing
    """
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # raise Exception("Not implemented yet")

    # # Force some of the command line args, so can't ruin the experimaent
    # args.load = ""
    # args.deepen_times = []
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget

    # # Make the data loaders for imagenet
    # train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    # val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # # Create a pre-trained inception network
    # model = inceptionv4()

    # # Train for zero epochs, to run a single validation epoch on the base network
    # args.shard = "teacher"
    # args.total_flops = 0
    # epochs_cache = args.epochs
    # args.epochs = 0
    # model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)
    # args.epochs = epochs_cache

    # # Widen the network with R2R and train
    # args.shard = "student_R2R"
    # args.total_flops = 0
    # # TODO: specify to widen the network with R2R (in the training loop)
    # model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)

    # # Widen the network with Net2Net and train
    # args.shard = "student_net2net"
    # args.total_flops = 0
    # del model
    # model = inceptionv4()
    # # TODO: specify to widen the network with Net2Net (in the training loop)
    # model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)

    # # Widen the network with NetMorph and train
    # args.shard = "student_netmorph"
    # args.total_flops = 0
    # del model
    # model = inceptionv4()
    # # TODO: specify to widen the network with NetMorph (in the training loop)
    # model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)

    # # Widen with random padding and train
    # args.shard = "student_random_padding"
    # args.total_flops = 0
    # del model
    # model = inceptionv4()
    # # TODO: specify to widen the network with random padding (in the training loop)
    # model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)

    # # Make an inception network without any pretraining
    # args.shard = "randomly_initialized"
    # args.total_flops = 0
    # del model
    # model = inceptionv4(pretrained=False)
    # model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                    _validation_loss, args)





def r_2_wider_r_resnet(args):
    # Fix some args for the test (shouldn't ever be loading anythin)
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # args.load = ""
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget

    # # Make the data loader objects
    # train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # orig_lr = args.lr

    # # R2R
    # model = resnet18_cifar(thin=True, thinning_ratio=8)
    # args.shard = "R2R_student"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.lr_drops = args.widen_times
    # # args.lr_drop_mag = [5.0]
    # # args.weight_decay = 2.0e-3
    # args.lr_drop_mag = [5.0]
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Net2Net
    # model = resnet18_cifar(thin=True, thinning_ratio=8, morphism_scheme="net2net")
    # args.shard = "Net2Net_student"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.lr_drops = args.widen_times
    # # args.lr_drop_mag = [2.0]
    # # args.weight_decay = 2.0e-3
    # args.lr_drop_mag = [5.0]
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # RandomPadding
    # model = resnet18_cifar(thin=True, thinning_ratio=8, function_preserving=False)
    # model.init_scheme = 'He'
    # args.shard = "RandomPadding_student"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.lr_drops = args.widen_times
    # # args.lr_drop_mag = [10.0]
    # # args.weight_decay = 3.0e-3
    # args.lr_drop_mag = [5.0]
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # # NetMorph
    # model = resnet18_cifar(thin=True, thinning_ratio=8, morphism_scheme="netmorph")
    # args.shard = "NetMorph_student"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.lr_drops = args.widen_times
    # args.lr_drop_mag = [5.0]
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Random init start
    # model = resnet18_cifar(thin=True, thinning_ratio=8)
    # model.widen(1.5)
    # args.shard = "Completely_Random_Init"
    # args.total_flops = 0
    # args.widen_times = []
    # args.deepen_times = []
    # args.lr = orig_lr / 2.0
    # args.lr_drops = []
    # args.lr_drop_mag = [0.0]
    # args.weight_decay = 5.0e-3
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Random init start v2
    # model = resnet18_cifar(thin=True, thinning_ratio=8, use_residual=False)
    # model.widen(1.5)
    # args.shard = "Completely_Random_Init_Net2Net"
    # args.total_flops = 0
    # args.widen_times = []
    # args.deepen_times = []
    # args.lr = orig_lr
    # args.lr_drops = []
    # args.lr_drop_mag = [0.0]
    # args.weight_decay = 5.0e-3
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Teacher network training loop
    # model = resnet18_cifar(thin=True, thinning_ratio=8)
    # args.shard = "teacher_w_residual"
    # args.total_flops = 0
    # args.widen_times = []
    # args.deepen_times = []
    # args.lr = orig_lr
    # args.lr_drops = []
    # args.lr_drop_mag = 0.0
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                            _validation_loss, args)

    # # Net2Net teacher
    # model = resnet18_cifar(thin=True, thinning_ratio=8, use_residual=False)
    # args.shard = "teacher_w_out_residual"
    # args.total_flops = 0
    # args.widen_times = []
    # args.deepen_times = []
    # args.lr = orig_lr
    # args.lr_drops = []
    # args.lr_drop_mag = [0.0]
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                            _validation_loss, args)






def r_2_deeper_r_resnet(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # args.load = ""
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget

    # # Make the data loader objects
    # train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # orig_lr = args.lr

    # # R2R
    # model = resnet10_cifar(thin=True, thinning_ratio=6)
    # args.deepen_indidces_list = [[2,2,0,0]]
    # args.shard = "R2R_student"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.lr_drops = args.deepen_times
    # # args.lr_drop_mag = [5.0]
    # # args.weight_decay = 3.0e-3
    # args.lr_drop_mag = [5.0]
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Net2Net
    # model = resnet10_cifar(thin=True, thinning_ratio=6, use_residual=False, morphism_scheme="net2net")
    # args.deepen_indidces_list = [[2,2,0,0]]
    # args.shard = "Net2Net_student"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.lr_drops = args.deepen_times
    # # args.lr_drop_mag = [5.0]
    # # args.weight_decay = 1.0e-3
    # args.lr_drop_mag = [5.0]
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # RandomPadding
    # model = resnet10_cifar(thin=True, thinning_ratio=6, function_preserving=False)
    # model.init_scheme = 'He'
    # args.deepen_indidces_list = [[2,2,0,0]]
    # args.shard = "RandomPadding_student"
    # args.total_flops = 0
    # args.lr = orig_lr
    # args.lr_drops = args.deepen_times
    # # args.lr_drop_mag = [10.0]
    # # args.weight_decay = 3.0e-3
    # args.lr_drop_mag = [5.0]
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Random init start
    # model = resnet10_cifar(thin=True, thinning_ratio=6)
    # model.deepen([2,2,0,0])
    # args.shard = "Completely_Random_Init"
    # args.total_flops = 0
    # args.widen_times = []
    # args.deepen_times = []
    # args.lr = orig_lr / 2.-0
    # args.lr_drops = []
    # args.lr_drop_mag = [0.0]
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Teacher network training loop
    # model = resnet10_cifar(thin=True, thinning_ratio=6)
    # args.shard = "teacher_w_residual"
    # args.total_flops = 0
    # args.widen_times = []
    # args.deepen_times = []
    # args.lr = orig_lr
    # args.lr_drops = []
    # args.lr_drop_mag = [0.0]
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                            _validation_loss, args)

    # # Net2Net teacher
    # model = resnet10_cifar(thin=True, thinning_ratio=6, use_residual=False)
    # model = cudafy(model)
    # model.deepen([2,2,0,0], minibatch=next(iter(train_loader))[0].to('cuda'))
    # args.shard = "teacher_w_out_residual"
    # args.total_flops = 0
    # args.widen_times = []
    # args.deepen_times = []
    # args.lr = orig_lr
    # args.lr_drops = []
    # args.lr_drop_mag = [0.0]
    # args.weight_decay = 1.0e-2
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #                            _validation_loss, args)






"""
Learning rate and weight decay adaption tests
"""





def quadruple_widen_run(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # args.load = ""
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # args.deepen_times = []
    # if len(args.widen_times) != 4:
    #     raise Exception("Widening times needs to be a list of length 4 for this test")

    # # Make the data loader objects
    # train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # # R2R
    # model = resnet18(thin=True, thinning_ratio=4*4)
    # args.shard = "R2R"
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Net2Net
    # model = resnet18(thin=True, thinning_ratio=4*4, morphism_scheme="net2net")
    # args.shard = "Net2Net_student"
    # args.total_flops = 0
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)





def double_deepen_run(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # args.load = ""
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # args.widen_times = []
    # if len(args.deepen_times) != 2:
    #     raise Exception("Deepening times needs to be a list of length 2 for this test")
    # args.deepen_indidces_list = [[2,2,0,0], [2,2,0,0]]

    # # Make the data loader objects
    # train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # # R2R
    # model = resnet10(thin=True, thinning_ratio=4)
    # args.shard = "R2R"
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Net2Net
    # model = resnet10(thin=True, thinning_ratio=4, morphism_scheme="net2net")
    # args.shard = "Net2Net_student"
    # args.total_flops = 0
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)





def double_widen_and_deepen_run(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # args.load = ""
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # if len(args.widen_times) != 2:
    #     raise Exception("Widening times needs to be a list of length 2 for this test")
    # if len(args.deepen_times) != 2:
    #     raise Exception("Deepening times needs to be a list of length 2 for this test")
    # args.deepen_indidces_list = [[2,2,0,0], [2,2,0,0]]

    # # Make the data loader objects
    # train_dataset = CifarDataset(mode="train", labels_as_logits=False)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # val_dataset = CifarDataset(mode="val", labels_as_logits=False)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.workers, pin_memory=True)

    # # R2R
    # model = resnet10(thin=True, thinning_ratio=4*2)
    # args.shard = "R2R"
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)

    # # Net2Net
    # model = resnet10(thin=True, thinning_ratio=4*2, morphism_scheme="net2net")
    # args.shard = "Net2Net_student"
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)






"""
Showing that R2R == faster training tests
"""








def r2r_faster_test_part_1(args):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # if len(args.widen_times) != 2:
    #     raise Exception("Widening times needs to be a list of length 2 for this test")
    # if len(args.deepen_times) != 2:
    #     raise Exception("Deepening times needs to be a list of length 2 for this test")
    # args.deepen_indidces_list = [[1,1,1,1], [0,1,2,1]]

    # # Make the data loaders for imagenet
    # train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    # val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # # R2R
    # model = resnet26(thin=True, thinning_ratio=2)
    # args.shard = "R2R_Then_Widened"
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)








def r2r_faster_test_part_2(args):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # if len(args.widen_times) != 0:
    #     raise Exception("Widening times needs to be a list of length 0 for this test")
    # if len(args.deepen_times) != 0:
    #     raise Exception("Deepening times needs to be a list of length 0 for this test")

    # # Make the data loaders for imagenet
    # train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    # val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # # Larger model trained staight up
    # model = resnet26(thin=True, thinning_ratio=2)
    # model.deepen([1,1,1,1])
    # model.widen(1.414)
    # model.deepen([0,1,2,1])
    # model.widen(1.414)
    # model = cudafy(model)
    # args.shard = "Full_Model"
    # args.widen_times = []
    # args.deepen_times = []
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)








def r2r_faster_test_part_3(args):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # if len(args.widen_times) != 0:
    #     raise Exception("Widening times needs to be a list of length 2 for this test")
    # if len(args.deepen_times) != 0:
    #     raise Exception("Deepening times needs to be a list of length 2 for this test")
    # args.deepen_indidces_list = [[1,1,1,1], [0,1,2,1]]

    # # Make the data loaders for imagenet
    # train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    # val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # # R2R
    # model = resnet26(thin=True, thinning_ratio=1.414)
    # args.shard = "R2R_Teacher"
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)






def r2r_faster_test_part_4(args):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # if len(args.widen_times) != 1:
    #     raise Exception("Widening times needs to be a list of length 2 for this test")
    # if len(args.deepen_times) != 2:
    #     raise Exception("Deepening times needs to be a list of length 2 for this test")
    # args.deepen_indidces_list = [[1,1,1,1], [0,1,2,1]]

    # # Make the data loaders for imagenet
    # train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    # val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # # R2R
    # model = resnet26(thin=True, thinning_ratio=1.414)
    # args.shard = "R2R_One_Widen"
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)








def r2r_faster_test_redo(args, shardname, optimizer='sgd'):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # if len(args.widen_times) > 1:
    #     raise Exception("Widening times needs to be less than a list of length 2 for this test")
    # if len(args.deepen_times) > 1:
    #     raise Exception("Deepening times needs to be less than a list of length 2 for this test")
    # args.deepen_indidces_list = [[1,1,1,1]]

    # # Make the data loaders for imagenet
    # train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    # val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # # Get the optimizer function to use
    # make_optimizer_fn = _make_optimizer_fn
    # if optimizer == 'sgd':
    #     make_optimizer_fn = _make_optimizer_fn_sgd
    # elif optimizer == 'adagrad':
    #     make_optimizer_fn = _make_optimizer_fn_adagrad
    # elif optimizer == 'rms':
    #     make_optimizer_fn = _make_optimizer_fn_rms


    # # R2R
    # model = resnet10(thin=True, thinning_ratio=1.414)
    # args.shard = shardname
    # train_loop(model, train_loader, val_loader, make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)








def r2r_faster_test_redo_18(args, shardname):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # if len(args.widen_times) != 0:
    #     raise Exception("Widening times needs to be less than a list of length 2 for this test")
    # if len(args.deepen_times) != 0:
    #     raise Exception("Deepening times needs to be less than a list of length 2 for this test")

    # # Make the data loaders for imagenet
    # train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    # val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # # R2R
    # model = resnet18()
    # args.shard = shardname
    # train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op,
    #            _validation_loss, args)







def r2r_faster(args, shardname, optimizer='sgd', resnet_class=resnet35, use_thin=True, deepen_indices=[1,1,2,1], function_preserving=True):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    if len(args.widen_times) > 1:
        raise Exception("Widening times needs to be less than a list of length 2 for this test")
    if len(args.deepen_times) > 1:
        raise Exception("Deepening times needs to be less than a list of length 2 for this test")
    args.deepen_indidces_list = [deepen_indices]

    # Make the data loaders for imagenet
    train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    # train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    # val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # Get the optimizer function to use
    make_optimizer_fn = _make_optimizer_fn
    if optimizer == 'sgd':
        make_optimizer_fn = _make_optimizer_fn_sgd
    elif optimizer == 'adagrad':
        make_optimizer_fn = _make_optimizer_fn_adagrad
    elif optimizer == 'rms':
        make_optimizer_fn = _make_optimizer_fn_rms


    # R2R
    model = resnet_class(thin=use_thin, thinning_ratio=1.5)
    if not function_preserving:
        model.function_preserving = False
        model.init_scheme = 'match_std_exact'
    args.shard = shardname
    train_loop(model, train_loader, val_loader, make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)







def r2wr_imagenet(args, shardname, optimizer='sgd', resnet_class=resnet35, use_thin=False, deepen_indices=[1,1,2,1], function_preserving=True):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    if len(args.widen_times) != 2:
        raise Exception("Widening times needs to be a list of length 2 for this test")
    if len(args.deepen_times) != 0:
        raise Exception("Deepening times needs to be a list of length 0 for this test")
    args.deepen_indidces_list = []

    # Make the data loaders for imagenet
    # train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # Get the optimizer function to use
    make_optimizer_fn = _make_optimizer_fn
    if optimizer == 'sgd':
        make_optimizer_fn = _make_optimizer_fn_sgd
    elif optimizer == 'adagrad':
        make_optimizer_fn = _make_optimizer_fn_adagrad
    elif optimizer == 'rms':
        make_optimizer_fn = _make_optimizer_fn_rms


    # R2R
    model = resnet_class(thin=use_thin, thinning_ratio=1.5)
    if not function_preserving:
        model.function_preserving = False
        model.init_scheme = 'match_std_exact'
    args.shard = shardname
    train_loop(model, train_loader, val_loader, make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)


def n2wn_imagenet(args, shardname, optimizer='sgd', resnet_class=resnet35, use_thin=False, deepen_indices=[1,1,2,1], function_preserving=True):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    if len(args.widen_times) != 2:
        raise Exception("Widening times needs to be a list of length 2 for this test")
    if len(args.deepen_times) != 0:
        raise Exception("Deepening times needs to be a list of length 0 for this test")
    args.deepen_indidces_list = []

    # Make the data loaders for imagenet
    # train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # Get the optimizer function to use
    make_optimizer_fn = _make_optimizer_fn
    if optimizer == 'sgd':
        make_optimizer_fn = _make_optimizer_fn_sgd
    elif optimizer == 'adagrad':
        make_optimizer_fn = _make_optimizer_fn_adagrad
    elif optimizer == 'rms':
        make_optimizer_fn = _make_optimizer_fn_rms


    # R2R
    model = resnet_class(thin=use_thin, thinning_ratio=1.5, use_residual=False, morphism_scheme="net2net")
    if not function_preserving:
        model.function_preserving = False
        model.init_scheme = 'match_std_exact'
    args.shard = shardname
    train_loop(model, train_loader, val_loader, make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)


def n2dn_imagenet(args, shardname, optimizer='sgd', resnet_class=resnet35, use_thin=False, deepen_indices=[1,1,2,1], function_preserving=True):
    """
    This is split into multiuple parts because otherwise it will take longer than 5 days to run.
    """
    # Fix some args for the test (shouldn't ever be loading anythin)
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    if len(args.widen_times) != 0:
        raise Exception("Widening times needs to be a list of length 0 for this test")
    if len(args.deepen_times) != 2:
        raise Exception("Deepening times needs to be a list of length 2 for this test")
    args.deepen_indidces_list = [deepen_indices] * 2

    # Make the data loaders for imagenet
    # train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    train_loader = get_imagenet_dataloader("train", batch_size=args.batch_size, num_workers=args.workers)
    val_loader = get_imagenet_dataloader("val", batch_size=args.batch_size, num_workers=args.workers)

    # Get the optimizer function to use
    make_optimizer_fn = _make_optimizer_fn
    if optimizer == 'sgd':
        make_optimizer_fn = _make_optimizer_fn_sgd
    elif optimizer == 'adagrad':
        make_optimizer_fn = _make_optimizer_fn_adagrad
    elif optimizer == 'rms':
        make_optimizer_fn = _make_optimizer_fn_rms


    # R2R
    model = resnet_class(thin=use_thin, thinning_ratio=1.5, use_residual=False, morphism_scheme="net2net")
    if not function_preserving:
        model.function_preserving = False
        model.init_scheme = 'match_std_exact'
    args.shard = shardname
    train_loop(model, train_loader, val_loader, make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)







"""
last set of tests - on CIFAR10/CIFAR100/SVHN
"""




def _make_cifar_data_loaders(args):
    train_dataset = CifarDataset(train=True, labels_as_logits=False)

    val_dataset = CifarDataset(train=False, labels_as_logits=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    joint_dataset = ProductDataset(train_dataset, val_dataset)
    train_loader = DataLoader(dataset=joint_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


def _make_cifar100_data_loaders(args):
    train_dataset = Cifar100Dataset(train=True, labels_as_logits=False)

    val_dataset = Cifar100Dataset(train=False, labels_as_logits=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    joint_dataset = ProductDataset(train_dataset, val_dataset)
    train_loader = DataLoader(dataset=joint_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


def _make_svhn_data_loaders(args, extended=False):
    train_dataset = SvhnDataset(train=True, labels_as_logits=False, use_extra_train=extended)

    val_dataset = SvhnDataset(train=False, labels_as_logits=False, use_extra_train=extended)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    joint_dataset = ProductDataset(train_dataset, val_dataset)
    train_loader = DataLoader(dataset=joint_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader






def last_cifar_r2wider_resnet_thin(args):
    raise Exception("TODO: set params for lrs/lr_drops/weight_decay/weight_decay_adjustments")
    var_args = {
        "r2r": (True, 5.0, 3.0e-3),
        "n2n": (True, 5.0, 3.0e-3),
        "rand_pad": (True, 5.0, 3.0e-3),
        "netmorph": (True, 5.0, 3.0e-3),
        "rand_init": (True, 5.0, 3.0e-3),
        "rand_init_n2n": (True, 5.0, 3.0e-3),
        "teacher": (True, 5.0, 3.0e-3),
        "teacher_n2n": (True, 5.0, 3.0e-3),
    }
    # Make the data loader objects
    train_loader, val_loader = _make_cifar_data_loaders(args)
    _last_r2wider_resnet(args, train_loader, val_loader, tr=4, var_args=var_args)


def last_cifar_r2wider_resnet_wide(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_cifar_data_loaders(args)
    # _last_r2wider_resnet(args, train_loader, val_loader, tr=1.5)


def last_svhn_r2wider_resnet_thin(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_svhn_data_loaders(args)
    # _last_r2wider_resnet(args, train_loader, val_loader, tr=4)


def last_svhn_r2wider_resnet_wide(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_svhn_data_loaders(args)
    # _last_r2wider_resnet(args, train_loader, val_loader, tr=1.5)


def last_svhn_extended_r2wider_resnet_thin(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    # _last_r2wider_resnet(args, train_loader, val_loader, tr=4)


def last_svhn_extended_r2wider_resnet_wide(args):
    var_args = {
        "r2r": (True, 10.0, 1.0e-5),
        "n2n": (False, 3.0, 0.0),
        "rand_pad": (True, 10.0, 1.0e-3),
        "netmorph": (True, 10.0, 1.0e-5),
        "rand_init": (False, 3.0, 1.0e-5),
        "rand_init_n2n": (False, 3.0, 0.0),
        "teacher": (True, 1.0, 1.0e-4),
        "teacher_n2n": (True, 1.0, 1.0e-4),
    }
    # Make the data loader objects
    train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    _last_r2wider_resnet(args, train_loader, val_loader, tr=1.5, var_args=var_args)







def last_cifar_r2deeper_resnet_thin(args):
    raise Exception("TODO: set params for lrs/lr_drops/weight_decay/weight_decay_adjustments")
    var_args = {
        "r2r": (True, 5.0, 3.0e-3),
        "n2n": (True, 5.0, 3.0e-3),
        "rand_pad": (True, 5.0, 3.0e-3),
        "rand_init": (True, 5.0, 3.0e-3),
        "teacher": (True, 5.0, 3.0e-3),
        "teacher_n2n": (True, 5.0, 3.0e-3),
    }
    # Make the data loader objects
    train_loader, val_loader = _make_cifar_data_loaders(args)
    _last_r2deeper_resnet(args, train_loader, val_loader, tr=2.7, var_args=var_args)


def last_cifar_r2deeper_resnet_wide(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_cifar_data_loaders(args)
    # _last_r2deeper_resnet(args, train_loader, val_loader, tr=1)


def last_svhn_r2deeper_resnet_thin(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_svhn_data_loaders(args)
    # _last_r2deeper_resnet(args, train_loader, val_loader, tr=2.7)


def last_svhn_r2deeper_resnet_wide(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_svhn_data_loaders(args)
    # _last_r2deeper_resnet(args, train_loader, val_loader, tr=1)


def last_svhn_extended_r2deeper_resnet_thin(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    # _last_r2deeper_resnet(args, train_loader, val_loader, tr=2.7)


def last_svhn_extended_r2deeper_resnet_wide(args):
    var_args = {
        "r2r": (True, 10.0, 1.0e-5),
        "n2n": (True, 10.0, 1.0e-5),
        "rand_pad": (True, 10.0, 1.0e-5),
        "rand_init": (True, 3.0, 0.0),
        "teacher": (True, 3.0, 1.0e-4),
        "teacher_n2n": (True, 3.0, 1.0e-4),
    }
    # Make the data loader objects
    train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    _last_r2deeper_resnet(args, train_loader, val_loader, tr=1, var_args=var_args)







def last_cifar_net2wider_resnet_thin(args):
    raise Exception("TODO: set params for lrs/lr_drops/weight_decay/weight_decay_adjustments")
    var_args = {
        "r2r": (True, 5.0, 3.0e-3),
        "n2n": (True, 5.0, 3.0e-3),
        "rand_pad": (True, 5.0, 3.0e-3),
        "netmorph": (True, 5.0, 3.0e-3),
        "rand_init": (True, 5.0, 3.0e-3),
        "rand_init_n2n": (True, 5.0, 3.0e-3),
    }
    # Make the data loader objects
    train_loader, val_loader = _make_cifar_data_loaders(args)
    _last_net2wider_resnet(args, train_loader, val_loader, tr=4, var_args=var_args)


def last_cifar_net2wider_resnet_wide(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_cifar_data_loaders(args)
    # _last_net2wider_resnet(args, train_loader, val_loader, tr=1.5)


def last_svhn_net2wider_resnet_thin(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_svhn_data_loaders(args)
    # _last_net2wider_resnet(args, train_loader, val_loader, tr=4)


def last_svhn_net2wider_resnet_wide(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_svhn_data_loaders(args)
    # _last_net2wider_resnet(args, train_loader, val_loader, tr=1.5)


def last_svhn_extended_net2wider_resnet_thin(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    # _last_net2wider_resnet(args, train_loader, val_loader, tr=4)


def last_svhn_extended_net2wider_resnet_wide(args):
    var_args = {
        "r2r": (True, 1.0, None),# ignore weight decay, as adapting here
        "n2n": (True, 1.0, None),# ignore weight decay, as adapting here
        "rand_pad": (True, 1.0, None),
        "netmorph": (True, 1.0, None), # ignore weight decay, as adapting here
        "rand_init": (True, 1.0, None),
        "rand_init_n2n": (True, 1.0, None),
    }
    # Make the data loader objects
    train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    _last_net2wider_resnet(args, train_loader, val_loader, tr=1.5, var_args=var_args)







def last_cifar_net2deeper_resnet_thin(args):
    raise Exception("TODO: set params for lrs/lr_drops/weight_decay/weight_decay_adjustments")
    var_args = {
        "r2r": (True, 1.0, None),
        "n2n": (True, 1.0, None),
        "rand_pad": (True, 1.0, None),
        "rand_init": (True, 1.0, None),
        "rand_init_n2n": (True, 1.0, None),
    }
    # Make the data loader objects
    train_loader, val_loader = _make_cifar_data_loaders(args)
    _last_net2deeper_resnet(args, train_loader, val_loader, tr=2.7, var_args=var_args)


def last_cifar_net2deeper_resnet_wide(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_cifar_data_loaders(args)
    # _last_net2deeper_resnet(args, train_loader, val_loader, tr=1)


def last_svhn_net2deeper_resnet_thin(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_svhn_data_loaders(args)
    # _last_net2deeper_resnet(args, train_loader, val_loader, tr=2.7)


def last_svhn_net2deeper_resnet_wide(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_svhn_data_loaders(args)
    # _last_net2deeper_resnet(args, train_loader, val_loader, tr=1)


def last_svhn_extended_net2deeper_resnet_thin(args):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Make the data loader objects
    # train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    # _last_net2deeper_resnet(args, train_loader, val_loader, tr=2.7)


def last_svhn_extended_net2deeper_resnet_wide(args):
    var_args = {
        "r2r": (True, 1.0, None),# ignore weight decay, as adapting here
        "n2n": (True, 1.0, None),# ignore weight decay, as adapting here
        "rand_pad": (True, 1.0, None),
        "rand_init": (True, 1.0, None),
        "rand_init_n2n": (True, 1.0, None),
    }
    # Make the data loader objects
    train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    _last_net2deeper_resnet(args, train_loader, val_loader, tr=1, var_args=var_args)







def last_cifar100_r2wider_resnet_wide(args):
    raise Exception("TODO: set params for lrs/lr_drops/weight_decay/weight_decay_adjustments")
    var_args = {
        "r2r": (True, 5.0, 3.0e-3),
        "n2n": (True, 5.0, 3.0e-3),
        "rand_pad": (True, 5.0, 3.0e-3),
        "netmorph": (True, 5.0, 3.0e-3),
        "rand_init": (True, 5.0, 3.0e-3),
        "rand_init_n2n": (True, 5.0, 3.0e-3),
        "teacher": (True, 5.0, 3.0e-3),
        "teacher_n2n": (True, 5.0, 3.0e-3),
    }
    # Make the data loader objects
    train_loader, val_loader = _make_cifar100_data_loaders(args)
    _last_r2wider_resnet(args, train_loader, val_loader, tr=1.5, var_args=var_args, nc=100)


def last_cifar100_r2deeper_resnet_wide(args):
    raise Exception("TODO: set params for lrs/lr_drops/weight_decay/weight_decay_adjustments")
    var_args = {
        "r2r": (True, 5.0, 3.0e-3),
        "n2n": (True, 5.0, 3.0e-3),
        "rand_pad": (True, 5.0, 3.0e-3),
        "rand_init": (True, 5.0, 3.0e-3),
        "teacher": (True, 5.0, 3.0e-3),
        "teacher_n2n": (True, 5.0, 3.0e-3),
    }
    # Make the data loader objects
    train_loader, val_loader = _make_cifar100_data_loaders(args)
    _last_r2deeper_resnet(args, train_loader, val_loader, tr=1, var_args=var_args, nc=100)


def last_cifar100_net2wider_resnet_wide(args):
    raise Exception("TODO: set params for lrs/lr_drops/weight_decay/weight_decay_adjustments")
    var_args = {
        "r2r": (True, 5.0, 3.0e-3),
        "n2n": (True, 5.0, 3.0e-3),
        "rand_pad": (True, 5.0, 3.0e-3),
        "netmorph": (True, 5.0, 3.0e-3),
        "rand_init": (True, 5.0, 3.0e-3),
        "rand_init_n2n": (True, 5.0, 3.0e-3),
    }
    # Make the data loader objects
    train_loader, val_loader = _make_cifar100_data_loaders(args)
    _last_net2wider_resnet(args, train_loader, val_loader, tr=1.5, var_args=var_args, nc=100)


def last_cifar100_net2deeper_resnet_wide(args):
    raise Exception("TODO: set params for lrs/lr_drops/weight_decay/weight_decay_adjustments")
    var_args = {
        "r2r": (True, 5.0, 3.0e-3),
        "n2n": (True, 5.0, 3.0e-3),
        "rand_pad": (True, 5.0, 3.0e-3),
        "rand_init": (True, 5.0, 3.0e-3),
        "rand_init_n2n": (True, 5.0, 3.0e-3),
    }
    # Make the data loader objects
    train_loader, val_loader = _make_cifar100_data_loaders(args)
    _last_net2deeper_resnet(args, train_loader, val_loader, tr=1, var_args=var_args, nc=100)











def last_cifar_r2wr_hp_search(args):
    # Make the data loader objects
    train_loader, val_loader = _make_cifar_data_loaders(args)
    _last_hyper_param_tune_r_2_wider_r(args, train_loader, val_loader, tr=4)

def last_cifar_r2dr_hp_search(args):
    # Make the data loader objects
    train_loader, val_loader = _make_cifar_data_loaders(args)
    _last_hyper_param_tune_r_2_deeper_r(args, train_loader, val_loader, tr=2.7)

def last_cifar_n2wn_hp_search(args):
    # Make the data loader objects
    train_loader, val_loader = _make_cifar_data_loaders(args)
    _last_hyper_param_tune_net_2_wider_net(args, train_loader, val_loader, tr=4)

def last_cifar_n2dn_hp_search(args):
    # Make the data loader objects
    train_loader, val_loader = _make_cifar_data_loaders(args)
    _last_hyper_param_tune_net_2_deeper_net(args, train_loader, val_loader, tr=2.7)



def last_extsvhn_r2wr_hp_search(args):
    # Make the data loader objects
    train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    _last_hyper_param_tune_r_2_wider_r(args, train_loader, val_loader, tr=1.5)

def last_extsvhn_r2dr_hp_search(args):
    # Make the data loader objects
    train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    _last_hyper_param_tune_r_2_deeper_r(args, train_loader, val_loader, tr=1.0)

def last_extsvhn_n2wn_hp_search(args):
    # Make the data loader objects
    train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    _last_hyper_param_tune_net_2_wider_net(args, train_loader, val_loader, tr=1.5)

def last_extsvhn_n2dn_hp_search(args):
    # Make the data loader objects
    train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    _last_hyper_param_tune_net_2_deeper_net(args, train_loader, val_loader, tr=1.0)



def last_cifar100_r2wr_hp_search(args):
    # Make the data loader objects
    train_loader, val_loader = _make_cifar100_data_loaders(args)
    _last_hyper_param_tune_r_2_wider_r(args, train_loader, val_loader, tr=1.5)

def last_cifar100_r2dr_hp_search(args):
    # Make the data loader objects
    train_loader, val_loader = _make_cifar100_data_loaders(args)
    _last_hyper_param_tune_r_2_deeper_r(args, train_loader, val_loader, tr=1.0)

def last_cifar100_n2wn_hp_search(args):
    # Make the data loader objects
    train_loader, val_loader = _make_cifar100_data_loaders(args)
    _last_hyper_param_tune_net_2_wider_net(args, train_loader, val_loader, tr=1.5)

def last_cifar100_n2dn_hp_search(args):
    # Make the data loader objects
    train_loader, val_loader = _make_cifar100_data_loaders(args)
    _last_hyper_param_tune_net_2_deeper_net(args, train_loader, val_loader, tr=1.0)







def last_svhn_weight_decay_tune(args):
    # Make the data loader objects
    train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)
    _last_weight_decay_tune(args, train_loader, val_loader)






#
#
# TODO: add orig_wd and rest args.weight_decay and args.adjust_weight_decay in tests below
# TODO: add a way to specify in the above functions different defaults for weight_decay/lr_drop/using weight decay matching
#
#


def _last_r2wider_resnet(args, train_loader, val_loader, tr, var_args, nc=10):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    orig_lr = args.lr

    # R2R
    model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
    args.shard = "R2R_student"
    args.total_flops = 0
    args.lr = orig_lr
    args.lr_drops = args.widen_times
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["r2r"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr_drop_mag = [lr_drop_mag]
    args.weight_decay = weight_decay
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)

    # Net2Net
    model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, morphism_scheme="net2net", num_classes=nc)
    args.shard = "Net2Net_student"
    args.total_flops = 0
    args.lr = orig_lr
    args.lr_drops = args.widen_times
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["n2n"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr_drop_mag = [lr_drop_mag]
    args.weight_decay = weight_decay
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)

    # RandomPadding
    model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, function_preserving=False, num_classes=nc)
    model.init_scheme = 'He'
    args.shard = "RandomPadding_student"
    args.total_flops = 0
    args.lr = orig_lr
    args.lr_drops = args.widen_times
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["rand_pad"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr_drop_mag = [lr_drop_mag]
    args.weight_decay = weight_decay
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)

    # # NetMorph
    model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, morphism_scheme="netmorph", num_classes=nc)
    args.shard = "NetMorph_student"
    args.total_flops = 0
    args.lr = orig_lr
    args.lr_drops = args.widen_times
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["netmorph"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr_drop_mag = [lr_drop_mag]
    args.weight_decay = weight_decay
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)

    # Random init start
    model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
    model.widen(1.5)
    args.shard = "Completely_Random_Init"
    args.total_flops = 0
    args.widen_times = []
    args.deepen_times = []
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["rand_init"]
    args.adjust_weight_decay = adjust_weight_decay
    args.weight_decay = weight_decay
    args.lr = orig_lr / lr_drop_mag
    args.lr_drops = []
    args.lr_drop_mag = [0.0]
    # args.weight_decay = 5.0e-3
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)

    # Random init start v2
    model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, use_residual=False, num_classes=nc)
    model.widen(1.5)
    args.shard = "Completely_Random_Init_Net2Net"
    args.total_flops = 0
    args.widen_times = []
    args.deepen_times = []
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["rand_init_n2n"]
    args.adjust_weight_decay = adjust_weight_decay
    args.weight_decay = weight_decay
    args.lr = orig_lr / lr_drop_mag
    args.lr_drops = []
    args.lr_drop_mag = [0.0]
    # args.weight_decay = 5.0e-3
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)

    # Teacher network training loop
    model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
    args.shard = "teacher_w_residual"
    args.total_flops = 0
    args.widen_times = []
    args.deepen_times = []
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["teacher"]
    args.adjust_weight_decay = adjust_weight_decay
    args.weight_decay = weight_decay
    args.lr = orig_lr / lr_drop_mag
    args.lr_drops = []
    args.lr_drop_mag = 0.0
    # args.weight_decay = 1.0e-4 #1.0e-2
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                               _validation_loss, args)

    # Net2Net teacher
    model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, use_residual=False, num_classes=nc)
    args.shard = "teacher_w_out_residual"
    args.total_flops = 0
    args.widen_times = []
    args.deepen_times = []
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["teacher_n2n"]
    args.adjust_weight_decay = adjust_weight_decay
    args.weight_decay = weight_decay
    args.lr = orig_lr / lr_drop_mag
    args.lr_drops = []
    args.lr_drop_mag = [0.0]
    # args.weight_decay = 1.0e-4 #1.0e-2
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                               _validation_loss, args)






def _last_r2deeper_resnet(args, train_loader, val_loader, tr, var_args, nc=10):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    orig_lr = args.lr

    # R2R
    model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
    args.deepen_indidces_list = [[1,1,1,0]]
    args.shard = "R2R_student"
    args.total_flops = 0
    args.lr = orig_lr
    args.lr_drops = args.deepen_times
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["r2r"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr_drop_mag = [lr_drop_mag]
    args.weight_decay = weight_decay
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)

    # Net2Net
    model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, use_residual=False, morphism_scheme="net2net", num_classes=nc)
    args.deepen_indidces_list = [[1,1,1,0]]
    args.shard = "Net2Net_student"
    args.total_flops = 0
    args.lr = orig_lr
    args.lr_drops = args.deepen_times
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["n2n"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr_drop_mag = [lr_drop_mag]
    args.weight_decay = weight_decay
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)

    # RandomPadding
    model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, function_preserving=False, num_classes=nc)
    model.init_scheme = 'He'
    args.deepen_indidces_list = [[1,1,1,0]]
    args.shard = "RandomPadding_student"
    args.total_flops = 0
    args.lr = orig_lr
    args.lr_drops = args.deepen_times
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["rand_pad"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr_drop_mag = [lr_drop_mag]
    args.weight_decay = weight_decay
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)

    # Random init start
    model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
    model.deepen([1,1,1,0])
    args.shard = "Completely_Random_Init"
    args.total_flops = 0
    args.widen_times = []
    args.deepen_times = []
    args.lr_drops = []
    args.lr_drop_mag = [0.0]
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["rand_init"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = orig_lr / lr_drop_mag
    args.weight_decay = weight_decay
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)

    # Teacher network training loop
    model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
    args.shard = "teacher_w_residual"
    args.total_flops = 0
    args.widen_times = []
    args.deepen_times = []
    args.lr_drops = []
    args.lr_drop_mag = [0.0]
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["teacher"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = orig_lr / lr_drop_mag
    args.weight_decay = weight_decay
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                               _validation_loss, args)

    # Net2Net teacher
    model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, use_residual=False, num_classes=nc)
    model = cudafy(model)
    model.deepen([1,1,1,0], minibatch=next(iter(train_loader))[0].to('cuda'))
    args.shard = "teacher_w_out_residual"
    args.total_flops = 0
    args.widen_times = []
    args.deepen_times = []
    args.lr_drops = []
    args.lr_drop_mag = [0.0]
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["teacher_n2n"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = orig_lr / lr_drop_mag
    args.weight_decay = weight_decay
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                               _validation_loss, args)





def _last_net2wider_resnet(args, train_loader, val_loader, tr, var_args, nc=10):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    args.widen_times = []
    args.deepen_times = []

    orig_lr = args.lr
    orig_lr_drops = args.lr_drops
    orig_lr_drop_mags = args.lr_drop_mag
    orig_wd = args.weight_decay
    scaling_factor = 1.5

    # Teacher network training loop
    args.shard = "teacher_w_residual"
    args.total_flops = 0
    args.lr = orig_lr
    args.weight_decay = orig_wd
    initial_model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
    teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                               _validation_loss, args)
    weight_before = parameter_magnitude(teacher_model)

    # initialize student networks lr w.r.t. end lr of teacher
    end_lr = args.lr

    # R2R
    model = copy.deepcopy(teacher_model)
    model.widen(scaling_factor)
    weight_after = parameter_magnitude(model)
    args.shard = "R2R_student"
    args.total_flops = 0
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["r2r"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = end_lr / lr_drop_mag
    args.weight_decay = orig_wd / weight_after * weight_before if adjust_weight_decay else weight_decay
    args.lr_drops = []
    args.lr_drop_mag = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)


    # # NetMorph
    model = copy.deepcopy(teacher_model)
    model.morphism_scheme="netmorph"
    model.widen(scaling_factor)
    weight_after = parameter_magnitude(model)
    args.shard = "NetMorph_student"
    args.total_flops = 0
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["netmorph"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = end_lr / lr_drop_mag
    args.weight_decay = orig_wd / weight_after * weight_before if adjust_weight_decay else weight_decay
    args.lr_drops = []
    args.lr_drop_mag = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)


    # RandomPadding
    model = copy.deepcopy(teacher_model)
    model.function_preserving = False
    model.init_scheme = 'He'
    model.widen(scaling_factor)
    weight_after = parameter_magnitude(model)
    args.shard = "RandomPadding_student"
    args.total_flops = 0
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["rand_pad"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = end_lr / lr_drop_mag
    args.weight_decay = orig_wd / weight_after * weight_before if adjust_weight_decay else weight_decay
    args.lr_drops = []
    args.lr_drop_mag = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)


    # Random init start
    model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
    model.widen(scaling_factor)
    args.shard = "Completely_Random_Init"
    args.total_flops = 0
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["rand_init"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = end_lr / lr_drop_mag
    args.weight_decay = weight_decay
    args.lr_drops = orig_lr_drops
    args.lr_drop_mag = orig_lr_drop_mags
    train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)


    # Net2Net teacher
    initial_model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, use_residual=False, morphism_scheme="net2net", num_classes=nc)
    args.shard = "teacher_w_out_residual"
    args.total_flops = 0
    args.lr = orig_lr
    args.weight_decay = orig_wd 
    args.lr_drops = orig_lr_drops
    args.lr_drop_mag = orig_lr_drop_mags
    teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, 
                               _update_op_cts_eval, _validation_loss, args)
    weight_before = parameter_magnitude(teacher_model)

    # Net2Net
    model = copy.deepcopy(teacher_model)
    model.widen(scaling_factor)
    weight_after = parameter_magnitude(model)
    args.shard = "Net2Net_student"
    args.total_flops = 0
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["n2n"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = end_lr / lr_drop_mag
    args.weight_decay = orig_wd / weight_after * weight_before if adjust_weight_decay else weight_decay
    args.lr_drops = []
    args.lr_drop_mag = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)


    # Random init start v2
    model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, use_residual=False, num_classes=nc)
    model.widen(scaling_factor)
    args.shard = "Completely_Random_Init_Net2Net"
    args.total_flops = 0
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["rand_init_n2n"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = orig_lr / lr_drop_mag
    args.weight_decay = weight_decay
    args.lr_drops = orig_lr_drops
    args.lr_drop_mag = orig_lr_drop_mags
    train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)




def _last_net2deeper_resnet(args, train_loader, val_loader, tr, var_args, nc=10):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    args.widen_times = []
    args.deepen_times = []

    orig_lr = args.lr
    orig_lr_drops = args.lr_drops
    orig_lr_drop_mags = args.lr_drop_mag
    orig_wd = args.weight_decay

    # Teacher network training loop
    args.shard = "teacher_w_residual"
    args.total_flops = 0
    args.lr = orig_lr
    args.weight_decay = orig_wd 
    # args.weight_decay = 5.0e-3
    initial_model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
    teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn,
                               _update_op_cts_eval, _validation_loss, args)
    weight_before = parameter_magnitude(teacher_model)

    # R2R
    model = copy.deepcopy(teacher_model)
    model.deepen([1,1,1,0])
    weight_after = parameter_magnitude(model)
    model = cudafy(model)
    args.shard = "R2R_student"
    args.total_flops = 0
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["r2r"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = orig_lr / lr_drop_mag
    args.weight_decay = orig_wd / weight_after * weight_before if adjust_weight_decay else weight_decay
    args.lr_drops = []
    args.lr_drop_mag = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)

    # RandomPadding
    model = copy.deepcopy(teacher_model)
    model.init_scheme = 'He'
    model.function_preserving = False
    model.deepen([1,1,1,0])
    weight_after = parameter_magnitude(model)
    model = cudafy(model)
    args.shard = "RandomPadding_student"
    args.total_flops = 0
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["rand_pad"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = orig_lr / lr_drop_mag
    args.weight_decay = orig_wd / weight_after * weight_before if adjust_weight_decay else weight_decay
    args.lr_drops = []
    args.lr_drop_mag = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)

    # Random init start
    model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
    model.deepen([1,1,1,0])
    model = cudafy(model)
    args.shard = "Completely_Random_Init"
    args.total_flops = 0
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["rand_init"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = orig_lr / lr_drop_mag
    args.weight_decay = weight_decay
    args.lr_drops = orig_lr_drops
    args.lr_drop_mag = orig_lr_drop_mags
    train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)

    # Net2Net teacher
    initial_model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, use_residual=False, morphism_scheme="net2net", num_classes=nc)
    args.shard = "teacher_w_out_residual"
    args.total_flops = 0
    args.lr = orig_lr
    args.weight_decay = orig_wd 
    args.lr_drops = orig_lr_drops
    args.lr_drop_mag = orig_lr_drop_mags
    teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn,
                               _update_op_cts_eval, _validation_loss, args)
    weight_before = parameter_magnitude(teacher_model)

    # Net2Net
    model = copy.deepcopy(teacher_model)
    model = cudafy(model)
    model.deepen([1,1,1,0], minibatch=next(iter(train_loader))[0].to('cuda'))
    weight_after = parameter_magnitude(model)
    model = cudafy(model)
    args.shard = "Net2Net_student"
    args.total_flops = 0
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["n2n"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = orig_lr / lr_drop_mag
    args.weight_decay = orig_wd / weight_after * weight_before if adjust_weight_decay else weight_decay
    args.lr_drops = []
    args.lr_drop_mag = []
    train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)


    # Random init start v2
    model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, use_residual=False, num_classes=nc)
    model.deepen([1,1,1,0])
    args.shard = "Completely_Random_Init_Net2Net"
    args.total_flops = 0
    adjust_weight_decay, lr_drop_mag, weight_decay = var_args["rand_init_n2n"]
    args.adjust_weight_decay = adjust_weight_decay
    args.lr = orig_lr / lr_drop_mag
    args.weight_decay = weight_decay
    args.lr_drops = orig_lr_drops
    args.lr_drop_mag = orig_lr_drop_mags
    train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op_cts_eval,
               _validation_loss, args)






def _last_hyper_param_tune_r_2_wider_r(args, train_loader, val_loader, tr, nc=10):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    orig_lr = args.lr

    for adapt_wd in [True, False]:
        for lr_drop in [1.0, 3.0, 10.0]:
            for weight_decay in [0.0, 1.0e-5, 1.0e-3, 3.0e-3, 1.0e-2]:

                # R2R
                model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
                args.shard = "R2R_student_a={a}_l={l}_w={w}".format(a=adapt_wd, l=lr_drop, w=weight_decay)
                args.total_flops = 0
                args.lr = orig_lr
                args.lr_drops = args.widen_times
                args.lr_drop_mag = [lr_drop]
                args.weight_decay = weight_decay
                args.adjust_weight_decay = adapt_wd
                # args.weight_decay = 1.0e-4 #1.0e-2
                train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                        _validation_loss, args)

                # Net2Net
                model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, morphism_scheme="net2net", num_classes=nc)
                args.shard = "Net2Net_student_a={a}_l={l}_w={w}".format(a=adapt_wd, l=lr_drop, w=weight_decay)
                args.total_flops = 0
                args.lr = orig_lr
                args.lr_drops = args.widen_times
                args.lr_drop_mag = [lr_drop]
                args.weight_decay = weight_decay
                args.adjust_weight_decay = adapt_wd
                train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                        _validation_loss, args)

                # RandomPadding
                model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, function_preserving=False, num_classes=nc)
                model.init_scheme = 'He'
                args.shard = "RandomPadding_student_a={a}_l={l}_w={w}".format(a=adapt_wd, l=lr_drop, w=weight_decay)
                args.total_flops = 0
                args.lr = orig_lr
                args.lr_drops = args.widen_times
                args.lr_drop_mag = [lr_drop]
                args.weight_decay = weight_decay
                args.adjust_weight_decay = adapt_wd
                train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                        _validation_loss, args)

                # # NetMorph
                model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, morphism_scheme="netmorph", num_classes=nc)
                args.shard = "NetMorph_student_a={a}_l={l}_w={w}".format(a=adapt_wd, l=lr_drop, w=weight_decay)
                args.total_flops = 0
                args.lr = orig_lr
                args.lr_drops = args.widen_times
                args.lr_drop_mag = [lr_drop]
                args.weight_decay = weight_decay
                args.adjust_weight_decay = adapt_wd
                train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                        _validation_loss, args)




def _last_hyper_param_tune_r_2_deeper_r(args, train_loader, val_loader, tr, nc=10):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget

    orig_lr = args.lr

    for adapt_wd in [True, False]:
        for lr_drop in [1.0, 3.0, 10.0]:
            for weight_decay in [0.0, 1.0e-5, 1.0e-3, 3.0e-3, 1.0e-2]:

                # R2R
                model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
                args.deepen_indidces_list = [[1,1,1,0]]
                args.shard = "R2R_student_a={a}_l={l}_w={w}".format(a=adapt_wd, l=lr_drop, w=weight_decay)
                args.total_flops = 0
                args.lr = orig_lr
                args.lr_drops = args.deepen_times
                args.lr_drop_mag = [lr_drop]
                args.weight_decay = weight_decay
                args.adjust_weight_decay = adapt_wd
                train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                        _validation_loss, args)

                # Net2Net
                model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, use_residual=False, morphism_scheme="net2net", num_classes=nc)
                args.deepen_indidces_list = [[1,1,1,0]]
                args.shard = "Net2Net_student_a={a}_l={l}_w={w}".format(a=adapt_wd, l=lr_drop, w=weight_decay)
                args.total_flops = 0
                args.lr = orig_lr
                args.lr_drops = args.deepen_times
                args.lr_drop_mag = [lr_drop]
                args.weight_decay = weight_decay
                args.adjust_weight_decay = adapt_wd
                train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                        _validation_loss, args)

                # RandomPadding
                model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, function_preserving=False, num_classes=nc)
                model.init_scheme = 'He'
                args.deepen_indidces_list = [[1,1,1,0]]
                args.shard = "RandomPadding_student_a={a}_l={l}_w={w}".format(a=adapt_wd, l=lr_drop, w=weight_decay)
                args.total_flops = 0
                args.lr = orig_lr
                args.lr_drops = args.deepen_times
                args.lr_drop_mag = [lr_drop]
                args.weight_decay = weight_decay
                args.adjust_weight_decay = adapt_wd
                train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                        _validation_loss, args)




def _last_hyper_param_tune_net_2_wider_net(args, train_loader, val_loader, tr, nc=10):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    args.widen_times = []
    args.deepen_times = []

    orig_lr = args.lr
    orig_wd = args.weight_decay
    scaling_factor = 1.5

    # Teacher network training loop
    args.shard = "teacher_w_residual"
    args.total_flops = 0
    args.lr = orig_lr
    args.weight_decay = orig_wd
    args.adjust_weight_decay = False
    initial_model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
    teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                               _validation_loss, args)

    # Net2Net teacher
    initial_model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, use_residual=False, morphism_scheme="net2net", num_classes=nc)
    args.shard = "teacher_w_out_residual"
    args.total_flops = 0
    args.lr = orig_lr
    args.weight_decay = orig_wd 
    args.adjust_weight_decay = False
    n2nteacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, 
                               _update_op_cts_eval, _validation_loss, args)

    for lr_drop in [1.0, 3.0, 10.0]:
        for weight_decay in [0.0, 1.0e-5, 1.0e-3, 3.0e-3, 1.0e-2, 100]:

            # R2R
            weight_before = parameter_magnitude(teacher_model)
            model = copy.deepcopy(teacher_model)
            model.widen(scaling_factor)
            weight_after = parameter_magnitude(model)
            args.shard = "R2R_student_l={l}_w={w}".format(l=lr_drop, w=weight_decay)
            args.total_flops = 0
            args.lr = orig_lr / lr_drop 
            args.weight_decay = weight_decay if weight_decay != 100 else orig_wd / weight_after * weight_before
            args.adjust_weight_decay = False
            train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                    _validation_loss, args)


            # # NetMorph
            weight_before = parameter_magnitude(teacher_model)
            model = copy.deepcopy(teacher_model)
            model.morphism_scheme="netmorph"
            model.widen(scaling_factor)
            weight_after = parameter_magnitude(model)
            args.shard = "NetMorph_student_l={l}_w={w}".format(l=lr_drop, w=weight_decay)
            args.total_flops = 0
            args.lr = orig_lr / lr_drop 
            args.weight_decay = weight_decay if weight_decay != 100 else orig_wd / weight_after * weight_before
            args.adjust_weight_decay = False
            train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                    _validation_loss, args)


            # RandomPadding
            weight_before = parameter_magnitude(teacher_model)
            model = copy.deepcopy(teacher_model)
            model.function_preserving = False
            model.init_scheme = 'He'
            model.widen(scaling_factor)
            weight_after = parameter_magnitude(model)
            args.shard = "RandomPadding_student_l={l}_w={w}".format(l=lr_drop, w=weight_decay)
            args.total_flops = 0
            args.lr = orig_lr / lr_drop 
            args.weight_decay = weight_decay if weight_decay != 100 else orig_wd / weight_after * weight_before
            args.adjust_weight_decay = False
            train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                    _validation_loss, args)


            # Random init start
            if weight_decay != 100:
                model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
                model.widen(scaling_factor)
                args.shard = "Completely_Random_Init_l={l}_w={w}".format(l=lr_drop, w=weight_decay)
                args.total_flops = 0
                args.lr = orig_lr / lr_drop 
                args.weight_decay = weight_decay 
                args.adjust_weight_decay = False
                train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                        _validation_loss, args)

            # Net2Net
            weight_before = parameter_magnitude(n2nteacher_model)
            model = copy.deepcopy(n2nteacher_model)
            model.widen(scaling_factor)
            weight_after = parameter_magnitude(model)
            args.shard = "Net2Net_student_l={l}_w={w}".format(l=lr_drop, w=weight_decay)
            args.total_flops = 0
            args.lr = orig_lr / lr_drop 
            args.weight_decay = weight_decay if weight_decay != 100 else orig_wd / weight_after * weight_before
            args.adjust_weight_decay = False
            train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                    _validation_loss, args)


            # Random init start v2
            if weight_decay != 100:
                model = orig_resnet18_cifar(thin=True, thinning_ratio=tr, use_residual=False, num_classes=nc)
                model.widen(scaling_factor)
                args.shard = "Completely_Random_Init_Net2Net_l={l}_w={w}".format(l=lr_drop, w=weight_decay)
                args.total_flops = 0
                args.lr = orig_lr / lr_drop 
                args.weight_decay = weight_decay 
                args.adjust_weight_decay = False
                train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                        _validation_loss, args)






def _last_hyper_param_tune_net_2_deeper_net(args, train_loader, val_loader, tr, nc=10):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    args.widen_times = []
    args.deepen_times = []

    orig_lr = args.lr
    orig_wd = args.weight_decay

    # Teacher network training loop
    args.shard = "teacher_w_residual"
    args.total_flops = 0
    args.lr = orig_lr
    args.weight_decay = orig_wd 
    initial_model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
    teacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn,
                               _update_op_cts_eval, _validation_loss, args)

    # Net2Net teacher
    initial_model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, use_residual=False, morphism_scheme="net2net", num_classes=nc)
    args.shard = "teacher_w_out_residual"
    args.total_flops = 0
    args.lr = orig_lr
    args.weight_decay = orig_wd 
    # args.weight_decay = 5.0e-3
    n2nteacher_model = train_loop(initial_model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn,
                               _update_op_cts_eval, _validation_loss, args)

    for lr_drop in [1.0, 3.0, 10.0]:
        for weight_decay in [0.0, 1.0e-5, 1.0e-3, 3.0e-3, 1.0e-2, 100]:

            # R2R
            weight_before = parameter_magnitude(teacher_model)
            model = copy.deepcopy(teacher_model)
            model.deepen([1,1,1,0])
            model = cudafy(model)
            weight_after = parameter_magnitude(model)
            args.shard = "R2R_student_l={l}_w={w}".format(l=lr_drop, w=weight_decay)
            args.total_flops = 0
            args.lr = orig_lr / lr_drop
            args.weight_decay = weight_decay if weight_decay != 100 else orig_wd / weight_after * weight_before
            args.adjust_weight_decay = False
            train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                    _validation_loss, args)

            # RandomPadding
            weight_before = parameter_magnitude(teacher_model)
            model = copy.deepcopy(teacher_model)
            model.init_scheme = 'He'
            model.function_preserving = False
            model.deepen([1,1,1,0])
            model = cudafy(model)
            weight_after = parameter_magnitude(model)
            args.shard = "RandomPadding_student_l={l}_w={w}".format(l=lr_drop, w=weight_decay)
            args.total_flops = 0
            args.lr = orig_lr / lr_drop
            args.weight_decay = weight_decay if weight_decay != 100 else orig_wd / weight_after * weight_before
            args.adjust_weight_decay = False
            train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                    _validation_loss, args)

            # Random init start
            if weight_decay != 100:
                model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, num_classes=nc)
                model.deepen([1,1,1,0])
                model = cudafy(model)
                args.shard = "Completely_Random_Init_l={l}_w={w}".format(l=lr_drop, w=weight_decay)
                args.total_flops = 0
                args.lr = orig_lr / lr_drop
                args.weight_decay = weight_decay if weight_decay != 100 else orig_wd / weight_after * weight_before
                args.adjust_weight_decay = False
                train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                        _validation_loss, args)

            # Net2Net
            weight_before = parameter_magnitude(n2nteacher_model)
            model = copy.deepcopy(n2nteacher_model)
            model = cudafy(model)
            weight_after = parameter_magnitude(model)
            model.deepen([1,1,1,0], minibatch=next(iter(train_loader))[0].to('cuda'))
            model = cudafy(model)
            args.shard = "Net2Net_student_l={l}_w={w}".format(l=lr_drop, w=weight_decay)
            args.total_flops = 0
            args.lr = orig_lr / lr_drop
            args.weight_decay = weight_decay if weight_decay != 100 else orig_wd / weight_after * weight_before
            args.adjust_weight_decay = False
            train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                    _validation_loss, args)


            # Random init start v2
            if weight_decay != 100:
                model = orig_resnet12_cifar(thin=True, thinning_ratio=tr, use_residual=False, num_classes=nc)
                model.deepen([1,1,1,0])
                args.shard = "Completely_Random_Init_Net2Net_l={l}_w={w}".format(l=lr_drop, w=weight_decay)
                args.total_flops = 0
                args.lr = orig_lr / lr_drop
                args.weight_decay = weight_decay if weight_decay != 100 else orig_wd / weight_after * weight_before
                args.adjust_weight_decay = False
                train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                        _validation_loss, args)



def _last_weight_decay_tune(args, train_loader, val_loader):
    raise Exception("Test outdated, unused, and won't work if uncommented")
    # # Fix some args for the test (shouldn't ever be loading anythin)
    # args.load = ""
    # if hasattr(args, "flops_budget"):
    #     del args.flops_budget
    # args.widen_times = []
    # args.deepen_times = []

    # orig_lr = args.lr
    # orig_wd = args.weight_decay
    # scaling_factor = 1.5

    # # Teacher network training loop
    # for wd in [0, 1e-6, 1e-5, 1e-4, 3e-3, 1e-3, 1e-2, 1e-1]:
    #     args.shard = str(wd)
    #     args.total_flops = 0
    #     args.lr = orig_lr
    #     args.weight_decay = wd
    #     train_loop(orig_resnet24_cifar(), train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, 
    #                _update_op_cts_eval, _validation_loss, args)]





def iclr_widen_time_experiment(args, lr_drop, widen_times):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    args.deepen_times = []

    orig_lr = args.lr
    max_epochs = args.epochs

    train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)

    # R2R
    for morphism_scheme in ["Net2Net", "r2r", "NetMorph"]:
        for widen_time in widen_times:
            model = orig_resnet12_cifar(thin=True, thinning_ratio=1.5, num_classes=10, use_residual=morphism_scheme!="Net2Net", morphism_scheme=morphism_scheme)
            args.shard = morphism_scheme + "_widen_at_" + str(widen_time)
            args.total_flops = 0
            args.adjust_weight_decay = True
            args.lr = orig_lr
            args.lr_drops = [widen_time]
            args.lr_drop_mag = [lr_drop]
            args.widen_times = [widen_time]
            args.epochs = min(max(2, widen_time//4772) + 1, max_epochs)
            train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op_cts_eval,
                    _validation_loss, args)





def a_svhn_train(args):
    # Fix some args for the test (shouldn't ever be loading anythin)
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    args.deepen_times = []

    train_loader, val_loader = _make_svhn_data_loaders(args, extended=True)

    # R2R
    args.shard = "lr_sched"
    model = orig_resnet18_cifar()
    train_loop(model, train_loader, val_loader, _make_optimizer_fn_sgd, _load_fn, _checkpoint_fn, _update_op_cts_eval,
            _validation_loss, args)