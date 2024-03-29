import os
import math
import copy
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.misc

from torch.utils.data import Dataset, DataLoader
from dataset import MnistDataset, CifarDataset, SvhnDataset

from utils import cudafy
from utils import train_loop
from utils import parameter_magnitude, gradient_magnitude, update_magnitude, update_ratio
from utils import flatten

from r2r import r_2_wider_r_, net_2_wider_net_





"""
Visualizations using a 2 layer FC network
"""





"""
Network Definition
"""




def _visualize_grid(Xs, ubound=255.0, padding=1, viz_width=0, kernel_norm=True):
    """
    Adapted from the cs231n starter code.
    """
    (N, H, W, C) = Xs.shape
    viz_height = int(math.ceil(N / viz_width))
    grid_height = H * viz_height + padding * (viz_height - 1)
    grid_width = W * viz_width + padding * (viz_width - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    low, high = np.min(Xs), np.max(Xs)
    Xs_norm = (Xs - low) / (high - low + 1.0e-6)
    y0, y1 = 0, H
    for y in range(viz_height):
        x0, x1 = 0, W
        for x in range(viz_width):
            if next_idx < N:
                if kernel_norm:
                    img = Xs_norm[next_idx]
                    grid[y0:y1, x0:x1] = ubound * img
                else:
                    img = Xs[next_idx]
                    low, high = np.min(img), np.max(img)
                    grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low + 1.0e-6)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid




class FC_Net(nn.Module):
    def __init__(self, hidden_units, in_channels=1, widen_method='r2r', multiplicative_widen=False, init_scale=1.0):
        super(FC_Net, self).__init__()
        self.widen_method = widen_method.lower()
        self.multiplicative_widen = multiplicative_widen
        self.in_channels = in_channels
        self.hidden_units = hidden_units
        self.init_scale = init_scale
        self.W1 = nn.Linear(in_channels * 32 * 32, hidden_units)
        self.bn = nn.BatchNorm1d(num_features=hidden_units)
        self.W2 = nn.Linear(hidden_units, 10)

    def forward(self, x):
        x = flatten(x)
        x = self.W1(x)
        x = F.relu(x)
        x = self.W2(x)

        return x

    def clip_(self):
        pass
        # for p in self.parameters():
        #     p.data.clamp_(-1.0, 1.0)

    def widen(self, num_channels=2):
        if self.widen_method == 'r2r':
            r_2_wider_r_(self.W1, (self.hidden_units,), self.W2, extra_channels=num_channels,
                         init_type=self.init_scale, function_preserving=True, multiplicative_widen=self.multiplicative_widen)
        elif self.widen_method == 'net2net':
            net_2_wider_net_(self.W1, self.W2, (self.hidden_units,), extra_channels=num_channels,
                             multiplicative_widen=self.multiplicative_widen, add_noise=True)
        elif self.widen_method == 'netmorph':
            r_2_wider_r_(self.W1, (self.hidden_units,), self.W2, extra_channels=num_channels,
                         init_type="match_std", function_preserving=True,
                         multiplicative_widen=self.multiplicative_widen, net_morph=True, net_morph_add_noise=True)

        if self.multiplicative_widen:
            self.hidden_units *= (num_channels - 1)
        else:
            self.hidden_units += num_channels

    def save_weights(self, iter, dir, viz_width):
        shape = (self.hidden_units, self.in_channels, 32, 32)
        weights = self.W1.weight.data.detach().cpu().view(shape).numpy()
        weights_scipy = np.transpose(weights, (0,2,3,1))
        weights_img = np.squeeze(_visualize_grid(weights_scipy, viz_width=viz_width, kernel_norm=True)) # (self.widen_method=='netmorph')))
        filename = "{iter:0>6d}.png".format(iter=iter)
        filepath = os.path.join(dir, filename)
        scipy.misc.imsave(filepath, weights_img)




class Conv_Net(nn.Module):
    def __init__(self, hidden_units, conv_channels, in_channels=1, widen_method='r2r', multiplicative_widen=False, init_scale=1.0):
        super(Conv_Net, self).__init__()
        self.widen_method = widen_method.lower()
        self.multiplicative_widen = multiplicative_widen
        self.init_scale = init_scale
        self.conv1 = nn.Conv2d(in_channels, conv_channels, kernel_size=9, padding=4, stride=1)
        self.pool1 = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(conv_channels, conv_channels * 4, kernel_size=3, padding=1, stride=1)
        self.W1 = nn.Linear(conv_channels * 16 * 16, hidden_units)
        self.bn = nn.BatchNorm1d(num_features=hidden_units)
        self.W2 = nn.Linear(hidden_units, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        x = flatten(x)
        x = self.W1(x)
        x = F.relu(x)
        x = self.W2(x)

        return x

    def clip_(self):
        pass
        # for p in self.parameters():
        #     p.data.clamp_(-1.0, 1.0)

    def widen(self, num_channels=16, num_hidden=50):
        if self.widen_method == 'r2r':
            r_2_wider_r_(prev_layers=self.conv1, volume_shape=(self.conv1.weight.data.size(0),16,16),
                         next_layers=self.W1, extra_channels=num_channels, init_type=self.init_scale,
                         function_preserving=True, multiplicative_widen=self.multiplicative_widen)
            # r_2_wider_r_(self.W1, (self.W1.weight.data.size(0),), self.W2, extra_channels=num_hidden,
            #              init_type="match_std_exact", function_preserving=True, multiplicative_widen=self.multiplicative_widen)
        elif self.widen_method == 'net2net':
            net_2_wider_net_(prev_layers=self.conv1, next_layers=self.W1, 
                             volume_shape=(self.conv1.weight.data.size(0),16,16), 
                             extra_channels=num_channels,
                             multiplicative_widen=self.multiplicative_widen, add_noise=True)
            # net_2_wider_net_(self.W1, self.W2, (self.W1.weight.data.size(0),), extra_channels=num_hidden,
            #                  multiplicative_widen=self.multiplicative_widen, add_noise=True)
        elif self.widen_method == 'netmorph':
            r_2_wider_r_(prev_layers=self.conv1, volume_shape=(self.conv1.weight.data.size(0),16,16),
                         next_layers=self.W1, extra_channels=num_channels, init_type="match_std",
                         function_preserving=True, multiplicative_widen=self.multiplicative_widen, net_morph=True)
            # r_2_wider_r_(self.W1, (self.W1.weight.data.size(0),), self.W2, extra_channels=num_hidden,
            #              init_type="match_std_exact", function_preserving=True,
            #              multiplicative_widen=self.multiplicative_widen, net_morph=True)

    def save_weights(self, iter, dir, viz_width):
        weights = self.conv1.weight.data.detach().cpu().numpy()
        weights_scipy = np.transpose(weights, (0,2,3,1))
        # weights_normalized = (weights_scipy + 1.0) / 2.0
        weights_img = np.squeeze(_visualize_grid(weights_scipy, viz_width=viz_width, kernel_norm=True))
        filename = "{iter:0>6d}.png".format(iter=iter)
        filepath = os.path.join(dir, filename)
        scipy.misc.imsave(filepath, weights_img)






"""
Defining the training loop.
"""





def _make_optimizer_fn(model, lr, weight_decay, args=None, adam=True):
    """
    The make optimizer function, as part of the interface for the "train_loop" function in utils.train_utils.

    :param model: The model to make an optimizer for.
    :param lr: The learning rate to use
    :param weight_decay: THe weight decay to use
    :return: The optimizer for the network optimizer, which is passed into the remaining
        training loop functions
    """
    # return t.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    if adam:
        return t.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    else:
        return t.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    # return t.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)





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
    # Switch model to training mode, and cudafy minibatch
    model.train()
    xs, ys = cudafy(minibatch[0]), cudafy(minibatch[1])

    # Widen or deepen the network at the correct times
    # if iter in args.widen_times:
    #     model.widen()
    #     #args.lr /= 2.0
    #     args.weight_decay /= 10
    #     if model.widen_method == 'netmorph':
    #         args.weight_decay = 1.0e-5


    #     if args.adjust_weight_decay:
    #         weight_decay_ratio = weight_mag_before / weight_mag_after
    #         args.weight_decay *= weight_decay_ratio

    #     optimizer = _make_optimizer_fn(model, args.lr, args.weight_decay, args)

    if iter in args.widen_times:
        weight_mag_before = parameter_magnitude(model)
        if iter in args.widen_times:
            print("Widening!")
            model.widen(1.5)
        weight_mag_after = parameter_magnitude(model)
        model = cudafy(model)
        if args.adjust_weight_decay:
            weight_decay_ratio = weight_mag_before / weight_mag_after
            args.weight_decay *= weight_decay_ratio
        optimizer = _make_optimizer_fn(model, args.lr, args.weight_decay, args)

    # Forward pass - compute a loss
    loss_fn = _make_loss_fn()
    ys_pred = model(xs)
    loss = loss_fn(ys_pred, ys)

    # Backward pass - make an update step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Clip parameters
    model.clip_()

    # Compute loss and accuracy
    losses = {}
    losses['loss'] = loss
    losses['accuracy'] = _accuracy(ys_pred, ys)[0]

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


    widen_pm = [w-1 for w in args.widen_times] + [w for w in args.widen_times] + [w+1 for w in args.widen_times]
    if iter % 10 == 0 or iter in widen_pm:
        img_dir = os.path.join("{folder}/{exp}_checkpoints".format(folder=args.checkpoint_dir, exp=args.exp), args.shard, "weight_visuals")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        model.save_weights(iter, img_dir, args.initial_channels)

    # Return the model, optimizer and dictionary of 'losses'
    return model, optimizer, losses





def _accuracy(output, target, topk=(1,)):
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
    with t.no_grad():
        # Put in eval mode
        model.eval()

        # Unpack minibatch
        xs, ys = cudafy(minibatch[0]), minibatch[1]

        # Compute loss and accuracy
        loss_fn = _make_loss_fn()
        ys_pred = model(xs).cpu()
        loss = loss_fn(ys_pred, ys)
        acc = _accuracy(ys_pred, ys)[0]

        # Return the dictionary of losses
        return {'loss': loss,
                'accuracy': acc}





"""
Tests
"""





def _mnist_weight_visuals(args, widen_method="r2r", use_conv=False, start_wide=False):
    """
    Trains the FC net, and provides weight visualizations to the checkpoint directory.
    """
    # Make the data loader objects
    train_dataset = MnistDataset(train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    val_dataset = MnistDataset(train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # Make the model
    if use_conv:
        args.initial_channels = 8
        init_channels = 32 if start_wide else 16
        init_hidden = 50 #250 if start_wide else 50
        model = Conv_Net(init_hidden, init_channels, in_channels=1, widen_method=widen_method)
    else:
        args.initial_channels = 2
        init_channels = 8 if start_wide else 2
        model = FC_Net(init_channels, in_channels=1, widen_method=widen_method)

    # Train
    model = train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)

    # return the model in case we want to do anything else with it
    return model, train_loader, val_loader





def _svhn_weight_visuals(args, conv=True, svhn=True, adam=True):
    """
    Trains the FC net, and provides weight visualizations to the checkpoint directory.
    """
    # Args stuff
    args.load = ""
    if hasattr(args, "flops_budget"):
        del args.flops_budget
    args.widen_times = []
    args.deepen_times = []

    orig_lr = args.lr
    orig_wd = args.weight_decay

    # Make the data loader objects
    if not svhn:
        train_dataset = CifarDataset(train=True, labels_as_logits=False)
    else:
        train_dataset = SvhnDataset(train=True, labels_as_logits=False, use_extra_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True)

    if not svhn:
        val_dataset = CifarDataset(train=False, labels_as_logits=False)
    else:
        val_dataset = SvhnDataset(train=False, labels_as_logits=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # Make the model
    args.initial_channels = 4
    args.shard = "teacher"
    args.total_flops = 0
    args.lr = orig_lr
    args.weight_decay = orig_wd
    if conv:
        init_channels = 4 #16
        init_hidden = 32 #150 #250 if start_wide else 50
        model = Conv_Net(init_hidden, init_channels, in_channels=3, widen_method='r2r', init_scale=1.0)
    else:
        init_channels = 4 
        init_hidden = 4 
        model = FC_Net(hidden_units=init_hidden, in_channels=3, widen_method='r2r', init_scale=1.0)

    # Setup using adam/sgd
    optimizer_fn = (lambda model, lr, weight_decay, args=None:
                        _make_optimizer_fn(model, lr, weight_decay, args, adam))

    # Train teacher
    teacher_model = train_loop(model, train_loader, val_loader, optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                       _validation_loss, args)
    weight_before = parameter_magnitude(teacher_model)

    for weight_decay_match in [True, False]:

        # N2N
        model = copy.deepcopy(teacher_model)
        model.widen_method = 'net2net' 
        model.widen(num_channels=init_channels)
        weight_after = parameter_magnitude(model)
        args.shard = "N2N_wd_match={w}".format(w=weight_decay_match)
        args.total_flops = 0
        args.adjust_weight_decay = weight_decay_match
        args.lr = orig_lr #/ 10.0
        args.weight_decay = orig_wd if not weight_decay_match else orig_wd / weight_after * weight_before
        model = train_loop(model, train_loader, val_loader, optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                        _validation_loss, args)


        for init_scale in [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.1]: 

            # R2R
            model = copy.deepcopy(teacher_model)
            model.init_scale = init_scale
            model.widen_method = 'r2r' 
            model.widen(num_channels=init_channels)
            weight_after = parameter_magnitude(model)
            args.shard = "R2R_scale={s}_wd_match={w}".format(s=init_scale, w=weight_decay_match)
            args.total_flops = 0
            args.adjust_weight_decay = weight_decay_match
            args.lr = orig_lr #/ 10.0
            args.weight_decay = orig_wd if not weight_decay_match else orig_wd / weight_after * weight_before
            model = train_loop(model, train_loader, val_loader, optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                            _validation_loss, args)
    
    # NetMorph
    model = copy.deepcopy(teacher_model)
    model.widen_method = 'netmorph'
    model.widen(num_channels=init_channels)
    weight_after = parameter_magnitude(model)
    args.shard = "NetMorph"
    args.total_flops = 0
    args.adjust_weight_decay = weight_decay_match
    args.lr = orig_lr #/ 10.0
    args.weight_decay = orig_wd if not weight_decay_match else orig_wd / weight_after * weight_before
    model = train_loop(model, train_loader, val_loader, optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
                    _validation_loss, args)






