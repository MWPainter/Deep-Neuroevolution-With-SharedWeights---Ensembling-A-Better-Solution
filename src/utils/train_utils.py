from tqdm import tqdm as Bar

from utils.plotting_utils import AverageMeter
from utils.pytorch_utils import cudafy

import os
import string
import time
from collections import defaultdict

from tensorboardX import SummaryWriter





__all__ = ['train_loop']





def train_loop(model, train_loader, val_loader, make_optimizer_fn, load_fn, checkpoint_fn, update_op, validation_loss, args):
    """
    A generic, parameterised training loop, to remove unecessary code duplication.

    A model using this function should contain the following fields for logging purposes:
    model.model_name = the name of the model, e.g. fully_connected_gan.

    Args:
    args.lr = the learning rate to use.
    args.weight_decay = the amount of weight decay to use.
    args.epochs = the number of epochs to train for.
    args.load = a checkpoint file to restart training from. If '' or None, then start training from a random init.
    args.tb_dir = the directory for which to store tensorboard summaries
    args.exp = the id of the current experiment being run (string)
    args.checkpoint_dir = the directory in which to save checkpoint files

    :param model: A PyTorch nn.Module to train
    :param train_loader: A PyTorch DataLoader object to draw minibatches for training from
    :param val_loader: A PyTorch DataLoader object to draw minibatches for validation from
    :param make_optimizer_fn: Function to create an optimizer for a given model.
        Usage "optimizer = make_optimizer_fn(model, learining_rate, weight_decay)"
    :param load_fn: Given a model, optimizer and checkpoint filename, loads the current state of training from a checkpoint.
        Usage "model, optimizer, cur_epoch, best_val_loss = load_fn(model, optimizer, load_dir)"
    :param checkpoint_fn: Given a model, optimizer, epoch number and if the current model is the "best" so far, save a
        current checkpoint for the current state of  training, and also save it as the "best" checkpoint if it's the best.
        Usage "checkpoint_fn(model, optimizer, epoch, best_val_loss, checkpoint_dir, is_best_so_far)"
    :param update_op: Given a minibatch_data and iter number, perform an update for the model, returns the training loss. We
        also pass a tensorboard summary writer to plot losses. The return value is a dictionary of losses. (So that we
        can support optimizations with multiple losses). Args is used to provide any parameters that are specific to the
        model. (E.g. the number of discriminator steps per generator step in a GAN).
        Note that the minibatch data could be just "x's" for a generative model, but for some supervised learning task
        it may be "(x,y,meta)'s". And so on. It should return a reference to the model and optimizer, for the case where
        we wish to make architectural changes to the model and optimizer.
        Usage "model, optimizer, train_losses = update_op(model, optimizer, minibatch, iter, args)"
    :param validation_loss: Given a model and a minibatch, compute a validation loss, returns losses corresponding to
        (some) of the training losses. Only validation losses will be plotted for epoch averages.
        Usage "validations_loss = validation_loss(model, minibatch, args)"
    :param args: Argparser arguments to use, required to contain the values mentioned above.
    :returns: The trained model, so that we could do something else with it later
    """
    # Move the model to the GPU if it is available
    model = cudafy(model)

    # Tensorboard summary writer, and progress bar (for babysitting training)
    log_file = "{folder}/{exp}_tb_log".format(folder=args.tb_dir, exp=args.exp)
    if hasattr(args, 'shard'):
        log_file = os.path.join(log_file, args.shard)
    writer = SummaryWriter(log_dir=log_file)

    # Add same directory structure for checkpoints
    checkpoint_dir = "{folder}/{exp}_checkpoints".format(folder=args.checkpoint_dir, exp=args.exp)
    if hasattr(args, 'shard'):
        checkpoint_dir = os.path.join(checkpoint_dir, args.shard)

    # Load models/make optimizers, and restore the state of training if loading from a checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    optimizer = make_optimizer_fn(model, args.lr, args.weight_decay, args)
    if args.load:
        print("Loading from checkpoint...")
        model, optimizer, start_epoch, best_val_loss = load_fn(model, optimizer, args.load)
        print("Loaded checkpoint!")

    # Run a validation epoch on the randomly initialized network
    else:
        print("Epoch {epoch} validation:".format(epoch=start_epoch))
        model.eval()
        validation_op = lambda model, optimizer, mbatch, _b, args: (model, optimizer, validation_loss(model, mbatch, args))
        _, _, avg_val_losses = _train_loop_epoch(model, val_loader, validation_op, optimizer,
                                                 start_epoch*len(train_loader), writer, "val/", args)

        # Log the initial validation stats
        for key in avg_val_losses:
            scalar_name = ''.join(['epoch/', key])
            writer.add_scalars(scalar_name, {'test': avg_val_losses[key]}, start_epoch)

    # Main train loop
    for epoch in range(start_epoch, args.epochs):
        # Training epoch
        print("Epoch {epoch} training:".format(epoch=epoch+1))
        model.train()
        cur_global_iter = epoch * len(train_loader)
        model, optimizer, avg_losses = _train_loop_epoch(model, train_loader, update_op, optimizer, cur_global_iter, 
                                                         writer, "train/", args)

        # Validation epoch (same as train, but replace 'update_op' and 'train_loader' appropriately and run network in eval 
        # mode)
        print("Epoch {epoch} validation:".format(epoch=epoch+1))
        model.eval()
        cur_global_iter = (epoch + 1) * len(train_loader)
        validation_op = lambda model, optimizer, mbatch, _b, args: (model, optimizer, validation_loss(model, mbatch, args))
        _, _, avg_val_losses = _train_loop_epoch(model, val_loader, validation_op, optimizer,
                                                 cur_global_iter, writer, "val/", args)

        # Logging per epoch
        for key in avg_val_losses:
            scalar_name = ''.join(['epoch/', key])
            writer.add_scalars(scalar_name, {'train': avg_losses[key], 'test': avg_val_losses[key]}, epoch+1)

        # Checkpointing (depending on the model the "best" model may or may not make sense (e.g. GAN it will not))
        avg_val_loss = sum(list(avg_val_losses.values()))
        is_best_model = avg_val_loss < best_val_loss
        best_val_loss = min(avg_val_loss, best_val_loss)
        checkpoint_fn(model, optimizer, epoch, best_val_loss, checkpoint_dir, is_best_model)

    print("Fin.")
    return model





def _train_loop_epoch(model, data_loader, step_op, optimizer, global_iter, writer, tb_prefix, args):
    """
    The inner training loop of an epoch.

    :param model: The PyTorch nn.Module to train
    :param data_loader: A PyTorch data loader that is used to provide minibatches
    :param step_op: This should either by 'validation_loss' to compute validation losses or 'update_op' to update
        weights in the network (as a side effect of the 'update_op' function)
    :param optimizer: A PyTorch optimizer (just passed to the step_op)
    :param global_iter: The current global step at the start of this loop (n.b. this is a local variable)
    :param bar: A progress bar to update the CLI on training
    :param writer: A tensorboardX summary writer
    :param tb_prefix: A prefix to prepend to any tensorboard logging (to differentiate between train and val plots in
        tensorboard).
    :param args: Argparser arguments, used to provide model specific parameters.
    :return: The updated model (may have changed architecture), optimizer and a dictionary of losses, keyed by strings, the 'name' for each loss.
    """
    # Average meters for the losses and times, for the progress bar
    batch_total_time = AverageMeter()
    data_load_time = AverageMeter()
    avg_losses_dict = defaultdict(AverageMeter)
    bar = Bar(total=len(data_loader))

    iter_end_time = time.time()
    iter = 0
    for minibatch_data in data_loader:
        # Compute the time needed to load the minibatch
        data_load_time.update(time.time() - iter_end_time)

        # Make a step
        model, optimizer, losses = step_op(model, optimizer, minibatch_data, global_iter, args)
        batch_total_time.update(time.time() - iter_end_time)

        # Tensorboard plotting, logging per minibatch and updating averages
        # if global_iter % args.tb_log_freq == 0:
        for key in losses:
            scalar_name = ''.join(['iter/', tb_prefix, key])
            writer.add_scalar(scalar_name, losses[key], global_iter)
            minibatch_size = minibatch_data[0].size(0)
            avg_losses_dict[key].update(losses[key], n=minibatch_size)
        
        if global_iter % args.tb_log_freq == 0 and hasattr(model, 'log_weights'):
            model.log_weights(summary_writer=writer, iteration=global_iter)

        # Update averages and progress bar
        prog_stats = {
            "batch": "{cb}/{tb}".format(cb=iter+1, tb=len(data_loader)),
            "data_time": "{s:.2f}sec".format(s=data_load_time.val),
            "batch_time": "{s:.2f}sec".format(s=batch_total_time.val),
        }
        bar.set_postfix(prog_stats)
        bar.update()

        # update the time and indices for the next iteration
        global_iter +=1
        iter += 1
        iter_end_time = time.time()

    bar.close()

    # return the model, optimizer and average losses (as floats)
    return model, optimizer, {key: avg_losses_dict[key].avg for key in avg_losses_dict}