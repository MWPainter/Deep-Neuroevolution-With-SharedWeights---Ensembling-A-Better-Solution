from __future__ import print_function
import sys
import os
import argparse
import random
import numpy as np
import torch as t

from batch_tests import mnist_identity_init_test, cifar_identity_init_test, mnist_widen_test, cifar_widen_test, mnist_deepen_test, cifar_deepen_test
from batch_tests import mnist_widen_with_budget_test, cifar_widen_with_budget_test, mnist_deepen_with_budget_test, cifar_deepen_with_budget_test
from batch_tests import mnist_net_to_net_style_test, cifar_net_to_net_style_test




"""
This contains the main entry point from which to run scripts.
"""





"""
TODO 0: Add defaults for deepen and widen times
TODO 1: tests
"""






def get_defaults(script_name):
    checkpoint_dir = "checkpoints"
    tb_log_dir = "tb_logs"
    exp_id = "{sn}_default".format(sn=script_name)

    if script == "mnist_identity_init":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [],
            "flops_budget": 0 #unused
        }
    elif script == "cifar_identity_init":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [],
            "flops_budget": 0 # unused
        }


    elif script == "mnist_widen":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [6000],
            "deepen_times": [],
            "flops_budget": 0 #unused
        }
    elif script == "cifar_widen":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [6000],
            "deepen_times": [],
            "flops_budget": 0 # unused
        }
    elif script == "mnist_widen_with_budget":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [6000],
            "deepen_times": [],
            "flops_budget": 1.0e12
        }
    elif script == "cifar_widen_with_budget":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [6000],
            "deepen_times": [],
            "flops_budget": 2.0e12
        }


    elif script == "cifar_widen_multi_stage":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [3000,6000,9000],
            "deepen_times": [],
            "flops_budget": 0 # unused
        }
    elif script == "cifar_widen_with_budget_multi_stage":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [3000,6000,9000],
            "deepen_times": [],
            "flops_budget": 2.0e12
        }


    elif script == "mnist_deepen":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [6000],
            "flops_budget": 0 #unused
        }
    elif script == "cifar_deepen":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [6000],
            "flops_budget": 0 # unused
        }
    elif script == "mnist_deepen_with_budget":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [6000],
            "flops_budget": 4.0e11
        }
    elif script == "cifar_deepen_with_budget":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [6000],
            "flops_budget": 8.0e11
        }


    elif script == "cifar_deepen_multi_stage":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 8,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [3000,6000,9000],
            "flops_budget": 0 # unused
        }
    elif script == "cifar_deepen_with_budget_multi_stage":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [3000,6000,9000],
            "flops_budget": 8.0e11
        }


    elif script == "mnist_net2widernet_style":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0 # unused
        }
    elif script == "mnist_net2deepernet_style":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0 # unused
        }



    elif script == "cifar_net2widernet_style":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0 # unused
        }
    elif script == "cifar_net2deepernet_style":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 20,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0 # unused
        }


    else:
        print("Couldn't find defaults for '{s}'".format(s=script))





def make_arg_parser(defaults):
    """
    Makes and returns an argparse object
    """
    parser = argparse.ArgumentParser()

    # things that can have a default value
    parser.add_argument('--lr', type=float, default=defaults["lr"],
                        help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=defaults["weight_decay"],
                        help='Coefficient for weight decay to apply')
    parser.add_argument('--epochs', type=int, default=defaults["epochs"],
                        help='Number of epochs to train for.')
    parser.add_argument('--tb_dir', type=str, default=defaults["tb_dir"],
                        help="Directory to write tensorboardX summaries.")
    parser.add_argument('--checkpoint_dir', type=str, default=defaults["checkpoint_dir"],
                        help='path to store model checkpoints')
    parser.add_argument('--exp', type=str, default=defaults["exp"],
                        help='ID of experiment')
    parser.add_argument('--workers', type=int, default=defaults["workers"],
                        help='The number of workers to use in a PyTorch data loader.')
    parser.add_argument('--flops_budget', type=int, default=defaults["flops_budget"],
                        help='A flops budget to adhere to (if used in the test).')
    parser.add_argument('--batch_size', type=int, default=defaults["batch_size"],
                        help='The batch size to use')

    # When to widen/deepen in widen/deepen tests
    parser.add_argument('--widen_times', type=int, nargs='+', default=defaults['widen_times'],
                        help='When the network should be widened in tests.')
    parser.add_argument('--deepen_times', type=int, nargs='+', default=defaults['deepen_times'],
                        help='When the network should be deepened in tests.')

    # Fix seed for reproducability
    parser.add_argument('--seed', type=int, default=234,
                        help='Specify a seed for random generation (math/numpy/PyTorch).')

    # Things that sbouldn't ever need to change
    parser.add_argument('--tb_log_freq', type=int, default=101,
                        help='How frequently to update tensorboard summaries (num of iters per update). Default is prime incase we are computing different losses on different iterations.')

    # Things that should be empty unless specified
    parser.add_argument('--load', type=str, default="",
                        help='path to load a pretrained checkpoint')

    return parser





if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: 'python main.py <script_name> <options>'")
        exit()

    # Get script name + options from command line
    script = sys.argv[1]
    defaults = get_defaults(script)
    parser = make_arg_parser(defaults)
    args, _ = parser.parse_known_args()

    # Fix seed for reproducability
    random.seed(args.seed)
    np.random.seed(args.seed + 11)
    t.manual_seed(args.seed + 101)
    t.cuda.manual_seed_all(args.seed + 1001)

    # Run the given script that we wish to run
    if script == "mnist_identity_init": ##########
        mnist_identity_init_test(args)
    elif script == "cifar_identity_init":
        cifar_identity_init_test(args)

    elif script == "mnist_widen":
        mnist_widen_test(args)
    elif script == "cifar_widen": #
        cifar_widen_test(args)
    elif script == "mnist_widen_with_budget":
        mnist_widen_with_budget_test(args)
    elif script == "cifar_widen_with_budget":
        cifar_widen_with_budget_test(args)

    elif script == "cifar_widen_multi_stage":
        cifar_widen_test(args)
    elif script == "cifar_widen_with_budget_multi_stage": #
        cifar_widen_with_budget_test(args)

    elif script == "mnist_deepen":
        mnist_deepen_test(args)
    elif script == "cifar_deepen":
        cifar_deepen_test(args)
    elif script == "mnist_deepen_with_budget":
        mnist_deepen_with_budget_test(args)
    elif script == "cifar_deepen_with_budget":
        cifar_deepen_with_budget_test(args)

    elif script == "cifar_deepen_multi_stage":
        cifar_deepen_test(args)
    elif script == "cifar_deepen_with_budget_multi_stage":
        cifar_deepen_with_budget_test(args)

    elif script == "mnist_net2widernet_style":
        mnist_net_to_net_style_test(args)
    elif script == "cifar_net2widernet_style": #
        cifar_net_to_net_style_test(args)

    elif script == "mnist_net2deepernet_style":
        mnist_net_to_net_style_test(args, widen=False)
    elif script == "cifar_net2deepernet_style":
        cifar_net_to_net_style_test(args, widen=False)

    else:
        print("Couldn't find script for '{s}'".format(s=script))