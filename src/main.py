from __future__ import print_function
import sys
import os
import argparse
import random
import numpy as np
import torch as t

from .batch_tests import mnist_identity_init_test, cifar_identity_init_test, mnist_widen_test, cifar_widen_test, mnist_deepen_test, cifar_deepen_test
from .batch_tests import mnist_widen_with_budget_test, cifar_widen_with_budget_test, mnist_deepen_with_budget_test, cifar_deepen_with_budget_test
from .batch_tests import mnist_net_to_net_style_test, cifar_net_to_net_style_test




"""
This contains the main entry point from which to run scripts.
"""





"""
TODO 0: Actually make the widen and deepen calls properly, using args.use_random_padding 
TODO 1: Work out the correct things to set as defaults, in lr and so on (compare with what there was beforew, and convert iters to batch)
TODO 2: Move flobs budget to options, and work out what's actually a good value for it (also make sure that epochs are set to be high enough)
TODO 3: Use an 'args='+'' type thing to have widen_times as an option + add it to the defaults for SOME tests. Defaults on all others are []
TODO 4: Repeat for deepening
TODO 5: Add correct defaults for the multi stage tests
TODO 6: Write the bit which widens/deepens the networks for the Net2Net style tests
TODO 7: Run the tests, look at the tb output and check that it's sane
TODO 8: Re-read the Net2Net paper to get exactly the tests that they ran
TODO 9: Deal with the limitation of not being able to specify teacher epochs vs student epochs in net2net tests
TODO 10: Make unused options == None when they're not intended to ever be used
"""






def get_defaults(script_name):
    checkpoint_dir = "checkpoints"
    tb_log_dir = "tb_logs"
    exp_id = "{sn}_0".format(sn=script_name)

    if script == "mnist_identity_init":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }
    elif script == "cifar_identity_init":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }


    elif script == "mnist_widen":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }
    elif script == "cifar_widen":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }
    elif script == "mnist_widen_with_budget":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }
    elif script == "cifar_widen_with_budget":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }


    elif script == "cifar_widen_multi_stage":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }
    elif script == "cifar_widen_with_budget_multi_stage":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }


    elif script == "mnist_deepen":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }
    elif script == "cifar_deepen":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }
    elif script == "mnist_deepen_with_budget":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }
    elif script == "cifar_deepen_with_budget":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }


    elif script == "cifar_deepen_multi_stage":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }
    elif script == "cifar_deepen_with_budget_multi_stage":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }


    elif script == "Net2WiderNet_style":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }
    elif script == "Net2DeeperNet_style":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }



    elif script == "mnist_net2net_style":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
        }
    elif script == "cifar_net2net_style":
        return {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 50,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32
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
    parser.add_argument('--worksers', type=int, default=defaults["workers"],
                        help='The number of workers to use in a PyTorch data loader.')

    # Batch sizes
    parser.add_argument('--batch_size', type=int, default=defaults["batch_size"],
                        help='The batch size to use')

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
    if len(sys.argv) != 2:
        print("Usage: 'python main.py <script_name>'")

    # Get script name + options from command line
    script = sys.argv[1]
    defaults = get_defaults(script)
    parser = make_arg_parser(defaults)
    args, _ = parser.parse_known_args()

    # Fix seed for reproducability
    random.seed(options.seed)
    np.random.seed(options.seed + 11)
    t.manual_seed(options.seed + 101)
    t.cuda.manual_seed_all(options.seed + 1001)

    # Run the given script that we wish to run
    if script == "mnist_identity_init":
        mnist_identity_init_test(args)
    elif script == "cifar_identity_init":
        cifar_identity_init_test(args)

    elif script == "mnist_widen":
        mnist_widen_test(args)
    elif script == "cifar_widen":
        cifar_widen_test(args)
    elif script == "mnist_widen_with_budget":
        mnist_widen_with_budget_test(args)
    elif script == "cifar_widen_with_budget":
        cifar_widen_with_budget_test(args)

    elif script == "cifar_widen_multi_stage":
        cifar_widen_test(args)
    elif script == "cifar_widen_with_budget_multi_stage":
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

    elif script == "mnist_net2net_style":
        mnist_net_to_net_style_test(args)
    elif script == "cifar_net2net_style":
        cifar_net_to_net_style_test(args)

    else:
        print("Couldn't find script for '{s}'".format(s=script))