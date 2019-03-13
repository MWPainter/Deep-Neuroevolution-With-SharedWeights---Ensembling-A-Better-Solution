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

# from run import _net_2_wider_net_inception_test, _r_2_wider_r_inception_test
# from run import net_2_wider_net_resnet, net_2_deeper_net_resnet
# from run import net_2_wider_net_resnet_hyper_search, net_2_deeper_net_resnet_hyper_search
# from run import r_2_wider_r_resnet, r_2_deeper_r_resnet
# from run import quadruple_widen_run, double_deepen_run, double_widen_and_deepen_run
# from run import r2r_faster_test_part_1, r2r_faster_test_part_2, r2r_faster_test_part_3, r2r_faster_test_part_4, r2r_faster_test_redo, r2r_faster_test_redo_18
# from run import r_2_r_weight_init_example, net_2_net_overfit_example
# from run import r2r_faster
from run import *

from viz import _mnist_weight_visuals, _cifar_weight_visuals

from r2r import *




"""
This contains the main entry point from which to run scripts.
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
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 1.0e12,
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 2.0e12,
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 2.0e12,
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 4.0e11,
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 8.0e11,
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 8.0e11,
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
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
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }



    elif script == "n2n_wider_inception":
        return {
            "lr": 1.0e-4,
            "weight_decay": 1.0e-7,
            "epochs": 8,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 4,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }
    elif script == "r2r_wider_inception":
        return {} # TODO: actually set params


    #######
    # Weight visualizations tests
    #######
    elif script == "mnist_weight_viz_r2r":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0,
            "epochs": 100,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": range(2350*2+1175, 2350*10, 2350*2+1175),
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }
    elif script == "cifar_weight_viz_r2r":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0,
            "epochs": 100,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": range(3830*2+1915, 3830*10, 3830*2+1915),
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }
    elif script == "mnist_weight_viz_r2r_conv":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "epochs": 100,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": range(2350*2+1175, 2350*10, 2350*2+1175),
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }
    elif script == "cifar_weight_viz_r2r_conv":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 800,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 256,
            "workers": 6,
            #"widen_times": range(3830*2+1915, 3830*10, 3830*2+1915),
            "widen_times": [383 * 150],
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }
    elif script == "mnist_weight_viz_net2net":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0,
            "epochs": 100,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": range(2350*2+1175, 2350*10, 2350*2+1175),
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }
    elif script == "cifar_weight_viz_net2net":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0,
            "epochs": 100,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": range(3830*2+1915, 3830*10, 3830*2+1915),
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }
    elif script == "mnist_weight_viz_net2net_conv":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 100,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": range(2350*2+1175, 2350*10, 2350*2+1175),
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }
    elif script == "cifar_weight_viz_net2net_conv":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 400,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 256,
            "workers": 6,
            #"widen_times": range(3830*2+1915, 3830*10, 3830*2+1915),
            "widen_times": [3830 * 5],
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }
    elif script == "mnist_weight_viz_netmorph":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0,
            "epochs": 100,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": range(2350*2+1175, 2350*10, 2350*2+1175),
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }
    elif script == "cifar_weight_viz_netmorph":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0,
            "epochs": 100,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": range(3830*2+1915, 3830*10, 3830*2+1915),
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }
    elif script == "mnist_weight_viz_netmorph_conv":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 600,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 256,
            "workers": 6,
            #"widen_times": range(3830*2+1915, 3830*10, 3830*2+1915),
            "widen_times": [383 * 150],
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }
    elif script == "cifar_weight_viz_netmorph_conv":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 400,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 256,
            "workers": 6,
            #"widen_times": range(3830*2+1915, 3830*10, 3830*2+1915),
            "widen_times": [3830 * 5],
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused
            "grad_clip": 0.0
        }
    elif script == "mnist_weight_viz":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0,
            "epochs": 100,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "cifar_weight_viz":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0,
            "epochs": 100,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "mnist_weight_viz_conv":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "epochs": 100,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "cifar_weight_viz_conv":
        return {
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "epochs": 400,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }


    #######
    # Net 2 Net Style tests, and R2R style tests
    #######
    elif script == "n2wn":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "n2dn":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "n2wn_hps":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "n2dn_hps":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "r2wr": # @7epoch widen r2r, @___ epoch widen n2n
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [1532*15],#7], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }

    elif script == "r2dr":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [1532*15],#10], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }





    #######
    # Examples where things go wrong
    #######
    elif script == "n2n_overfit_problem":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 120,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "r2r_weight_init_problem":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 160,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }





    #######
    # Learning rate adaption tests
    #######
    elif script == "quad_widen":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 40,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [1532*6,1532*12,1532*18,1532*24], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "double_deepen":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 40,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [1532*10,1532*20], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "double_widen_deepen":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-6,
            "epochs": 40,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 32,
            "workers": 6,
            "widen_times": [1532*6,1532*18], # unused
            "deepen_times": [1532*12,1532*24], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }





    #######
    # Fast training tests
    #######
    elif script == "r2fasterr_part_1": # fixed grad drop @ every widen and deepen
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 90,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 256,
            "workers": 6,
            "widen_times": [5005*30], # 30 epochs
            "deepen_times": [5005*15], # 15 epochs
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [5005*15, 5005*30, 5005*60], # 30, 60
            "lr_drop_mag": [10.0],
            "grad_clip": 10.0
        }
    elif script == "r2fasterr_part_2": # split grad drops during widenings
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 90,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 256,
            "workers": 6,
            "widen_times": [5005*30], # 30 epochs
            "deepen_times": [5005*15], # 15 epochs
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [5005*15, 5005*30, 5005*60], # 30, 60
            "lr_drop_mag": [np.sqrt(10.0), np.sqrt(10.0), 10.0],
            "grad_clip": 10.0
        }
    elif script == "r2fasterr_part_3": # Only lr drops when there usually are in training a resnet18
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 60,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [5005*30, 5005*60], # 30, 60
            "lr_drop_mag": [10.0],
            "grad_clip": 10.0
        }
    elif script == "r2fasterr_part_4": # Training student network to completion
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 60,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [20019*30, 20019*45],
            "lr_drop_mag": [10.0],
            "grad_clip": 10.0
        }
    elif script == "r2fasterr_part_5": # Training teacher network to completion
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 60,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [5005*30, 5005*60], # 30, 60
            "lr_drop_mag": [10.0],
            "grad_clip": 10.0
        }





    #######
    # Fast training tests, with resnet18
    #######
    elif script == "r2fasterr_redo_part_0": # thin resnet10->18
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 60,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 256,
            "workers": 6,
            "widen_times": [5005*7], # 30
            "deepen_times": [5005*15], # 20
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [5005*15, 5005*30], # 30, 60
            "lr_drop_mag": [np.sqrt(10.0), np.sqrt(10.0), 10.0],
            "grad_clip": 1.0
        }
    elif script == "r2fasterr_redo_part_1": # thin resnet10 -> resnet18
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 60,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 256,
            "workers": 6,
            "widen_times": [5005*15], # 30
            "deepen_times": [0], # 20
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [5005*15, 5005*30], # 30, 60
            "lr_drop_mag": [10.0],
            "grad_clip": 1.0
        }
    elif script == "r2fasterr_redo_part_2": # resnet10->18 at start
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 60,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [0],
            "deepen_times": [5005*15],
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [5005*15, 5005*30],
            "lr_drop_mag": [10.0],
            "grad_clip": 1.0
        }
    elif script == "r2fasterr_redo_part_3": # resnet10
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 60,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 256,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [],
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [5005*15, 5005*30], # 30, 60
            "lr_drop_mag": [10.0],
            "grad_clip": 10.0
        }
    elif script == "r2fasterr_redo_part_4": # resnet18
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 60,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 256,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [],
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [5005*15, 5005*30],
            "lr_drop_mag": [10.0],
            "grad_clip": 10.0
        }



    ######
    # Final R2FasterR tests
    ######
    elif script == "f1aster_teacher":
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 45,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [],
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [20020*15, 20020*30],
            "lr_drop_mag": [10.0],
            "grad_clip": 10.0
        }
    elif script == "f2aster_student": 
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 45,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [0],
            "deepen_times": [0],
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [20020*15, 20020*30],
            "lr_drop_mag": [10.0],
            "grad_clip": 10.0
        }
    elif script == "f3aster_resnet50":
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 45,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [],
            "deepen_times": [],
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [20020*15, 20020*30],
            "lr_drop_mag": [10.0],
            "grad_clip": 10.0
        }
    elif script == "f4aster_r2r":
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 45,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [20020*30],
            "deepen_times": [20020*15],
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [20020*15, 20020*30],
            "lr_drop_mag": [10.0],
            "grad_clip": 10.0
        }
    elif script == "f5aster_r2r_adagrad":
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 45,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [20020*30],
            "deepen_times": [20020*15],
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [20020*15, 20020*30],
            "lr_drop_mag": [10.0],
            "grad_clip": 10.0
        }
    elif script == "f6aster_r2r_rms":
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 45,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [20020*30],
            "deepen_times": [20020*15],
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [20020*15, 20020*30],
            "lr_drop_mag": [10.0],
            "grad_clip": 10.0
        }
    elif script == "f7aster_r2r_adam":
        return {
            "lr": 0.1,
            "weight_decay": 1.0e-4,
            "epochs": 45,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 64,
            "workers": 6,
            "widen_times": [20020*30],
            "deepen_times": [20020*15],
            "flops_budget": 0, # unused
            "momentum": 0.9,
            "lr_drops": [20020*15, 20020*30],
            "lr_drop_mag": [10.0],
            "grad_clip": 10.0
        }




    ######
    # Last set of tests, using SVHN and proper resnet architectures
    ######
    elif script == "last_cwrt":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [391*50],#7], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "last_swrt":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 750,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [573*150], ### TOCHANCE
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script in ["last_ewrt", "last_ewrw"]:
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 6,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [int(4722*1.5)], ### TOCHANGE
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }

    elif script == "last_cdrt":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [391*50], ### TOCHANGE
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "last_sdrt":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 750,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [573*150], ### TOCHANGE
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script in ["last_edrt", "last_edrw"]:
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 6,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [int(4722*1.5)], ### TOCHANGE
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
        
    elif script in ["last_cwnt","last_cwnw"]:
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
        
    elif script in ["last_swnt","last_swnw"]:
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 750,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
        
    elif script in ["last_ewnt","last_ewnw"]:
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 12,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
        
    elif script in ["last_cdnt", "last_cdnw"]:
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
        
    elif script in ["last_sdnt", "last_sdnw"]:
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 750,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
        
    elif script in ["last_ednt", "last_ednw"]:
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 12,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    
    elif script == "last_cwr100":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [391*50],#7], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "last_cdr100":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [391*50], ### TOCHANGE
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "last_cwn100":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }
    elif script == "last_cdn100":
        return {
            "lr": 3.0e-3,
            "weight_decay": 1.0e-3,
            "epochs": 250,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
        }

    elif script == "last_svhn_wdtune":
        return {
            "lr": 3.0e-3,
            "weight_decay": 0.0,
            "epochs": 6,
            "tb_dir": tb_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "exp": exp_id,
            "batch_size": 128,
            "workers": 6,
            "widen_times": [], # unused
            "deepen_times": [], # unused
            "flops_budget": 0, # unused
            "momentum": 0.0, # unused
            "lr_drops": [], # unused
            "lr_drop_mag": [0.0], # unused,
            "grad_clip": 0.0
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
    parser.add_argument('--grad_clip', type=float, default=defaults['grad_clip'], help="A value to clip gradient norms to")

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

    # Things just for the imagenet redo tests
    parser.add_argument('--momentum', type=float, default=defaults["momentum"], help='Momentum in SGD optimizer.')
    parser.add_argument('--lr_drops', type=float, nargs='+', default=defaults["lr_drops"], help='Iterations to drop the learning rate on.')
    parser.add_argument('--lr_drop_mag', type=float, nargs='+', default=defaults["lr_drop_mag"], help='The amount to drop the learning rate by when it is dropped.')

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
    elif script == "cifar_identity_init": #########
        cifar_identity_init_test(args)

    elif script == "mnist_widen": #########
        mnist_widen_test(args)
    elif script == "cifar_widen": #########
        cifar_widen_test(args)
    elif script == "mnist_widen_with_budget":
        mnist_widen_with_budget_test(args)
    elif script == "cifar_widen_with_budget": #########
        cifar_widen_with_budget_test(args)

    elif script == "cifar_widen_multi_stage": # TODO: fix!!
        cifar_widen_test(args)
    elif script == "cifar_widen_with_budget_multi_stage": # TODO: fix!!
        cifar_widen_with_budget_test(args)

    elif script == "mnist_deepen": #########
        mnist_deepen_test(args)
    elif script == "cifar_deepen": #########
        cifar_deepen_test(args)
    elif script == "mnist_deepen_with_budget":
        mnist_deepen_with_budget_test(args)
    elif script == "cifar_deepen_with_budget": #########
        cifar_deepen_with_budget_test(args)

    elif script == "cifar_deepen_multi_stage": #########
        cifar_deepen_test(args)
    elif script == "cifar_deepen_with_budget_multi_stage": #########
        cifar_deepen_with_budget_test(args)

    elif script == "mnist_net2widernet_style": #########
        mnist_net_to_net_style_test(args)
    elif script == "cifar_net2widernet_style": #########
        cifar_net_to_net_style_test(args)

    elif script == "mnist_net2deepernet_style": #########
        mnist_net_to_net_style_test(args, widen=False)
    elif script == "cifar_net2deepernet_style": #########
        cifar_net_to_net_style_test(args, widen=False)

    elif script == "n2n_wider_inception":
        _net_2_wider_net_inception_test(args)
    elif script == "r2r_wider_inception":
        _r_2_wider_r_inception_test(args)


    #######
    # Weight visualizations tests
    #######
    elif script == "mnist_weight_viz_r2r":
        _mnist_weight_visuals(args, widen_method="r2r")
    elif script == "cifar_weight_viz_r2r":
        _cifar_weight_visuals(args, widen_method="r2r")
    elif script == "mnist_weight_viz_r2r_conv":
        _mnist_weight_visuals(args, widen_method="r2r", use_conv=True)
    elif script == "cifar_weight_viz_r2r_conv":
        _cifar_weight_visuals(args, widen_method="r2r", use_conv=True)
    elif script == "mnist_weight_viz_net2net":
        _mnist_weight_visuals(args, widen_method="net2net")
    elif script == "cifar_weight_viz_net2net":
        _cifar_weight_visuals(args, widen_method="net2net")
    elif script == "mnist_weight_viz_net2net_conv":
        _mnist_weight_visuals(args, widen_method="net2net", use_conv=True)
    elif script == "cifar_weight_viz_net2net_conv":
        _cifar_weight_visuals(args, widen_method="net2net", use_conv=True)
    elif script == "mnist_weight_viz_netmorph":
        _mnist_weight_visuals(args, widen_method="netmorph")
    elif script == "cifar_weight_viz_netmorph":
        _cifar_weight_visuals(args, widen_method="netmorph")
    elif script == "mnist_weight_viz_netmorph_conv":
        _mnist_weight_visuals(args, widen_method="netmorph", use_conv=True)
    elif script == "cifar_weight_viz_netmorph_conv":
        _cifar_weight_visuals(args, widen_method="netmorph", use_conv=True)
    elif script == "mnist_weight_viz":
        _mnist_weight_visuals(args, widen_method="netmorph", start_wide=True)
    elif script == "cifar_weight_viz":
        _cifar_weight_visuals(args, widen_method="netmorph", start_wide=True)
    elif script == "mnist_weight_viz_conv":
        _mnist_weight_visuals(args, widen_method="netmorph", use_conv=True, start_wide=True)
    elif script == "cifar_weight_viz_conv":
        _cifar_weight_visuals(args, widen_method="netmorph", use_conv=True, start_wide=True)

    #######
    # Net 2 Net Style tests, and R2R style tests
    #######
    elif script == "n2wn":
        net_2_wider_net_resnet(args)
    elif script == "n2dn":
        net_2_deeper_net_resnet(args)
    elif script == "n2wn_hps":
        net_2_wider_net_resnet_hyper_search(args)
    elif script == "n2dn_hps":
        net_2_deeper_net_resnet_hyper_search(args)
    elif script == "r2wr":
        r_2_wider_r_resnet(args)
    elif script == "r2dr":
        r_2_deeper_r_resnet(args)

    ######
    # Anomolies in training examples
    ######
    elif script == "n2n_overfit_problem":
        net_2_net_overfit_example(args)
    elif script == "r2r_weight_init_problem":
        r_2_r_weight_init_example(args)

    #######
    # Learning rate adaption tests
    #######
    elif script == "quad_widen":
        quadruple_widen_run(args)
    elif script == "double_deepen":
        double_deepen_run(args)
    elif script == "double_widen_deepen":
        double_widen_and_deepen_run(args)

    #######
    # Fast training tests, using resnet18
    #######
    elif script == "r2fasterr_part_1":
        r2r_faster_test_redo(args, "transforms_fixed_lr_drop_each_transform")
    elif script == "r2fasterr_part_2":
        r2r_faster_test_redo(args, "transforms_lr_drop_split_between_widen_and_deepen")
    elif script == "r2fasterr_part_3":
        r2r_faster_test_redo(args, "transforms_lr_schedule_unchanged")
    elif script == "r2fasterr_part_4":
        r2r_faster_test_redo_18(args, "student_arch")
    elif script == "r2fasterr_part_5":
        r2r_faster_test_redo(args, "teacher_arch")

    #######
    # Fast training tests, with resnet18
    #######
    elif script == "r2fasterr_redo_part_0":
        r2r_faster_test_redo(args, "widened_and_deepened")
    elif script == "r2fasterr_redo_part_1":
        r2r_faster_test_redo(args, "widened", optimizer='rms')
    elif script == "r2fasterr_redo_part_2":
        r2r_faster_test_redo(args, "deepened", optimizer='adagrad')
    elif script == "r2fasterr_redo_part_3":
        r2r_faster_test_redo(args, "teacher_arch")
    elif script == "r2fasterr_redo_part_4":
        r2r_faster_test_redo_18(args, "student_arch")
    # elif script == "r2fasterr_redo_part_6":
    #     r2r_faster_test_redo(args, "widened_attempt_three")
    # elif script == "r2fasterr_redo_part_7":
    #     r2r_faster_test_redo(args, "widened_attempt_three")
    # elif script == "r2fasterr_redo_part_8":
    #     r2r_faster_test_redo(args, "widened_attempt_four")


    ######
    # Final R2FasterR tests
    ######
    elif script == "f1aster_teacher":
        r2r_faster(args, shardname="teacher", optimizer='sgd', resnet_class=resnet35, use_thin=True)
    elif script == "f2aster_student":
        r2r_faster(args, shardname="student", optimizer='sgd', resnet_class=resnet35, use_thin=True, function_preserving=False) # widen at epoch 0
    elif script == "f3aster_resnet50":
        r2r_faster(args, shardname="resnet50", optimizer='sgd', resnet_class=resnet50, use_thin=False)
    elif script == "f4aster_r2r":
        r2r_faster(args, shardname="r2r", optimizer='sgd', resnet_class=resnet35, use_thin=True) # widen
    elif script == "f5aster_r2r_adagrad":
        r2r_faster(args, shardname="r2r_adagrad", optimizer='rms', resnet_class=resnet35, use_thin=True) # widen
    elif script == "f6aster_r2r_rms":
        r2r_faster(args, shardname="r2r_rms", optimizer='adagrad', resnet_class=resnet35, use_thin=True) # widen
    elif script == "f7aster_r2r_adam":
        r2r_faster(args, shardname="r2r_adam", optimizer='adam', resnet_class=resnet35, use_thin=True) # widen


    ######
    # Last set of tests, using SVHN and proper resnet architectures
    ######
    elif script == "last_cwrt":
        last_cifar_r2wider_resnet_thin(args)
    # elif script == "last_cwrw":
    #     last_cifar_r2wider_resnet_wide(args)
    elif script == "last_swrt":
        last_svhn_r2wider_resnet_thin(args)
    # elif script == "last_swrw":
    #     last_svhn_r2wider_resnet_wide(args)
    elif script == "last_ewrt":
        last_svhn_extended_r2wider_resnet_thin(args)
    elif script == "last_ewrw":
        last_svhn_extended_r2wider_resnet_wide(args)

    elif script == "last_cdrt":
        last_cifar_r2deeper_resnet_thin(args)
    # elif script == "last_cdrw":
    #     last_cifar_r2deeper_resnet_wide(args)
    elif script == "last_sdrt":
        last_svhn_r2deeper_resnet_thin(args)
    # elif script == "last_sdrw":
    #     last_svhn_r2deeper_resnet_wide(args)
    elif script == "last_edrt":
        last_svhn_extended_r2deeper_resnet_thin(args)
    elif script == "last_edrw":
        last_svhn_extended_r2deeper_resnet_wide(args)
        
    elif script == "last_cwnt":
        last_cifar_net2wider_resnet_thin(args)
    elif script == "last_cwnw":
        last_cifar_net2wider_resnet_wide(args)
    elif script == "last_swnt":
        last_svhn_net2wider_resnet_thin(args)
    elif script == "last_swnw":
        last_svhn_net2wider_resnet_wide(args)
    elif script == "last_ewnt":
        last_svhn_extended_net2wider_resnet_thin(args)
    elif script == "last_ewnw":
        last_svhn_extended_net2wider_resnet_wide(args)
        
    elif script == "last_cdnt":
        last_cifar_net2deeper_resnet_thin(args)
    elif script == "last_cdnw":
        last_cifar_net2deeper_resnet_wide(args)
    elif script == "last_sdnt":
        last_svhn_net2deeper_resnet_thin(args)
    elif script == "last_sdnw":
        last_svhn_net2deeper_resnet_wide(args)
    elif script == "last_ednt":
        last_svhn_extended_net2deeper_resnet_thin(args)
    elif script == "last_ednw":
        last_svhn_extended_net2deeper_resnet_wide(args)

    elif script == "last_cwr100":
        last_cifar100_r2wider_resnet_wide(args)
    elif script == "last_cdr100":
        last_cifar100_r2deeper_resnet_wide(args)
    elif script == "last_cwn100":
        last_cifar100_net2wider_resnet_wide(args)
    elif script == "last_cdn100":
        last_cifar100_net2deeper_resnet_wide(args)

    elif script == "last_svhn_wdtune":
        last_svhn_weight_decay_tune(args)


    else:
        print("Couldn't find script for '{s}'".format(s=script))
