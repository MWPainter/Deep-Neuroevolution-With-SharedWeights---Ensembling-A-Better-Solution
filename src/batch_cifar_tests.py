### HACKY SCRIPT. This just does the same thing as in the Deriving resnet2resnet jupyter notebook.
### There's about 700 lines just c&p from the jupyter notebook

from __future__ import print_function
import random
import time
import math

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import copy

import os
import sys
dataset_module_path = os.path.abspath(os.path.join('..'))
if dataset_module_path not in sys.path:
    sys.path.append(dataset_module_path)
from dataset import MnistDataset
from dataset import CifarDataset
from dataset import DatasetCudaWrapper

from flops_utils import *

import pickle


#################
# Running tests #
#################

def save(dic, dic_name):
    with open("./results/%s.dat" % dic_name, "wb") as f:
        pickle.dump(dic, f)

def init_check_test():
    dataset = DatasetCudaWrapper(CifarDataset(64))
    
    our_init_results = []
    our_init_results_val = []
    our_init_time_updates = None
    our_init_time_flops = None
    
    dataset = DatasetCudaWrapper(CifarDataset(64))
    
    our_init_results = []
    our_init_results_val = []
    our_init_time_updates = None
    our_init_time_flops = None
    our_init_eval_times = []
    for i in range(5):
        print()
        print("Iter %d our init" % i)
        
        model = Cifar_Resnet_v1(identity_initialize=True, thin=False, deeper=True)
        model = model.cuda()
        start_time = time.time()
        train_acc, val_acc, our_init_time_updates, our_init_time_flops = evaluate(model, dataset)
        elapsed_time = time.time() - start_time
        del model

        our_init_results.append(train_acc)
        our_init_results_val.append(val_acc)
        our_init_eval_times.append(elapsed_time)

    our_init_label =  "Our init. No R2R. (%.1f +/- %.1f)" % (np.mean(our_init_eval_times), np.std(our_init_eval_times))
    our_init_eval_times = []
    for i in range(5):
        print()
        print("Iter %d our init" % i)
        
        model = Cifar_Resnet_v1(identity_initialize=True, thin=False, deeper=True)
        model = model.cuda()
        start_time = time.time()
        train_acc, val_acc, our_init_time_updates, our_init_time_flops = evaluate(model, dataset)
        elapsed_time = time.time() - start_time
        del model

        our_init_results.append(train_acc)
        our_init_results_val.append(val_acc)
        our_init_eval_times.append(elapsed_time)

    our_init_label =  "Our init. No R2R. (%.1f +/- %.1f)" % (np.mean(our_init_eval_times), np.std(our_init_eval_times))

    # run he initialization 5 times
    he_init_results = []
    he_init_results_val = []
    he_init_time_updates = None
    he_init_time_flops = None
    he_init_eval_times = []
    for i in range(5):
        print()
        print("Iter %d he init" % i)
        
        model = Cifar_Resnet_v1(identity_initialize=False, thin=False, deeper=True)
        model = model.cuda()
        start_time = time.time()
        train_acc, val_acc, he_init_time_updates, he_init_time_flops = evaluate(model, dataset)
        elapsed_time = time.time() - start_time
        del model
        he_init_results.append(train_acc)
        he_init_results_val.append(val_acc)
        he_init_eval_times.append(elapsed_time)
        
    he_init_label =  "He init. No R2R. (%.1f +/- %.1f)" % (np.mean(he_init_eval_times), np.std(he_init_eval_times))
        
        
    # Construct dict to save
    save_dict = {"our_init": our_init_results,
                 "our_init_val": our_init_results_val,
                 "our_init_time_updates": our_init_time_updates,
                 "our_init_time_flops": our_init_time_flops,
                 "our_init_label": our_init_label,
                 "he_init": he_init_results,
                 "he_init_val": he_init_results_val,
                 "he_init_time_updates": he_init_time_updates,
                 "he_init_time_flops": he_init_time_flops,
                 "he_init_label": he_init_label}
    save(save_dict, "init_check_test")
    
    
    
    
def padding_init_test():
    dataset = DatasetCudaWrapper(CifarDataset(64))
    
    # run xavier initialization 5 times
    x_init_results = []
    x_init_results_val = []
    x_init_time_updates = None
    x_init_time_flops = None
    for i in range(5):
        print()
        print("Iter %d xavier init" % i)
        
        model = Cifar_Resnet_v1(identity_initialize=True, deeper=True, widen_init_type="Xavier")
        model = model.cuda()
        train_acc, val_acc, x_init_time_updates, x_init_time_flops = evaluate(model, dataset, widen=1500, train_iters=3000)
        del model
        x_init_results.append(train_acc)
        x_init_results_val.append(val_acc)
        
    x_init_label =  "Xavier." 
    
    our_init_results = []
    our_init_results_val = []
    our_init_time_updates = None
    our_init_time_flops = None
    our_init_eval_times = []
    for i in range(5):
        print()
        print("Iter %d matching stddev" % i)
        
        model = Cifar_Resnet_v1(identity_initialize=True, deeper=True, widen_init_type="stddev_match")
        model = model.cuda()
        train_acc, val_acc, our_init_time_updates,our_init_time_flops = evaluate(model, dataset, widen=1500, train_iters=3000)
        del model

        our_init_results.append(train_acc)
        our_init_results_val.append(val_acc)

    our_init_label =  "Matching stddev."

    # run he initialization 5 times
    he_init_results = []
    he_init_results_val = []
    he_init_time_updates = None
    he_init_time_flops = None
    he_init_eval_times = []
    for i in range(5):
        print()
        print("Iter %d he extend" % i)
        
        model = Cifar_Resnet_v1(identity_initialize=True, deeper=True, widen_init_type="He")
        model = model.cuda()
        train_acc, val_acc, he_init_time_updates, he_init_time_flops = evaluate(model, dataset, widen=1500, train_iters=3000)
        del model
        he_init_results.append(train_acc)
        he_init_results_val.append(val_acc)
        
    he_init_label =  "He." 
        
        
    # Construct dict to save
    save_dict = {"our_init": our_init_results,
                 "our_init_val": our_init_results_val,
                 "our_init_time_updates": our_init_time_updates,
                 "our_init_time_flops": our_init_time_flops,
                 "our_init_label": our_init_label,
                 "he_init": he_init_results,
                 "he_init_val": he_init_results_val,
                 "he_init_time_updates": he_init_time_updates,
                 "he_init_time_flops": he_init_time_flops,
                 "he_init_label": he_init_label,
                 "x_init": x_init_results,
                 "x_init_val": x_init_results_val,
                 "x_init_time_updates": x_init_time_updates,
                 "x_init_time_flops": x_init_time_flops,
                 "x_init_label": x_init_label}
    save(save_dict, "padding_init_test")
    
    
    
    
def adapt_lr_test():
    dataset = DatasetCudaWrapper(CifarDataset(64))
    
    lr_change_results = []
    lr_change_results_val = []
    lr_change_time_updates = None
    lr_change_time_flops = None
    lr_change_eval_times = []
    for i in range(5):
        print()
        print("Iter %d adaptive lr" % i)
        
        model = Cifar_Resnet_v1(identity_initialize=True, deeper=True)
        model = model.cuda()
        train_acc, val_acc, lr_change_time_updates, lr_change_time_flops = evaluate(model, dataset, widen=1500, 
                                                                                    train_iters=3000, adapt_lr_rate=True)
        del model

        lr_change_results.append(train_acc)
        lr_change_results_val.append(val_acc)

    lr_change_label =  "Adaptive learning rate."

    
    no_lr_change_results = []
    no_lr_change_results_val = []
    no_lr_change_time_updates = None
    no_lr_change_time_flops = None
    no_lr_change_eval_times = []
    for i in range(5):
        print()
        print("Iter %d constant lr" % i)
        
        model = Cifar_Resnet_v1(identity_initialize=True, deeper=True)
        model = model.cuda()
        res = evaluate(model, dataset, widen=3000, adapt_lr_rate = False, train_iters=3000)
        train_acc, val_acc, no_lr_change_time_updates, no_lr_change_time_flops = res
        del model
        no_lr_change_results.append(train_acc)
        no_lr_change_results_val.append(val_acc)
        
    no_lr_change_label =  "Constant learning rate." 
    
    # Construct dict to save
    save_dict = {"lr_change": lr_change_results,
                 "lr_change_val": lr_change_results_val,
                 "lr_change_time_updates": lr_change_time_updates,
                 "lr_change_time_flops": lr_change_time_flops,
                 "lr_change_label": lr_change_label,
                 "no_lr_change": no_lr_change_results,
                 "no_lr_change_val": no_lr_change_results_val,
                 "no_lr_change_time_updates": no_lr_change_time_updates,
                 "no_lr_change_time_flops": no_lr_change_time_flops,
                 "no_lr_change_label": no_lr_change_label}
    save(save_dict, "lr_check_test")
    
    
    
    
def symmetry_test():
    dataset = DatasetCudaWrapper(CifarDataset(64))
    
    sym_results = []
    sym_results_val = []
    sym_time_updates = None
    sym_time_flops = None
    sym_eval_times = []
    for i in range(5):
        print()
        print("Iter %d symmetry breaking" % i)
        
        model = Cifar_Resnet_v1(identity_initialize=True, deeper=True, noise_ratio=1.0e-6)
        model = model.cuda()
        train_acc, val_acc, sym_time_updates, sym_time_flops = evaluate(model, dataset, widen=1500, 
                                                                                    train_iters=3000)
        del model

        sym_results.append(train_acc)
        sym_results_val.append(val_acc)

    sym_label =  "'Symmetry breaking'."

    
    no_sym_results = []
    no_sym_results_val = []
    no_sym_time_updates = None
    no_sym_time_flops = None
    no_sym_eval_times = []
    for i in range(5):
        print()
        print("Iter %d no symmetry breaking" % i)
        
        model = Cifar_Resnet_v1(identity_initialize=True, deeper=True)
        model = model.cuda()
        res = evaluate(model, dataset, widen=3000, adapt_lr_rate = False, train_iters=3000)
        train_acc, val_acc, no_sym_time_updates, no_sym_time_flops = res
        del model
        no_sym_results.append(train_acc)
        no_sym_results_val.append(val_acc)
        
    no_sym_label =  "No 'symmetry breaking'." 
    
    # Construct dict to save
    save_dict = {"sym": sym_results,
                 "sym_val": sym_results_val,
                 "sym_time_updates": sym_time_updates,
                 "sym_time_flops": sym_time_flops,
                 "sym_label": sym_label,
                 "no_sym": no_sym_results,
                 "no_sym_val": no_sym_results_val,
                 "no_sym_time_updates": no_sym_time_updates,
                 "no_sym_time_flops": no_sym_time_flops,
                 "no_sym_label": no_sym_label}
    save(save_dict, "sym_test")

    
    
    
    
def widen_test():
    widen_results = {}
    widen_results_val = {}
    widen_time_updates = {}
    widen_time_flops = {}
    widen_labels = {}

    dataset = DatasetCudaWrapper(CifarDataset(64))

    for widen in [1500,3000,4500]:
        results = []
        results_val = []
        times = []

        for i in range(5):
            print()
            print("Iter %d of widen at %d" % (i,widen))
            model = Cifar_Resnet_v1(thin=True)
            model = model.cuda()
            start_time = time.time()
            train_acc, val_acc, time_updates, time_flops = evaluate(model, dataset, widen=widen)
            elapsed_time = time.time() - start_time
            del model

            results.append(train_acc)
            results_val.append(val_acc)
            times.append(elapsed_time)
            
            widen_time_updates[widen] = time_updates
            widen_time_flops[widen] = time_flops

        widen_results[widen] = results
        widen_results_val[widen]= results_val
        widen_labels[widen] = "Widen at %d iters. (%.1f +/- %.1f)" % (widen, np.mean(times), np.std(times))
    
    # construct dict to save
    save_dict = {"widen_results": widen_results,
                 "widen_results_val": widen_results_val,
                 "widen_time_updates": widen_time_updates,
                 "widen_time_flops": widen_time_flops,
                 "widen_labels": widen_labels}
    save(save_dict, "widen_test")
    
    
    
    
    
def widen_convergence_test():
    widen_results = {}
    widen_results_val = {}
    widen_time_updates = {}
    widen_time_flops = {}
    widen_labels = {}

    dataset = DatasetCudaWrapper(CifarDataset(64))

    for widen in [0,1500,3000,4500]:
        results = []
        results_val = []
        times = []

        for i in range(5):
            print()
            print("Iter %d of widen at %d" % (i,widen))
            model = Cifar_Resnet_v1(thin=True)
            model = model.cuda()
            train_acc, val_acc, time_updates, time_flops = evaluate(model, dataset, widen=widen, train_iters=10000+widen)
            del model

            results.append(train_acc)
            results_val.append(val_acc)
            
            widen_time_updates[widen] = time_updates
            widen_time_flops[widen] = time_flops

        widen_results[widen] = results
        widen_results_val[widen]= results_val
        widen_labels[widen] = "Widen at %d iters." % widen
        if widen == 0: widen_labels[widen] = "No widening."
    
    # construct dict to save
    save_dict = {"widen_results": widen_results,
                 "widen_results_val": widen_results_val,
                 "widen_time_updates": widen_time_updates,
                 "widen_time_flops": widen_time_flops,
                 "widen_labels": widen_labels}
    save(save_dict, "widen_convergence_test")

    
    
    
def deepen_test():
    deepen_results = {}
    deepen_results_val = {}
    deepen_time_updates = {}
    deepen_time_flops = {}
    deepen_labels = {}

    dataset = DatasetCudaWrapper(CifarDataset(64))

    for deepen in [1500,3000,4500]:
        results = []
        results_val = []
        times = []

        for i in range(5):
            print()
            print("Iter %d of deepen at %d" % (i,deepen))
            model = Cifar_Resnet_v1(deeper=False)
            model = model.cuda()
            start_time = time.time()
            train_acc, val_acc, time_updates, time_flops = evaluate(model, dataset, deepen=deepen)
            elapsed_time = time.time() - start_time
            del model

            results.append(train_acc)
            results_val.append(val_acc)
            times.append(elapsed_time)
            
            deepen_time_updates[deepen] = time_updates
            deepen_time_flops[deepen] = time_flops

        deepen_results[deepen] = results
        deepen_results_val[deepen]= results_val
        deepen_labels[deepen] = "deepen at %d iters. (%.1f +/- %.1f)" % (deepen, np.mean(times), np.std(times))
        if deepen == 0: deepen_labels[deepen] = "No deepening."
    
    # construct dict to save
    save_dict = {"deepen_results": deepen_results,
                 "deepen_results_val": deepen_results_val,
                 "deepen_time_updates": deepen_time_updates,
                 "deepen_time_flops": deepen_time_flops,
                 "deepen_labels": deepen_labels}
    save(save_dict, "deepen_test")
    
    
    
def deepen_convergence_test():
    deepen_results = {}
    deepen_results_val = {}
    deepen_time_updates = {}
    deepen_time_flops = {}
    deepen_labels = {}

    dataset = DatasetCudaWrapper(CifarDataset(64))

    for deepen in [0,1500,3000,4500]:
        results = []
        results_val = []
        times = []

        for i in range(5):
            print()
            print("Iter %d of deepen at %d" % (i,deepen))
            model = Cifar_Resnet_v1(deeper=False)
            model = model.cuda()
            train_acc, val_acc, time_updates, time_flops = evaluate(model, dataset, deepen=deepen, train_iters=10000+deepen)
            del model

            results.append(train_acc)
            results_val.append(val_acc)
            
            deepen_time_updates[deepen] = time_updates
            deepen_time_flops[deepen] = time_flops

        deepen_results[deepen] = results
        deepen_results_val[deepen]= results_val
        deepen_labels[deepen] = "deepen at %d iters." % deepen
    
    # construct dict to save
    save_dict = {"deepen_results": deepen_results,
                 "deepen_results_val": deepen_results_val,
                 "deepen_time_updates": deepen_time_updates,
                 "deepen_time_flops": deepen_time_flops,
                 "deepen_labels": deepen_labels}
    save(save_dict, "deepen_convergence_test")
    
    
    
    
    
def grid_search_test():
    results = {}
    results_val = {}
    result_time_updates = {}
    result_time_flops = {}
    grid_labels = {}
    split_times = [0,1500,3000,4500]
    
    dataset = DatasetCudaWrapper(CifarDataset(64))
    
    for widen in split_times:
        results[widen] = {}
        results_val[widen] = {}
        result_time_updates[widen] = {}
        result_time_flops[widen] = {}
        grid_labels[widen] = {}

        for deepen in split_times:
            res = []
            res_val = []
            times = []

            for i in range(5):
                print()
                print("Iter %d of widen at %d/deepen at %d" % (i,widen,deepen))
                model = Cifar_Resnet_v1(deeper=False,thin=True)
                model = model.cuda()
                start_time = time.time()
                train_acc, val_acc, time_updates, time_flops = evaluate(model, dataset, widen=widen, deepen=deepen)
                elapsed_time = time.time() - start_time
                del model

                res.append(train_acc)
                res_val.append(val_acc)
                times.append(elapsed_time)
                
                result_time_updates[widen][deepen] = time_updates
                result_time_flops[widen][deepen] = time_flops

            results[widen][deepen] = res
            results_val[widen][deepen] = res_val
            grid_labels[widen][deepen] = "Widen: %d. Deepen: %d. (%.1f +/- %.1f)" % (widen, deepen, np.mean(times), np.std(times))
    
    # construct dict to save
    save_dict = {"results": results,
                 "results_val": results_val,
                 "result_time_updates": result_time_updates,
                 "result_time_flops": result_time_flops,
                 "grid_labels": grid_labels}
    save(save_dict, "grid_search_test")
             
        


def net_2_wider_net_tests():
    dataset = DatasetCudaWrapper(CifarDataset(64))
    
    init_model = Cifar_Resnet_v1(identity_initialize=True, deeper=True, thin=True)
    init_model = init_model.cuda()
    teacher_acc, teacher_val_acc, teacher_time_updates, _ = evaluate(init_model, dataset, train_iters=10000)
    init_model = init_model.cpu() # necessary to make deep copy work!
        
    our_pad_results = []
    our_pad_results_val = []
    our_pad_time_upates = None
    rand_pad_results = []
    rand_pad_results_val = []
    rand_pad_time_upates = None
    no_trans_results = []
    no_trans_results_val = []
    no_trans_time_upates = None
    
    for i in range(5):
        print()
        print("Iter %d our padding" % i)
        model = copy.deepcopy(init_model)
        weight_ratio = model.widen()
        new_lr = 3e-3 / weight_ratio
        model = model.cuda()
        train_acc, val_acc, our_pad_time_updates, _ = evaluate(model, dataset, init_lr=new_lr, train_iters=7000)
        our_pad_results.append(train_acc)
        our_pad_results_val.append(val_acc)

        print()
        print("Iter %d rand padding" % i)
        model = copy.deepcopy(init_model)
        model.fn_preserving_transform = False
        weight_ratio = model.widen()
        new_lr = 3e-3 / weight_ratio
        model = model.cuda()
        train_acc, val_acc, rand_pad_time_updates, _ = evaluate(model, dataset, init_lr=new_lr, train_iters=7000)
        rand_pad_results.append(train_acc)
        rand_pad_results_val.append(val_acc)
        
        print()
        print("Iter %d fresh model" % i)
        model = Cifar_Resnet_v1(identity_initialize=True, thin=False, deeper=True)
        model = model.cuda()
        train_acc, val_acc, no_trans_time_updates, _ = evaluate(model, dataset, init_lr=new_lr, train_iters=7000)
        no_trans_results.append(train_acc)
        no_trans_results_val.append(val_acc)
    
    our_pad_label = "R2WiderR."
    rand_pad_label = "Random padding."
    no_trans_label = "No knowledge transfer."
    
    
    # construct dict to save
    save_dict = {'teacher_results': teacher_acc,
                 'teacher_results_val': teacher_val_acc,
                 'teacher_time_updates': teacher_time_updates,
                 'our_pad_results': our_pad_results,
                 'our_pad_results_val': our_pad_results_val,
                 'our_pad_time_updates': our_pad_time_updates,
                 'our_pad_label': our_pad_label,
                 'rand_pad_results': rand_pad_results,
                 'rand_pad_results_val': rand_pad_results_val,
                 'rand_pad_time_updates': rand_pad_time_updates,
                 'rand_pad_label': rand_pad_label,
                 'no_trans_results': no_trans_results,
                 'no_trans_results_val': no_trans_results_val,
                 'no_trans_time_updates': no_trans_time_updates,
                 'no_trans_label': no_trans_label}
    save(save_dict, "r2widerr_test")
             
        


def net_2_deeper_net_tests():
    dataset = DatasetCudaWrapper(CifarDataset(64))
    
    init_model = Cifar_Resnet_v1(identity_initialize=True, deeper=False, thin=False)
    init_model = init_model.cuda()
    teacher_acc, teacher_val_acc, teacher_time_updates, teacher_time_flops = evaluate(init_model, dataset, train_iters=5000)
    init_model = init_model.cpu() # necessary to make deep copy work!
        
    our_pad_results = []
    our_pad_results_val = []
    our_pad_time_upates = None
    rand_pad_results = []
    rand_pad_results_val = []
    rand_pad_time_upates = None
    no_trans_results = []
    no_trans_results_val = []
    no_trans_time_updates = None
    
    for i in range(5):
        print()
        print("Iter %d our padding" % i)
        model = copy.deepcopy(init_model)
        weight_ratio = model.deepen()
        new_lr = 3e-3 / weight_ratio
        model = model.cuda()
        train_acc, val_acc, our_pad_time_updates, _ = evaluate(model, dataset, init_lr=new_lr, train_iters=7000)
        our_pad_results.append(train_acc)
        our_pad_results_val.append(val_acc)
        
        print()
        print("Iter %d rand padding" % i)
        model = copy.deepcopy(init_model)
        model.fn_preserving_transform = False
        weight_ratio = model.deepen()
        new_lr = 3e-3 / weight_ratio
        model = model.cuda()
        train_acc, val_acc, rand_pad_time_updates, _ = evaluate(model, dataset, init_lr=new_lr, train_iters=7000)
        rand_pad_results.append(train_acc)
        rand_pad_results_val.append(val_acc)
        
        print()
        print("Iter %d fresh model" % i)
        model = Cifar_Resnet_v1(identity_initialize=True, thin=False, deeper=True)
        model = model.cuda()
        train_acc, val_acc, our_pad_time_updates, _ = evaluate(model, dataset, init_lr=new_lr, train_iters=7000)
        no_trans_results.append(train_acc)
        no_trans_results_val.append(val_acc)
    
    our_pad_label = "R2DeeperR."
    rand_pad_label = "Random padding."
    no_trans_label = "No knowledge transfer."
    
    
    # construct dict to save
    save_dict = {'teacher_results': teacher_acc,
                 'teacher_results_val': teacher_val_acc,
                 'teacher_time_updates': teacher_time_updates,
                 'our_pad_results': our_pad_results,
                 'our_pad_results_val': our_pad_results_val,
                 'our_pad_time_updates': our_pad_time_updates,
                 'our_pad_label': our_pad_label,
                 'rand_pad_results': rand_pad_results,
                 'rand_pad_results_val': rand_pad_results_val,
                 'rand_pad_time_updates': rand_pad_time_updates,
                 'rand_pad_label': rand_pad_label,
                 'no_trans_results': no_trans_results,
                 'no_trans_results_val': no_trans_results_val,
                 'no_trans_time_updates': no_trans_time_updates,
                 'no_trans_label': no_trans_label}
    save(save_dict, "r2deeperr_test")
    
        




    
    
    
    
########################################################################
# Running the tests... (this is c&p at the end, and uncommented there) #
########################################################################
    
"""
if __name__ == "__main__":
    # pick which tests to run, use booleans to indicate
    run_init = True
    run_wider = True
    run_deepen = True
    run_grid = True
    
    if run_init:
        print("Running init check tests\n")
        init_check_test()
    
    if run_wider:
        print("Running widen tests\n")
        widen_test()
    
    if run_deeper:
        print("Running deepen tests\n")
        deepen_test()
    
    if run_grid:
        print("Running grid search tests\n")
        grid_search_test()
"""
    
    
    



















#########################
# Jupyter Notebook Code #
#########################

def _get_device(tensor):
    if tensor.is_cuda:
        return t.device('cuda')
    else:
        return t.device('cpu')

def _conv_xavier_initialize(filter_shape, override_input_channels=None, override_output_channels=None):
    """
    Initialize a convolutional filter, with shape 'filter_shape', according to "He initialization".
    The weight for each hidden unit should be drawn from a normal distribution, with zero mean and stddev of 
    sqrt(2/(n_in + n_out)).
    
    This is the initialization of choice for layers with non ReLU activations.
    
    The filter shape should be [output_channels, input_channels, width, height]. So here, n_in = input_channels 
    and n_out = width * height * output_channels.
    
    When "widening" an filter, from C1 output filters, to C1 + 2*C2 filters, then we want to initialize the 
    additional 2*C2 layers, as if there are C1+2*C2 filters in the output, and therefore we provide the 
    option to override the number of output filters.
    
    :param filter_shape: THe shape of the filter that we want to produce an initialization for
    :param override_output_channels: Override for the number of input filters in the filter_shape (optional)
    :param override_output_channels: Override for the number of output filters in the filter_shape (optional)
    :return: A numpy array, of shape 'filter_shape', randomly initialized according to He initialization.
    """
    out_channels, in_channels, width, height = filter_shape
    if override_input_channels is not None:
        in_channels = override_input_channels
    if override_output_channels is not None:
        out_channels = override_output_channels  
    
    scale = np.sqrt(2.0 / (in_channels + width*height*out_channels))
    return scale * np.random.randn(*filter_shape).astype(np.float32) 
    
    
    
    

def _conv_he_initialize(filter_shape, override_input_channels=None, override_output_channels=None):
    """
    Initialize a convolutional filter, with shape 'filter_shape', according to "Xavier initialization".
    Each weight for each hidden unit should be drawn from a normal distribution, with zero mean and stddev of 
    sqrt(2/n_in).
    
    This is the initialization of choice for layers with ReLU activations.
    
    The filter shape should be [output_channels, input_channels, width, height]. So here, n_in = input_channels.
    
    As the initization only depends on the number of inputs (the number of input channels), unlike Xavier 
    initialization, we don't need to be able to override the number of output_channels.
    
    :param filter_shape: THe shape of the filter that we want to produce an initialization 
    :param override_output_channels: Override for the number of input filters in the filter_shape (optional)
    :param override_output_channels: unused
    :return: A numpy array, of shape 'filter_shape', randomly initialized according to He initialization.
    """
    in_channels = filter_shape[1]
    if override_input_channels is not None:
        in_channels = override_input_channels
    scale = np.sqrt(2.0 / in_channels)
    return scale * np.random.randn(*filter_shape).astype(np.float32)





def _init_filter_with_repeated_output(extending_filter_shape, existing_filter=None, init_type='He'):
    """
    We want to initialize a filter with appropriately initialized weights.
    
    Let F be the 'existing_filter', with shape [C1,I,W,H]. Let extending_filter_shape be [2*C2,I,W,H]. 
    If the value for 2*C2 is odd or non-positive, then it's an error
    We then want to initialize E of shape [C2,I,W,H], according to the given initialization, 
    and then we want to return the concatenation [F;E;E].
    
    To make a fresh/new filter, with repeated weights, let 'existing_filter' be None, and it will 
    return just [E;rE], as F is "empty".
    
    :param extending_filter_shape: The shape of the new portion of the filter to return. I.e. [W,H,I,2*C2]
    :param existing_filter: If not None, it must have shape [W,H,I,C1], this is the existing filter.
    :param init_type: The type of initialization to use
    :return: A filter, extended by 2*C2 channels. I.e. the filter [F;E;E]
    """
    # Unpack params input
    twoC2, I, W, H = extending_filter_shape
    C2 = twoC2 // 2
    C1 = 0 if existing_filter is None else existing_filter.shape[0]
    
    # Error checking
    if twoC2 % 2 != 0:
        # TODO: log a descriptive error in final implementation
        raise Exception()
    elif existing_filter is not None and (W != existing_filter.shape[2] 
        or H != existing_filter.shape[3] or I != existing_filter.shape[1]):
        # TODO: log a descriptive error in final implementation
        raise Exception()
    
    # Canvas for the new numpy array that we want to return, and copy existing filter weights
    canvas = np.zeros((C1+twoC2, I, W, H)).astype(np.float32)
    if existing_filter is not None:
        canvas[:C1,:,:,:] = existing_filter

    # Initialize the new weights, and copy that into the canvas (twice)
    new_channels_weights = None
    if init_type == 'He':
        new_channels_weights = _conv_he_initialize((C2,I,W,H))
    elif init_type == 'Xavier':
        new_channels_weights = _conv_xavier_initialize((C2,I,W,H), C1+twoC2)
    else:
        # TODO: log a descriptive error in final implementation
        raise Exception()
    
    canvas[C1:C1+C2,:,:,:] = new_channels_weights
    canvas[C1+C2:C1+twoC2,:,:,:] = new_channels_weights

    # Done :)
    return canvas
    
        
        
class R2R_block_v1(nn.Module):
    """
    Defines a R2R module, version 1, this will be our first iteration and so we will make some (arbitrary) design choices for 
    now, just to make prototyping easier.
    
    An R2R module consists of the following network layers/operations:
        (optional) max pool
        k*k conv2d
        batchnorm
        activation function
        1*1 conv2d
        
    Outline of the computation that we perform:
        0. (Optionally) reduce the input dimensions, using a max pool
        1. Input shape of [D,A,A], and desired output shape is [2K,A,A]
        2. Make a filter, with shape [2K,D,4,4], where [:K,D,4,4] is identitcal to [K:,D,4,4], 
            that is: set the weights for [K+j,:,:,:] equal to  [j,:,:,:]
        3. Add noise, at an appropriate scale, to break symmetry
        4. Add batch norm, if we want to
        5. Apply our non-linearity, 
        6. By the symmetry introduced, the output after the non-linearity should be of the form [W'; W'] 
            (that is, still symmetric)
        7. Use a 1x1 convolution to use the symmetry to make the output from the previous layer cancel out 
            (note that this also allows use to provide dimensionality reduction, and, output a consistent number 
            of filters, helping to keep everything modular and all layers independent. (The next layer doesn't 
            need to change its input shape)).
        
    Really, this is just a glorified conv2d layer. We add the 1*1 conv2d so that we can provide a consistent output shape 
    from the R2R module. 
    
    We provide the option to initialize the module in such a way that the output is zero for ANY input.
    
    What is there left to do in the final version? 
        1. We don't actually want k*k conv2d's. We want to be able to mimic inception resnet v2, so will need to be able to 
           replace the k*k conv2d by a 1*7 and a 7*1 conv2d. This would most easily be done by passing in a nn.Module to 
           the constructor. Or maybe having an enum (we can decide later :))
        2. The final version may need to be a little more complex with the kernel sizing. Here we assume that its 3*3 and 
           therefore add a padding of 1 to preserve spatial dimensions. We don't necessarily want to preserve all spatial 
           dimensions in every module, nor do we always want a conv of 3*3
        3. The WHOLE point of this is that we can actually EXTEND the convolutions at runtime, but, preserve the function
           they represent. We've completely omitted this in v1, but, we want to sanity check our "zero initializations" first 
           so this R2R_v1 is still a good step.
        4. Provide more activation functions than just ReLUs
        5. We automatically use he initialization. We should provide an option to use Xavier (or any other inits we want)
    """
    def __init__(self, input_channels, intermediate_channels, output_channels, 
                 add_max_pool=False, add_batch_norm=True, zero_initialize=True, noise_ratio=1e-3):
        """
        Initializes each of the layers in the R2R module, and initializes the variables appropriately 
        
        :param input_channels: THe number of input channels provided to the conv2d layer
        :param intermediate_channels: The number of output channels from the first conv2d layer, and the number input to the 
                1*1 convolution
        :param output_channels: The number of channels output by the whole module
        :param add_max_pool: if we should add a max pool layer at the beginning
        :param add_batch_norm: if we should add a batch norm layer in the middle, before the activation function
        :param zero_initialize: should we initialize the module such that the output is always zero?
        :param noise_ratio: the amount of noise to add, as ratio (of the max init weight in the conv2d kernel) (break symmetry)
        """
        # Superclass initializer
        super(R2R_block_v1, self).__init__()
    
        # Make the layers
        self.max_pool = None
        if add_max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv = nn.Conv2d(input_channels, intermediate_channels, kernel_size=3, padding=1)
        self.batch_norm = None
        if add_batch_norm:
            self.batch_norm = nn.BatchNorm2d(num_features=intermediate_channels)
        self.activation_function = F.relu
        self.reduction = nn.Conv2d(intermediate_channels, output_channels, kernel_size=1)

        # If providing a zero initialization, do all of our repeated weight trickery!
        if zero_initialize:
            # Initialize the conv weights as appropriate, adding noise ass according to the 
            filter_shape = (intermediate_channels, input_channels, 3, 3)
            filter_init = _init_filter_with_repeated_output(filter_shape, init_type='He')
            
            max_entry = np.max(np.abs(filter_init))
            noise_scale = noise_ratio * max_entry
            filter_init += noise_ratio * np.random.randn(*filter_shape).astype(np.float32)
            
            self.conv.weight.data = Parameter(t.Tensor(filter_init))
            self.conv.bias.data *= 0.0
            
            # Initialize the batch norm variables so that the scale is one and the mean is zero
            if add_batch_norm:
                self.batch_norm.weight.data = Parameter(t.Tensor(t.ones(intermediate_channels)))
                self.batch_norm.bias.data = Parameter(t.Tensor(t.zeros(intermediate_channels)))

            # Intitialize the reduction convolution weights as appropriate 
            # (negate the input for the second set of (repeated) filters from the previous conv)
            filter_shape = (output_channels, intermediate_channels, 1, 1)
            filter_init = _conv_he_initialize(filter_shape)
            
            half_filters = intermediate_channels // 2
            filter_init[:, half_filters:] = - filter_init[:, :half_filters]
            
            self.reduction.weight.data = Parameter(t.Tensor(filter_init))
            self.reduction.bias.data *= 0.0
            
            
            
    def forward(self, x):
        """
        Forward pass of the module.
        
        :param x: The input tensor
        :return: The output from the module
        """
        if self.max_pool is not None: x = self.max_pool(x)
        x = self.conv(x)
        if self.batch_norm is not None: x = self.batch_norm(x)
        x = self.activation_function(x)
        x = self.reduction(x)
        return x
    
    
    


class R2R_residual_block_v1(nn.Module):
    """
    A small residual block to be used for mnist/cifar10 tests.
    
    It consists one set of convolutional layers (not multiple sizes of convolutions like Inception ResNet)
    It has the following architecture:
        Conv2D
        BatchNorm
        ReLU
        Conv2D
        BatchNorm
        ReLU
        R2R_block_v1         <- can chose to initialize so that the output is zero (and this block is an identity transform)
            Conv2D          
            BatchNorm
            ReLU
        
        + residual connection
    """
    def __init__(self, input_channels, output_channels, identity_initialize=True, noise_ratio=0.0):
        """
        Initialize the filters, optionally making this identity initialized.
        All convolutional filters have the same number of output channels
        """
        # Superclass initializer
        super(R2R_residual_block_v1, self).__init__()
        
        if output_channels < input_channels:
            # TODO: log an appropriate error message here
            raise Exception()
    
        self.input_channels = input_channels
        self.noise_ratio = noise_ratio
        
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channels)
        self.relu = F.relu
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=output_channels)
        
        self.r2r = R2R_block_v1(output_channels, output_channels, output_channels, 
                                zero_initialize=identity_initialize, noise_ratio=noise_ratio)
        
        
        
    def forward(self, x):
        """
        Forward pass through this residual block
        
        :param x: the input
        :return: THe output of applying this residual block to the input
        """
        # Forward pass through residual part of the network
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        res = self.relu(res)
        res = self.r2r(res)
        
        # add residual connection
        out = res
        out[:,:self.input_channels] += x 
        
        return out        



def _zero_pad_1d(old_val, new_params):
    """
    Zero pads an old tensor to match the new number of outputs
    
    :param old_val the old torch tensor to zero pad
    :param new_params: the number of new params needed
    :return: a new, padded tensor
    """
    old_len = old_val.size()[0]
    canvas = t.zeros((old_len+new_params,))
    canvas[:old_len] = old_val
    return canvas



def _one_pad_1d(old_val, new_params):
    """
    One pads an old tensor to match the new number of outputs
    
    :param old_val the old torch tensor to one pad
    :param new_params: the number of new params needed
    :return: a new, padded tensor
    """
    old_len = old_val.size()[0]
    canvas = t.ones((old_len+new_params,))
    canvas[:old_len] = old_val
    return canvas




def _conv_stddev_match_initialize(filter_shape, stddev):
    """
    Just initializes a filter with with a given stddev
    
    :param filter_shape: The shape of the np array to initialize (for a new kernel)
    :param stddev: The stddev to initialized the new kernel with
    :return: The new kernel, initialized with an appropriate stddev
    """
    return stddev * np.random.randn(*filter_shape).astype(np.float32)



def _widen_kernel_with_repeated_in_repeated_output(old_kernel, old_bias, extra_in_channels=0, 
                                                   extra_out_channels=0, repeated_in=True, 
                                                   repeated_out=True, init_type='stddev_match'):
    """
    A revised version of the logic to widen a kernel. We pass in the old kernel, the number of channels 
    extended in a convolution before it (zero if the convolution before it hasn't been widened), and 
    the number of new channels for the output.
    
    If the old shape was [Ou,I,W,H] and the new shape is [Ou2,I2,W,H], so Ou2 is the new number of output 
    channels and Ou is the old number of output channels, when we initialize as follows:
    
    Letting hOu = (Ou2-Ou)/2
    and hI = (I2-I)/2
    and r1, r2, r3 are appropriately randomly initialized 
    
        In \ Out |      0:Ou      |    Ou:Ou+hOu    |   Ou+hOu:Ou2    
        -------------------------------------------------------------
             0:I |  old[0:Ou,0:I] |  r1[0:hOu,0:I]  |  r1[0:hOu,0:I]  
        -------------------------------------------------------------       
          I:I+hI |  r2[0:Ou,0:hI] |  r3[0:hOu,0:hI] |  r3[0:hOu,0:hI] 
        -------------------------------------------------------------
         I+hI:I2 | -r2[0:Ou,0:hI] | -r3[0:hOu,0:hI] | -r3[0:hOu,0:hI]   
         
         
    Note, if we change either repeated_input or repeated_output, then, we no longer have a function preserving
    transfom. And we ignore the above grid.
    
    Init types = "He", "Xavier" and "stddev_match"
    recommended = stddev_match
    
    :param old_kernel: the old kernel of the convolution layer (of shape (A,B,W,H)) - a PyTorch tensor
    :param old_bias: the old bias of the convolution layer (of shape (A,)) - a PyTorch tensor
    :param extra_in_channels: the number of new channels being input to this layer (A2-A)
    :param extra_out_channels: the number of new channels being output from this layer (B2-B)
    :param repeated_in: if we want to handle repeated input (of NEW input channels)
    :param repeated_out: if we want to make the NEW output channels repeated 
    :param init_type: The type of initialization to use for the kernel
    :return: A new, larger filter and bias. (Initialized appropriately for a function preserving transform)
    """
    if extra_in_channels % 2 != 0 or extra_out_channels % 2 != 0:
            # TODO: log an appropraite error message
            raise Exception()
    
    # compute values related to kernel size
    Ou, I, W, H = old_kernel.size()
    hOu = extra_out_channels // 2
    hI = extra_in_channels // 2
    total_new_in_channels = I + extra_in_channels
    total_new_out_channels = Ou + extra_out_channels
    
    # init function
    init = None
    if init_type == "He":
        init = lambda shape: t.tensor(_conv_he_initialize(shape, total_new_in_channels, total_new_out_channels))
    elif init_type == "Xavier":
        init = lambda shape: t.tensor(_conv_xavier_initialize(shape, total_new_in_channels, total_new_out_channels))
    elif init_type == "stddev_match":
        weight_stddev = t.std(old_kernel).cpu().numpy() * 0.5
        init = lambda shape: t.tensor(_conv_stddev_match_initialize(shape, weight_stddev))
    else:
        # TODO: log an appropriate error message
        raise Exception()
    
    # compute r1, r2, r3 as above.
    if hOu > 0:
        r1 = init((hOu,  I, W, H)) 
    if hI > 0:
        r2 = init(( Ou, hI, W, H))
    if hOu > 0 and hI > 0:
        r3 = init((hOu, hI, W, H))
    
    # make a canvas and fill it appropriately
    # ignore repetitions appropriately if either negate_repeated_new_input, repeat_new_output is false
    canvas = t.zeros((total_new_out_channels, total_new_in_channels, W, H))
    
    # top left four squares
    canvas[:Ou, :I] = old_kernel
    if hOu > 0:
        canvas[Ou:Ou+hOu, :I] = r1
    if hI > 0:
        canvas[:Ou, I:I+hI] = r2
    if hOu > 0 and hI > 0:
        canvas[Ou:Ou+hOu, I:I+hI] = r3
    
    # bottom left two squares
    if hI > 0:
         canvas[:Ou, I+hI:I+2*hI] = -r2 if repeated_in else init((Ou, hI, W, H))
    if hOu > 0 and hI > 0:
        canvas[Ou:Ou+hOu, I+hI:I+2*hI] = -r3 if repeated_in else init((hOu, hI, W, H))
        
    # right three squares    
    if hOu > 0:
        canvas[Ou+hOu:Ou+2*hOu] = canvas[Ou:Ou+hOu] if repeated_out else init((hOu, I+2*hI, W, H))
        
    # Bias just needs to be zero padded appropriately
    return canvas, _zero_pad_1d(old_bias, extra_out_channels)







class R2R_residual_block_v2(R2R_residual_block_v1):
    """
    A small residual block to be used for mnist/cifar10 tests.
    
    Extends v1, and adds a widen operation.
    """
    def __init__(self, input_channels, output_channels, identity_initialize=True, noise_ratio=0.0):
        """
        Pass off to the superclass initializer
        """
        super(R2R_residual_block_v2, self).__init__(input_channels, output_channels, identity_initialize, noise_ratio)
        
        
    def widen(self, num_channels, function_preserving=True, init_type="stddev_match"):
        """
        Widens the network appropriately (widens each of the filters). This is slightly hacky, as really 
        we want to call self.r2r.widen() rather than manipulating the r2r block ourselves. We will do this 
        in the final version.
        
        :param num_channels: The number of channels to add to every 
        :param function_preserving: If the widen should be a function preserving transform
        :param init_type: The initialization to use for new parameters
        """ 
        # update new conv variable 
        self.conv1 = self._extend_conv(self.conv1, 0, num_channels, function_preserving,  # don't extend input!
                                       init_type=init_type) 
        self.conv2 = self._extend_conv(self.conv2, num_channels, num_channels, function_preserving,
                                       init_type=init_type)
        self.r2r.conv = self._extend_conv(self.r2r.conv, num_channels, num_channels, function_preserving,
                                       init_type=init_type)
        self.r2r.reduction = self._extend_conv(self.r2r.reduction, num_channels, 0,         # don't extend output!
                                       function_preserving, kernel_size=1, padding=0, init_type=init_type)
        
        
        # update new bn variables
        self.bn1 = self._extend_batch_norm(self.bn1, num_channels)
        self.bn2 = self._extend_batch_norm(self.bn2, num_channels)
        self.r2r.batch_norm = self._extend_batch_norm(self.r2r.batch_norm, num_channels)
    
    
    def _extend_conv(self, conv, extra_channels_in=0, extra_channels_out=0, function_preserving=True, 
                     kernel_size=3, padding=1, init_type="stddev_match"):
        """
        Combines all of the logic to extend conv, new convolution module. We create a new conv module, and 
        initialize it so that the filter is appropriately 'extended' from the old convolution module, 
        appropriately implementing the widening transform
        
        :param conv: The nn.Module convolution subclass instance to extend
        :param extra_channels_in: The number of extra channels in when widening the convolution
        :param extra_channels_out: The number of extra channels out when widening the convolution
        :param function_preserving: Should the convolution be extended in such a way that it preserves the function
        :param init_type: The type of initialization to use
        :returns: A new convolution, which is the result of this widen transform
        """
        # make the new conv module
        new_conv = nn.Conv2d(conv.in_channels+extra_channels_in, 
                             conv.out_channels+extra_channels_out, 
                             kernel_size=kernel_size, 
                             padding=padding)
        
        # shorthand, for long function name
        widen = _widen_kernel_with_repeated_in_repeated_output
        
        # Compute new kernel and bias
        new_kernel, new_bias = widen(old_kernel = conv.weight.data, 
                                     old_bias = conv.bias.data, 
                                     extra_in_channels = extra_channels_in,
                                     extra_out_channels = extra_channels_out,
                                     repeated_in=function_preserving, 
                                     repeated_out=function_preserving,
                                     init_type=init_type)
        
        # Add noise
        max_entry = t.max(t.abs(new_kernel))
        noise_scale = self.noise_ratio * max_entry
        noise = t.tensor(np.random.randn(*(new_kernel.size())).astype(np.float32))
        if new_kernel.is_cuda:
            noise = noise.cuda()
        new_kernel += self.noise_ratio * noise
        
        # Assign 
        new_conv.weight.data = Parameter(new_kernel)
        new_conv.bias.data = Parameter(new_bias)
        
        return new_conv
    
        
    def _extend_batch_norm(self, bn, num_channels):
        """
        Combines all of the logic to IN PLACE extend the 'bn' batch norm nn.Module appropriately to 
        extend a batch norm layer, and reset all of it's params appropriately for the widening transform.
        
        :param bn: The nn.Module batch norm subclass instance to extend
        :param num_channels: The number of extra channels (input and output) from this layer
        """
        # new batch norm module
        new_bn = nn.BatchNorm2d(num_features=bn.num_features + num_channels)
        
        # Zero/One pad the each of the sets of parameters maintained by batch norm layers appropriately
        new_scale = _one_pad_1d(bn.weight.data, num_channels)
        new_shift = _zero_pad_1d(bn.bias.data, num_channels)
        new_running_mean = _zero_pad_1d(bn.running_mean.data, num_channels)
        new_running_var = _zero_pad_1d(bn.running_var.data, num_channels)
        
        # Assign in the Module appropriately
        new_bn.weight.data = Parameter(new_scale)
        new_bn.bias.data = Parameter(new_shift)
        new_bn.running_mean.data = Parameter(new_running_mean)
        new_bn.running_var.data = Parameter(new_running_var)
        
        return new_bn
         
        
        
        
        
        
        
        

class Cifar_Resnet_v1(nn.Module):
    def __init__(self, identity_initialize=True, thin=False, deeper=True, noise_ratio=0.0, fn_preserving_transform=True,
                widen_init_type="stddev_match"):
        # Superclass initializer
        super(Cifar_Resnet_v1, self).__init__()
        
        # Channel sizes to use across layers (widen will increase this if necessary)
        self.channel_sizes = [3, 16, 32, 64]
        self.linear_hidden_units = 128
        self.noise_ratio = noise_ratio
        self.identity_initialize = identity_initialize
        self.fn_preserving_transform = fn_preserving_transform
        self.widen_init_type=widen_init_type
        
        # set deeper and thin, these will be update by the self.widen and self.deepen calls if necessary
        self.deeper = False
        self.thin = True
        
        # Make the three conv layers, with three max pools        
        self.resblock1 = R2R_residual_block_v2(input_channels=self.channel_sizes[0], 
                                               output_channels=self.channel_sizes[1], 
                                               identity_initialize=identity_initialize,
                                               noise_ratio=noise_ratio) # [-1, 32, 32, 32]
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                        # [-1, 32, 16, 16]  
        self.resblock2 = R2R_residual_block_v2(input_channels=self.channel_sizes[1], 
                                               output_channels=self.channel_sizes[2], 
                                               identity_initialize=identity_initialize,
                                               noise_ratio=noise_ratio) # [-1, 32, 16, 16]
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                        # [-1, 64, 8, 8]  
#         self.resblock3 = R2R_residual_block_v2(input_channels=self.channel_sizes[2], 
#                                                output_channels=self.channel_sizes[3], 
#                                                identity_initialize=identity_initialize,
#                                                noise_ratio=noise_ratio) # [-1, 64, 8, 8]
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                        # [-1, 128, 4, 4]
        
        # fully connected out
        self.linear1 = nn.Linear(4*4*self.channel_sizes[3], self.linear_hidden_units)
        self.linear2 = nn.Linear(self.linear_hidden_units, 10)
        
        # Deepen and widen now if necessary
        if deeper:
            self.deepen()
        if not thin:
            self.widen()
        
    
    def forward(self, x):
        # convs
        x = self.resblock1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.pool2(x)
        if self.deeper:
            x = self.resblock3(x)
        else:
            # if ignoring 3rd block, then make it an identity (v. hacky here...)
            buf = t.zeros((x.size()[0], self.channel_sizes[3], 8, 8), device=_get_device(x))
            buf[:,:x.size()[1]] = x
            x = buf
            
        x = self.pool3(x)
        
        # fc
        x = x.view((-1, 4*4*self.channel_sizes[3]))
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        
        return x
    
    
    def widen(self, init_type="stddev_match"):
        # Num params before
        old_num_params = self._count_params()
        
        # Set that we're no longer thing
        self.thin = False
        
        # Pass on the widen call to all of the constituent residual blocks
        self.resblock1.widen(self.channel_sizes[1], function_preserving=self.fn_preserving_transform, 
                             init_type=self.widen_init_type)
        self.resblock2.widen(self.channel_sizes[2], function_preserving=self.fn_preserving_transform, 
                             init_type=self.widen_init_type)
        if self.deeper: 
            self.resblock3.widen(self.channel_sizes[3], function_preserving=self.fn_preserving_transform, 
                                 init_type=self.widen_init_type)
        
        # Compute the ratio of parameters
        num_params = self._count_params()
        weight_ratio = num_params / old_num_params
        return weight_ratio
        
        
        
    def deepen(self):
        # Num params before
        old_num_params = self._count_params()
        
        # Set that we're no longer shallow
        self.deeper = True
        
        # Work out if we've widened yet or not
        in_channels = self.channel_sizes[2]
        out_channels = self.channel_sizes[3]
        if not self.thin:
            in_channels *= 2
            out_channels *= 2
        
        # Perform the deepening
        self.resblock3 = R2R_residual_block_v2(input_channels=self.channel_sizes[2], 
                                               output_channels=self.channel_sizes[3], 
                                               identity_initialize=self.fn_preserving_transform,
                                               noise_ratio=self.noise_ratio) # [-1, 64, 8, 8]
        
        # Compute the ratio of parameters
        num_params = self._count_params()
        weight_ratio = num_params / old_num_params 
        return weight_ratio
    
    
    
    def _add_modules_to_optimizer(self, modules, optimizer):
        """
        Optimizer is a non optional parameter, but we pass none in from the initializer. It should never 
        be None if we are calling widen/deepen from outside of the module. We allow it to be None for when 
        its called (from the initializer) before an Optimizer is created.
        """
        if optimizer is not None:
            for module in modules:
                for param_group in module.parameters():
                    optimizer.add_param_group({'params': param_group})
    
    
    def _count_params(self):
        """
        Comput the number of parameters
        
        :return: the number of parameters
        """
        total_num_params = 0
        for parameter in self.parameters():
            num_params = np.prod(t.tensor(parameter.size()).numpy())
            total_num_params += num_params
        return total_num_params
    
    
    def _magnitude(self):
        """
        Compute the number of parameters, and the average magnitude of the parameters
        Uses a (probably over complicated) running average
        
        :return: Numbers of parameters, and, mean *magnitude* of parameters
        """
        total_num_params = 0
        params_mean_mag = 0.0
        for parameter in self.parameters():
            num_params = np.prod(t.tensor(parameter.size()).numpy())
            ratio = num_params / (total_num_params + num_params)
            params_mean_mag = ratio * np.mean(np.abs(parameter.data.cpu().numpy())) + (1.0 - ratio) * params_mean_mag
            total_num_params += num_params
        return (total_num_params, params_mean_mag)
                    
                    

# compute the accuracy of a prediction
def accuracy(prediction, target):
    _, pred_classes = t.max(prediction, 1)
    _, actual_classes = t.max(target, 1)
    return t.mean((pred_classes == actual_classes).type(t.float))


# Compute the flops of a model
def model_flops(model, dataset):
    xs, _ = dataset.next_batch(32)
    model = add_flops_counting_methods(model)
    model.start_flops_count()
    _ = model(xs)
    flops = model.compute_average_flops_cost()
    model.stop_flops_count() # remove side effects
    return flops

# mutating the model, returns a new model, learning rate and optimizer...
# this got pretty complicated, using logic to maintain the update magnitude between 
def mutate_model(model, dataset, lr, optimizer, loss_fn, i, widen, deepen, weight_decay, adapt_lr_rate):
    # Compute bias portion of the adam momentum term (typically the same for all params)
    adam_momentum_term = 1.0
#     if i > 0:
#         for group in optimizer.param_groups:
#             for p in group['params']:
#                 state = optimizer.state[p]
#                 beta1, beta2 = group['betas']
#                 bias_correction1 = 1 - beta1 ** state['step']
#                 bias_correction2 = 1 - beta2 ** state['step']
#                 adam_momentum_term = math.sqrt(bias_correction2) / bias_correction1
#                 break
#             break

    # Transform
    weight_ratio = 1.0
    if i == widen:
        weight_ratio = model.widen()
    if i == deepen:
        weight_ratio = model.deepen()
    model = model.cuda()

    # Update the learning rate (using a ratio of update magnitudes) and make a new optimizer
    if adapt_lr_rate:
        lr *= adam_momentum_term / weight_ratio 
        weight_decay *= adam_momentum_term / weight_ratio
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

    # return 
    return model, lr, weight_decay, optimizer

# Evaluate a model
def evaluate(model, dataset, widen=-1, deepen=-1, weight_decay=1e-6, train_iters=6000, init_lr=3e-3, adapt_lr_rate=True):
    # Setup bookeeping and optimizer
    epoch_len = 10
    train_accuracies = []
    val_accuracies = []
    time_updates = []
    time_flops = []
    flops_per_update = model_flops(model, dataset)
    
    loss_fn = nn.BCEWithLogitsLoss()
    lr = init_lr
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    
    # train for 300 iterations, evaluating every epoch
    for i in range(train_iters+1):
        # if need to widen/deepen, apply the appropriate R2R transform (and recompute flops per update)
        if i ==  widen or i == deepen:
            ret = mutate_model(model, dataset, lr, optimizer, loss_fn, i, widen, deepen, weight_decay, adapt_lr_rate)
            model, lr, weight_decay, optimizer = ret
            flops_per_update = model_flops(model, dataset)
        
        # compute loss
        xs, ys = dataset.next_batch(32)
        ys_pred = model(xs)
        loss = loss_fn(ys_pred, ys)
        
        # make a step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # log at the beginning/end of every epoch
        if i % epoch_len == 0:
            # training accuracy
            train_acc = accuracy(ys_pred, ys)
            
            # validation accuracy
            xs, ys = dataset.next_val_batch(64)
            ys_pred = model(xs)            
            val_acc = accuracy(ys_pred, ys)
            
            # xaxis for plotting
            time_updates.append(i)
            if len(time_flops) == 0:
                time_flops.append(0)
            else:
                time_flops.append(time_flops[-1] + epoch_len * flops_per_update)
            
            # log accuracies
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            print("(Iter %d). Current train: %0.6f, Current val: %0.6f." % (i, train_acc, val_acc))
            
    return train_accuracies, val_accuracies, time_updates, time_flops











#####################
# Running the tests #
#####################
if __name__ == "__main__":
    # pick which tests to run, use booleans to indicate
    run_init = False
    run_padding_init = False
    run_adaptive_lr = False
    run_sym = False
    run_widen = False
    run_widen_convergence = False
    run_deepen = False
    run_deepen_convergence = False
    run_grid = False
    run_r2wr = False
    run_r2dr = True
    
    if run_init:
        print("\nRunning init check tests")
        init_check_test()
    
    if run_padding_init:
        print("\nRunning padding init check tests")
        padding_init_test()
        
    if run_adaptive_lr:
        print("\nRunning adaptive lr tests")
        adapt_lr_test()
        
    if run_sym:
        print("\nRunning symmetry breaking tests")
        symmetry_test()
    
    if run_widen:
        print("\nRunning widen tests")
        widen_test()
        
    if run_widen_convergence:
        print("\nRunning widen convergence tests")
        widen_convergence_test()
    
    if run_deepen:
        print("\nRunning deepen tests")
        deepen_test()
        
    if run_deepen_convergence:
        print("\nRunning deepen convergence tests")
        deepen_convergence_test()
    
    if run_grid:
        print("\nRunning grid search tests")
        grid_search_test()
        
    if run_r2wr:
        print("\nRunning R2WiderR tests")
        net_2_wider_net_tests()
        
    if run_r2dr:
        print("\nRunning R2WiderR tests")
        net_2_deeper_net_tests()
