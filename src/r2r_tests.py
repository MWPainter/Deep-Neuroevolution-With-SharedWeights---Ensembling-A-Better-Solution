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





def _conv_stddev_match_initialize(filter_shape, stddev):
    """
    Just initializes a filter with with a given stddev
    
    :param filter_shape: The shape of the np array to initialize (for a new kernel)
    :param stddev: The stddev to initialized the new kernel with
    :return: The new kernel, initialized with an appropriate stddev
    """
    return stddev * np.random.randn(*filter_shape).astype(np.float32)



         
        
# compute the accuracy of a prediction
def accuracy(prediction, target):
    _, pred_classes = t.max(prediction, 1)
    _, actual_classes = t.max(target, 1)
    return t.mean((pred_classes == actual_classes).type(t.float))


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
