#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np 

def create_local_init(glob, local, bias_ratio):
    assert bias_ratio <=1.0 and bias_ratio >= 0.0
    for k in glob.keys():
      if bias_ratio > 0:
        glob[k] = glob[k]*(1-bias_ratio) + local[k]*(bias_ratio)
      else:
        glob[k] = (glob[k]+local[k]*bias_ratio)/(1.0 + bias_ratio)
    return glob
          
          
def FedAvgM(w, gpu, w_org, mom, size_arr=None):
    (global_w, momentum) = w_org
    w_avg = {}
    for k in w[0].keys():
      w_avg[k] = torch.zeros(w[0][k].size())
    w_mom = dict(w_avg)
    
    # Prepare p 
    if size_arr is not None:
      total_num = np.sum(size_arr)
      size_arr = np.array([float(p)/total_num for p in size_arr])*len(size_arr)
    else:
      size_arr = np.array([1.0]*len(size_arr))

    for k in w_avg.keys():
      for i in range(0, len(w)):
        grad = global_w[k] - w[i][k] 
        w_avg[k] += size_arr[i]*grad
      
      mom_k = torch.div(w_avg[k], len(w))*(1-mom) + momentum[k]*mom
      w_avg[k] = global_w[k] - mom_k
      w_mom[k] = mom_k
          
    return w_avg, w_mom
