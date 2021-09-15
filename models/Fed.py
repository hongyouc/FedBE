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
          
      
def FedAvg(w, gpu, global_w=None, size_arr=None):
    w_avg = {}
    for k in w[0].keys():
      w_avg[k] = torch.zeros(w[0][k].size())
      
    # Prepare p 
    if size_arr is not None:
      total_num = np.sum(size_arr)
      size_arr = np.array([float(p)/total_num for p in size_arr])*len(size_arr)
    else:
      size_arr = np.array([1.0]*len(size_arr))

    if global_w is not None:
      for k in w_avg.keys():
          for i in range(0, len(w)):
            grad = w[i][k]
            grad_norm = torch.norm(grad, p=2) / torch.norm(global_w[k], p=2) 
            w_avg[k] += size_arr[i]*grad / grad_norm   
          w_avg[k] = torch.div(w_avg[k], len(w))  
    else:
      for k in w_avg.keys():
          for i in range(0, len(w)):
            w_avg[k] += size_arr[i]*w[i][k] 
          w_avg[k] = torch.div(w_avg[k], len(w))
          
    return w_avg
