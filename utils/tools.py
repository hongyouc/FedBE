import copy
import pdb
import os
import pickle  

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from scipy.stats import mode
from scipy.stats import entropy
from scipy.stats import entropy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


def store_model(iter, model_dir, w_glob_org, client_w_list):
    torch.save(w_glob_org, os.path.join(model_dir, "w_org_%d"%iter)) 
    for i in range(len(client_w_list)):
      torch.save(client_w_list[i], os.path.join(model_dir, "client_%d_%d"%(iter, i)))  

def adaptive_schedule(local_ep, total_ep, rounds, adap_ep):
  if rounds<5:
    running_ep = adap_ep
  else:
    running_ep = local_ep
  return running_ep

def lr_schedule(base_lr, iter, total_ep):
  if iter==0:
    return base_lr*0.5
    
  elif iter>total_ep*0.9:
    return base_lr* 0.01
        
  elif iter>total_ep*0.6: 
    return base_lr* 0.1
    
  elif iter>total_ep*0.3: 
    return base_lr* 0.2
    
  else:
    return base_lr
    
def get_entropy(logits):
    mean_entropy = np.mean([entropy(logit) for logit in logits])
    return mean_entropy
    
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label    
        
def get_input_logits(inputs, model, is_logit=False, net_org=None):
    model.eval()
    with torch.no_grad():
      logit = model(inputs).detach()
      if not is_logit:
        logit = F.softmax(logit, dim=1)
       
    logit = logit.cpu().numpy()
    return logit 
    
def temp_softmax(x, axis=-1, temp=1.0):
    x = x/temp
    e_x = np.exp(x - np.max(x)) # same code
    e_x = e_x / e_x.sum(axis=axis, keepdims=True)
    return e_x
    
def temp_sharpen(x, axis=-1, temp=1.0):
    x = np.maximum(x**(1/temp), 1e-8)
    return x / x.sum(axis=axis, keepdims=True)

    
def merge_logits(logits, method, loss_type, temp=0.3, global_ep=1000):
    if "vote" in method:
      if loss_type=="CE":
        votes = np.argmax(logits, axis=-1) 
        logits_arr = mode(votes, axis=1)[0].reshape((len(logits)))
        logits_cond = np.mean(np.max(logits, axis=-1), axis=-1)
      else:  
        logits = np.mean(logits, axis=1)
        logits_arr = temp_softmax(logits, temp=temp) 
        logits_cond = np.max(logits_arr, axis=-1)
    else:
      logits = np.mean(logits, axis=1)
      
      if loss_type=="MSE":
        logits_arr = temp_softmax(logits, temp=1)
        logits_cond = np.max(logits_arr, axis=-1)
      elif "KL" in loss_type: 
        logits_arr = temp_sharpen(logits, temp=temp)   
        logits_cond = np.max(logits_arr, axis=-1)
      else:
        logits_arr = logits
        logits_cond = softmax(logits, axis=-1)
        logits_cond = np.max(logits_cond, axis=-1)    

    return logits_arr, logits_cond  

def weights_init(m): 
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      torch.nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
          torch.nn.init.zeros_(m.bias)
          
class logger():
  def __init__(self, name):
    self.name = name
    self.loss_train_list = []
    self.loss_test_list = []
    
    
    self.train_acc_list = []
    self.test_acc_list = []
    self.val_acc_list = [] 
    self.loss_val_list = []
    
    self.ens_train_acc_list = []
    self.ens_test_acc_list = []
    self.ens_val_acc_list = []
    
    
    self.teacher_loss_train_list = []
    self.teacher_loss_test_list = []
    
    self.swa_train_acc_list=[]
    self.swa_test_acc_list=[]
    self.swa_val_acc_list = []
    
    self.swag_train_acc_list=[]
    self.swag_test_acc_list=[]
    self.swag_val_acc_list = []    

