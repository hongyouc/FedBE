"""
    implementation of SWAG
"""

import torch
import numpy as np
import itertools
from torch.distributions.normal import Normal
import copy
    
class SWAG_client(torch.nn.Module):
    def __init__(self, args, base_model, lr=0.01, max_num_models=25, var_clamp=1e-5, concentrate_num=1):
        self.base_model = base_model
        self.max_num_models=max_num_models
        self.var_clamp=var_clamp
        self.concentrate_num = concentrate_num
        self.args = args
        self.lr = lr
    
    def compute_var(self, mean, sq_mean): 
        var_dict = {}
        for k in mean.keys():
          var = torch.clamp(sq_mean[k] - (mean[k] ** 2), self.var_clamp) 
          var_dict[k] = var 

        return var_dict
        
    def construct_models(self, w):
      (w_avg, w_sq_avg, w_norm) = w
      self.w_var = self.compute_var(w_avg, w_sq_avg)
      
      mean_grad = {k:torch.zeros(w.size()) for k,w in w_avg.items()}
      
      for i in range(self.concentrate_num):
        for k in w_avg.keys():
          mean = w_avg[k]
          var = self.w_var[k]
          
          eps = torch.randn_like(mean)
          sample_grad = mean + torch.sqrt(var) * eps * self.args.var_scale
          mean_grad[k] += sample_grad
      
      for k in w_avg.keys():
        grad_length = w_norm[k]/float(self.concentrate_num)*self.args.client_stepsize 
        mean_grad[k] = mean_grad[k]*grad_length + self.base_model[k].cpu()
      
      self.w_avg = w_avg
      return mean_grad

class SWAG_server(torch.nn.Module):
    def __init__(self, args, base_model, avg_model=None, max_num_models=25, var_clamp=1e-5, concentrate_num=1, size_arr=None):
        self.base_model = base_model
        self.max_num_models=max_num_models
        self.var_clamp=var_clamp
        self.concentrate_num = concentrate_num
        self.args = args
        self.avg_model = avg_model
        self.size_arr = size_arr
         
    def compute_var(self, mean, sq_mean): 
        var_dict = {}
        for k in mean.keys():
          var = torch.clamp(sq_mean[k] - mean[k] ** 2, self.var_clamp) 
          var_dict[k] = var 

        return var_dict

    def compute_mean_sq(self, teachers):
        w_avg = {}
        w_sq_avg = {}
        w_norm ={}
        
        for k in teachers[0].keys():
          if "batches_tracked" in k: continue
          w_avg[k] = torch.zeros(teachers[0][k].size())
          w_sq_avg[k] = torch.zeros(teachers[0][k].size())
          w_norm[k] = 0.0 
          
        for k in w_avg.keys():
            if "batches_tracked" in k: continue
            for i in range(0, len(teachers)):
              grad = teachers[i][k].cpu()- self.base_model[k].cpu()
              norm = torch.norm(grad, p=2)
              
              grad = grad/norm
              sq_grad = grad**2
              
              w_avg[k] += grad
              w_sq_avg[k] += sq_grad
              w_norm[k] += norm
              
            w_avg[k] = torch.div(w_avg[k], len(teachers))
            w_sq_avg[k] = torch.div(w_sq_avg[k], len(teachers))
            w_norm[k] = torch.div(w_norm[k], len(teachers))
            
        return w_avg, w_sq_avg, w_norm
        
    def construct_models(self, teachers, mean=None, mode="dir"):
      if mode=="gaussian":
        w_avg, w_sq_avg, w_norm= self.compute_mean_sq(teachers)
        w_var = self.compute_var(w_avg, w_sq_avg)      
        
        mean_grad = copy.deepcopy(w_avg)
        for i in range(self.concentrate_num):
          for k in w_avg.keys():
            mean = w_avg[k]
            var = torch.clamp(w_var[k], 1e-6)
            
            eps = torch.randn_like(mean)
            sample_grad = mean + torch.sqrt(var) * eps * self.args.var_scale
            mean_grad[k] = (i*mean_grad[k] + sample_grad) / (i+1)
        
        for k in w_avg.keys():
          mean_grad[k] = mean_grad[k]*self.args.swag_stepsize*w_norm[k] + self.base_model[k].cpu()
          
        return mean_grad  
      
      elif mode=="random":
        num_t = 3
        ts = np.random.choice(teachers, num_t, replace=False)
        mean_grad = {}
        for k in ts[0].keys():
          mean_grad[k] = torch.zeros(ts[0][k].size())
          for i, t in enumerate(ts):
            mean_grad[k]+= t[k]
          
        for k in ts[0].keys():
          mean_grad[k]/=num_t  
          
        return mean_grad
      
      elif mode=="dir":
        proportions = np.random.dirichlet(np.repeat(self.args.alpha, len(teachers)))
        mean_grad = {}
        for k in teachers[0].keys():
          mean_grad[k] = torch.zeros(teachers[0][k].size())
          for i, t in enumerate(teachers):
            mean_grad[k]+= t[k]*proportions[i]
          
        for k in teachers[0].keys():
          mean_grad[k]/=sum(proportions)  

        return mean_grad   
        


