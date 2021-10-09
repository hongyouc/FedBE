#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random, pdb, os
from sklearn import metrics
import torch.nn.functional as F
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from models.test import test_img
from torchvision import datasets, transforms

from scipy.stats import mode
from utils.tools import *
from sklearn.utils import shuffle
from PIL import Image

import torch.multiprocessing as mp
from models.swa import SWA 
   
class SWAGLocalUpdate(object):
    def __init__(self, args, device, dataset=None, idxs=None, server_ids=None, test=(None, None), num_per_cls=None):
        self.args = args
        self.device = device
        self.num_per_cls = num_per_cls
        
        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        (self.test_dataset, self.test_ids) = test
        (self.train_dataset, self.user_train_ids) = (dataset, idxs)
        
        self.server_ids = server_ids

    def apply_weight_decay(self, *modules, weight_decay_factor=0., wo_bn=True):
        '''
        https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/5
        Apply weight decay to pytorch model without BN;
        In pytorch:
            if group['weight_decay'] != 0:
                grad = grad.add(p, alpha=group['weight_decay'])
        p is the param;
        :param modules:
        :param weight_decay_factor:
        :return:
        '''
        for module in modules:
            for m in module.modules():
                if hasattr(m, 'weight'):
                    if wo_bn and isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        continue
                    m.weight.grad += m.weight * weight_decay_factor
                    
    def reg_loss(self, net, grad_org):         
      if self.args.reg_type == "FedProx":
        reg_loss = 0.0
        for name, param in net.named_parameters():
          if 'weight' in name:
            reg_loss += torch.norm(param-grad_org[name].to(self.device), 2)    
        reg_loss = reg_loss*0.5*self.args.mu
      return reg_loss
        
    def train(self, net, running_ep, lr): 
        net.cpu()
        grad_org = copy.deepcopy(net.state_dict())
        net.to(self.device)
        net.train()
        
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)      
        if self.args.ens:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                      step_size=30,
                                                      gamma=0.1)
        epoch_loss = []
        acc = 0.0

        num_model = 0
        cnt = 0
        for iter in range(running_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                
                if self.args.reg_type == "FedProx":
                  reg_loss = self.reg_loss(net, grad_org)    
                  loss += reg_loss
                
                loss.backward()
                self.apply_weight_decay(net, weight_decay_factor=self.args.weight_decay)
                optimizer.step()
                  
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())   
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            if self.args.ens:
              lr_scheduler.step()
              
        net = net.cpu()    
        return net

class ServerUpdate(object):
    def __init__(self, args, device, dataset=None, 
                 server_dataset=None, server_idxs=None, train_idx=None, 
                 test=(None, None), 
                 w_org=None, base_teachers=None):
                 
        self.args = args
        self.device = device
        self.loss_type = args.loss_type
        self.loss_func = nn.KLDivLoss() if self.loss_type =="KL" else nn.CrossEntropyLoss()
        self.selected_clients = []
        
        self.server_data_size = len(server_idxs)
        self.aug = args.aug and args.use_SWA
        self.ldr_train = DataLoader(DatasetSplit(dataset, server_idxs), batch_size=1024, shuffle=False)
        self.ldr_local_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=self.args.server_bs, shuffle=False)
        self.test_dataset = DataLoader(test[0], batch_size=self.args.server_bs, shuffle=False) 
        self.aum_dir = os.path.join(self.args.log_dir, "aum")  
        
        server_train_dataset = DataLoader(DatasetSplit(server_dataset, server_idxs), batch_size=self.args.server_bs, shuffle=False)
        self.server_train_dataset = [images for images, labels in server_train_dataset]
        
        self.w_org = w_org
        self.base_teachers = base_teachers
        
        # Get one batch for testing
        (self.eval_images, self.eval_labels) = next(iter(self.ldr_train))

    def transform_train(self, images):
        images = random_crop(images, 4)
        images = torch.Tensor(images).cuda()
        return images 

    def get_ensemble_logits(self, teachers, inputs, method='mean', global_ep=1000):
        logits = np.zeros((len(teachers), len(inputs), self.args.num_classes))
        for i, t_net in enumerate(teachers):
          logit = get_input_logits(inputs, t_net.cuda(), is_logit = self.args.is_logit) #Disable res
          logits[i] = logit
          
        logits = np.transpose(logits, (1, 0, 2)) # batchsize, teachers, 10
        logits_arr, logits_cond = merge_logits(logits, method, self.args.loss_type, temp=self.args.temp, global_ep=global_ep)
        batch_entropy = get_entropy(logits.reshape((-1, self.args.num_classes)))
        return logits_arr, batch_entropy

    def eval_ensemble(self, teachers, dataset):
        acc = 0.0
        cnt = 0
        
        if self.args.soft_vote:
          num_votes_list, soft_vote = get_aum(self.args, teachers, dataset)
          for batch_idx, (_, labels) in enumerate(dataset):
              logits = soft_vote[batch_idx]
              logits=np.argmax(logits, axis=-1)
              acc += np.sum(logits==labels.numpy())
              cnt += len(labels)            

        else:
          for batch_idx, (images, labels) in enumerate(dataset):
              images = images.cuda()
              logits, _ = self.get_ensemble_logits(teachers, images, method=self.args.logit_method, global_ep=1000)
              
              if self.args.logit_method != "vote":
                logits=np.argmax(logits, axis=-1)

              acc += np.sum(logits==labels.numpy())
              cnt += len(labels)

        return float(acc)/cnt*100.0

    def loss_wrapper(self, log_probs, logits, labels):        
        # Modify target logits
        if self.loss_type=="CE":
          if self.args.logit_method != "vote":
            logits = np.argmax(logits, axis=-1)
          acc_cnt=np.sum(logits==labels)
          cnt=len(labels)
          logits = torch.Tensor(logits).long().cuda(non_blocking=True)  
            
        else:  
          acc_cnt=np.sum(np.argmax(logits, axis=-1)==labels)
          cnt=len(labels)
          logits = torch.Tensor(logits).cuda(non_blocking=True)  


        # For loss function
        if self.args.use_oracle:
          loss = nn.CrossEntropyLoss()(log_probs, torch.Tensor(labels).long().cuda())
        else:      
          if "KL" in self.loss_type:
            log_probs = F.softmax(log_probs, dim=-1)
            if self.loss_type== "reverse_KL":
              P = log_probs  
              Q = logits                
            else:
              P = logits 
              Q = log_probs
            
            one_vec = (P * (P.log() - torch.Tensor([0.1]).cuda(non_blocking=True).log()))
            loss = (P * (P.log() - Q.log())).mean()
          else:
            loss = self.loss_func(log_probs, logits)
            
        return loss, acc_cnt, cnt     
    
    def test_net(self, tmp_net):
        tmp_net = tmp_net.cuda()
        (input, label) = (self.eval_images.cuda(), self.eval_labels.cuda())
        log_probs = tmp_net(input)
        loss = nn.CrossEntropyLoss()(log_probs, label)
        return not torch.isnan(loss)

    def record_teacher(self, ldr_train, net, teachers, global_ep, log_dir=None, probe=True, resample=False):
        entropy = []  
        ldr_train = []

        acc_per_teacher = np.zeros((len(teachers)))
        conf_per_teacher = np.zeros((len(teachers)))
        teacher_per_sample = 0.0
        has_correct_teacher_ratio = 0.0
        
        num = self.server_data_size
        if "cifar" in self.args.dataset:  
          imgsize = 32 
        elif "mnist" in self.args.dataset:    
          imgsize = 28
          
        channel = 1 if self.args.dataset == "mnist" else 3        
        all_images = np.zeros((num, channel, imgsize, imgsize))
        all_logits = np.zeros((num, self.args.num_classes))
        all_labels = np.zeros((num))
        cnt = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            logits, batch_entropy = self.get_ensemble_logits(teachers, images.cuda(), method=self.args.logit_method, global_ep=global_ep)
            entropy.append(batch_entropy)
            
            all_images[cnt:cnt+len(images)] = images.numpy()
            all_logits[cnt:cnt+len(images)] = logits
            all_labels[cnt:cnt+len(images)] = labels.numpy()
            cnt+=len(images)

        ldr_train = (all_images, all_logits, all_labels)
        #=============================
        # If args.soft_vote = True: 
        #    soft_vote from experts
        # Else: 
        #    just mean of all logits
        #=============================
        if not probe:
          return ldr_train, 0.0, 0.0
        else:
          test_acc = self.eval_ensemble(teachers, self.test_dataset)
          train_acc = self.eval_ensemble(teachers, self.ldr_local_train)
          
          plt.plot(range(len(teachers)), acc_per_teacher, marker="o", label="Acc")
          plt.plot(range(len(teachers)), conf_per_teacher, marker="o", label="Confidence")
          plt.plot(range(len(teachers)), conf_per_teacher - acc_per_teacher, marker="o", label="Confidence - Acc")
          plt.ylim(ymax = 1.0, ymin = -0.2)
          plt.title("Round %d, correct teacher/per sample %.2f, upperbound correct %.1f percentage"%(global_ep, teacher_per_sample,has_correct_teacher_ratio*100.0))
          plt.legend(loc='best')
          plt.savefig(os.path.join(log_dir, "acc_per_teacher_%d.png"% global_ep))       
          plt.clf()        

          return ldr_train, train_acc, test_acc

    def set_opt(self, net):
        base_opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.00001)
        if self.args.use_SWA:
            self.optimizer = SWA(base_opt, swa_start=500, swa_freq=25, swa_lr=None)
        else:
            self.optimizer = base_opt
    
    def train(self, net, teachers, log_dir, global_ep, server_dataset=None):
        #======================Record teachers========================
        self.set_opt(net)
        
        to_probe = True if global_ep%self.args.log_ep==0 else False
        ldr_train = []
        ldr_train, train_acc, test_acc = self.record_teacher(ldr_train, net, teachers, global_ep, log_dir, probe=to_probe)
        (all_images, all_logits, all_labels) = ldr_train 
        #======================Server Train========================
        print("Start server training...")
        net.cuda()        
        net.train()

        epoch_loss = []
        acc = 0
        cnt = 0
        
        step = 0
        train_ep = self.args.server_ep
        for iter in range(train_ep):
            all_ids = list(range(len(all_images)))
            np.random.shuffle(all_ids)
            
            batch_loss = []    
            for batch_idx in range(0, len(all_images), self.args.server_bs):
                ids = all_ids[batch_idx:batch_idx+self.args.server_bs]                
                images = all_images[ids]

                if self.aug:
                  images = self.transform_train(images)
                else:
                  images = torch.Tensor(images).cuda()   
                logits = all_logits[ids]
                labels = all_labels[ids]
                
                net.zero_grad()
                log_probs = net(images)
                
                loss, acc_cnt_i, cnt_i = self.loss_wrapper(log_probs, logits, labels)
                acc+=acc_cnt_i
                cnt+=cnt_i                
                loss.backward()
                
                self.optimizer.step()
                step+=1
            
                if batch_idx == 0 and iter%5==0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        val_acc = float(acc)/cnt*100.0   
        net_glob = copy.deepcopy(net)
        
        if self.args.use_SWA:
          self.optimizer.swap_swa_sgd()
          if "resnet" in self.args.model:
            self.optimizer.bn_update(self.ldr_train, net, device=None)
            
        net = net.cpu()   
        w_glob_avg = copy.deepcopy(net.state_dict())
        w_glob = net_glob.cpu().state_dict()
        
        print("Ensemble Acc Train %.2f Val %.2f Test %.2f mean entropy %.5f"%(train_acc, val_acc, test_acc, 0.0))    
        return w_glob_avg, w_glob, train_acc, val_acc, test_acc, sum(epoch_loss) / len(epoch_loss), 0.0


def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size   
    
def random_crop(images, crop_size):
    for i, image in enumerate(images):
      image = np.pad(image, crop_size)
      _, h, w = image.shape
      top = np.random.randint(0, crop_size*2)
      left = np.random.randint(0, crop_size*2)
      bottom = top + (h - 2*crop_size)
      right = left + (w - 2*crop_size)
      
      images[i] = image[crop_size:-crop_size, top:bottom, left:right]
      
    return images

def horizontal_flip(image, rate=0.5):
    if np.random.rand() < rate:
        #image = image[:, :, :, ::-1]
        image = np.flip(image, axis=-1)
    return image
    
