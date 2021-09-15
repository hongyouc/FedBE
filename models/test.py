#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np 

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
        
def onehot_encode(target, n_classes):
    y = torch.zeros(len(target), n_classes).cuda()
    y[range(y.shape[0]), target]=1
    
    return y
    
def test_img(net_g, datatest, args, idxs, reweight=None, cls_num=10):
    net_g.eval()
    test_loss = 0
    correct = 0
    cnt = 0.0
    
    data_loader = DataLoader(DatasetSplit(datatest, idxs), batch_size=1024, shuffle=False)
    l = len(data_loader)
    net_g = net_g.cuda()
    with torch.no_grad():
      for idx, (data, target) in enumerate(data_loader):
          if args.gpu != -1:
              data, target = data.cuda(), target.cuda()
          log_probs = net_g(data)
          # sum up batch loss
          test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
          # get the index of the max log-probability
          y_pred = log_probs.data.max(1, keepdim=True)[1]
          target = target.data.view_as(y_pred)
          correct += y_pred.eq(target).long().cpu().sum()
          cnt += len(data)

    test_loss /= cnt
    accuracy = 100.00 * correct / cnt

    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy.numpy(), test_loss

