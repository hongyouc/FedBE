
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import pdb
import os
import pickle

import numpy as np
from swag import SWAG_server

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.sampling import *
from utils.options import args_parser
from utils.tools import *
from utils.main_extensions import *

from models.Update import SWAGLocalUpdate, ServerUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, create_local_init
from models.FedM import FedAvgM
from models.test import test_img
import resnet

# parse args
args = args_parser()

# make all the directories
args.log_dir = os.path.join(args.log_dir)   

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)  

with open(os.path.join(args.log_dir, "args.txt"), "w") as f:
    for arg in vars(args):
        print (arg, getattr(args, arg), file=f)

args.acc_dir = os.path.join(args.log_dir, "acc")
if not os.path.exists(args.acc_dir):
    os.makedirs(args.acc_dir)  
    
model_dir = os.path.join(args.log_dir, "models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)         

# transform train parameters
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
print("args.iid")
print(args.iid)
# load dataset and split users
if args.dataset == 'mnist':
    args.num_classes = 10
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    dataset_eval = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)        
    # sample users
    if args.iid:
        dict_users = mnist_iid(dataset_train, args.num_users)
    else:
        dict_users, server_id, cnts_dict  = mnist_noniid(dataset_train, args.num_users)
elif args.dataset == 'cifar':
    args.num_classes = 10
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_val)
    dataset_eval = datasets.CIFAR10('./data/cifar', train=True, transform=transform_val, target_transform=None, download=True)
    if args.iid:
        dict_users, server_id = cifar_iid(dataset_train, args.num_users, num_data=args.num_data)
    else:
        dict_users, server_id, cnts_dict = cifar_noniid(dataset_train, args.num_users, num_data=args.num_data, method=args.split_method)
else:
    exit('Error: unrecognized dataset')
    
train_ids = set()
# dict_users.items() is the content of the dictionary
for u,v in dict_users.items():
    train_ids.update(v)
# train_ids is the liats of all the ids in dict_users for all the users in a 1d array
train_ids = list(train_ids)     

img_size = dataset_train[0][0].shape
# build model
# models stored in models.Nets
if args.model == 'cnn' and 'cifar' in args.dataset:
    net_glob = CNNCifar(args=args)
elif args.model == 'cnn' and args.dataset == 'mnist':
    net_glob = CNNMnist(args=args)
elif args.model == 'mlp':
    len_in = 1
    for x in img_size:
        len_in *= x
    net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
elif "resnet" in args.model and 'cifar' in args.dataset:
    net_glob = resnet.resnet32(num_classes=args.num_classes)   
else:
    exit('Error: unrecognized model')    
    
print(net_glob)
net_glob.train()

# copy weights
w_glob = net_glob.state_dict()

# training
loss_local_list = []
loss_local_test_list = []
entropy_list = []
cv_loss, cv_acc = [], []


acc_local_list = []
acc_local_test_list = []
acc_local_val_list = []

val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []

net_glob.apply(weights_init)    
# Arguments:
# q: mp.Manager.Queue
# device_id: the gpu thread being used to train the client
# net_glob: deep copy of net_glob
# iters: Current round number
# idx: idx of the client participating in the current round (range(m))
# val_id: server_id
# generator: None
#
# return: 
# A trained teacher model and its index (also put on manger queue)
def client_train(q, device_id, net_glob, iters, idx, val_id=server_id, generator=None):
    device=torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() and args.num_gpu != -1 else 'cpu')
    print(device)
    #device=torch.device('cpu')
    # LearningRate schedule (def in "tools")
    lr = lr_schedule(args.lr, iters, args.rounds)  

    # local_sch : either step or adaptive. 
    running_ep = args.local_ep
    if args.local_sch == "adaptive":
        # local_ep : the number of local epochs
        # adaptive scheduler (def in "tools")
        running_ep = adaptive_schedule(args.local_ep, args.epochs, iters, args.adap_ep)
    if running_ep != args.local_ep:
        print("Using adaptive scheduling, local ep = %d."%args.adap_ep)
    else:
        running_ep = args.local_ep

    # In models
    local = SWAGLocalUpdate(args=args, 
                            device=device, 
                            dataset=dataset_train, 
                            idxs=dict_users[idx], 
                            server_ids=val_id, 
                            test=(dataset_test, range(len(dataset_test))), 
                            num_per_cls=cnts_dict[idx]   )

    # train a model using SWAGLocalUpdate, this model is called teacher
    teacher = local.train(net=net_glob.to(device), running_ep=running_ep, lr=lr)
    q.put([teacher, idx])
    return [teacher, idx]

# Arguments:
# q : mp.Manager.Queue()
# device_id : a constant. pretty sure this one doesn't do anything.
# net_glob : global deep network
# teachers: set of sampled teachers (and maybe also clients)
# global_ep: current round number
# w_org : None
# base_teachers = None
#
# !!!!!!!!!Output:
# w_swa : weight after stocastic weight averaging
# w_glob : 
# train_acc, val_acc, test_acc : 
# loss, entropy : 
def server_train(q, device_id, net_glob, teachers, global_ep, w_org=None, base_teachers=None):
    device=torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() and args.num_gpu != -1 else 'cpu')
    student = ServerUpdate(args=args, 
                        device=device, 
                        dataset=dataset_eval, 
                        server_dataset=dataset_eval, 
                        server_idxs=server_id, 
                        train_idx=train_ids, 
                        test=(dataset_test, range(len(dataset_test))), 
                        w_org=w_org, 
                        base_teachers=base_teachers)
    
    w_swa, w_glob, train_acc, val_acc, test_acc, loss, entropy = student.train(net_glob, teachers, args.log_dir, global_ep)

    q.put([w_swa, w_glob, train_acc, val_acc, test_acc, entropy])
    return [w_swa, w_glob, train_acc, val_acc, test_acc, entropy]
    
# Arguments:
# q : mp.Manager.Queue()
# net_glob : blobal deep network
# dataset : dataset being tested
# ids : a list of indexes of the data from the dataset being tested
#
# Output:
# [acc, loss] : accuracy and loss of the model (also put on Queue)
def test_thread(q, net_glob, dataset, ids):
    # acc: accuracy
    # loss: test loss
    acc, loss = test_img(net_glob, dataset, args, ids, cls_num=args.num_classes)
    q.put([acc, loss])
    return [acc, loss]

def eval(net_glob, tag='', server_id=None):
    # testing
    q = mp.Manager().Queue()

    p_tr = mp.Process(target=test_thread, args=(q, net_glob, dataset_eval, train_ids))  
    p_tr.start()
    p_tr.join()
    [acc_train, loss_train] = q.get()

    q2 = mp.Manager().Queue()
    p_te = mp.Process(target=test_thread, args=(q2, net_glob, dataset_test, range(len(dataset_test))))  
    p_te.start()
    p_te.join()

    [acc_test,  loss_test] = q2.get()

    q3 = mp.Manager().Queue()
    p_val = mp.Process(target=test_thread, args=(q3, net_glob, dataset_eval, server_id))
    p_val.start()
    p_val.join()

    [acc_val,  loss_val] = q3.get()

    print(tag, "Training accuracy: {:.2f}".format(acc_train))
    print(tag, "Server accuracy: {:.2f}".format(acc_val))
    print(tag, "Testing accuracy: {:.2f}".format(acc_test))

    del q
    del q2 
    del q3 

    return [acc_train, loss_train], [acc_test,  loss_test], [acc_val,  loss_val]


def put_log(logger, net_glob, tag, iters=-1):
    [acc_train, loss_train], [acc_test,  loss_test], [acc_val,  loss_val] = eval(net_glob, tag=tag, server_id=server_id)

    if iters==0:
        open(os.path.join(args.acc_dir, tag+"_train_acc.txt"), "w")
        open(os.path.join(args.acc_dir, tag+"_val_acc.txt"), "w")
        open(os.path.join(args.acc_dir, tag+"_test_acc.txt"), "w")
        open(os.path.join(args.acc_dir, tag+"_test_loss.txt"), "w")

    with open(os.path.join(args.acc_dir, tag+"_train_acc.txt"), "a") as f:
        f.write("%d %f\n"%(iters, acc_train))
    with open(os.path.join(args.acc_dir, tag+"_test_acc.txt"), "a") as f:
        f.write("%d %f\n"%(iters, acc_test))
    with open(os.path.join(args.acc_dir, tag+"_val_acc.txt"), "a") as f:
        f.write("%d %f\n"%(iters, acc_val))          
    with open(os.path.join(args.acc_dir, tag+"_test_loss.txt"), "a") as f:
        f.write("%d %f\n"%(iters, loss_test))
        
    if "SWA" not in tag:
        logger.loss_train_list.append(loss_train)
        logger.train_acc_list.append(acc_train)

        logger.loss_test_list.append(loss_test)
        logger.test_acc_list.append(acc_test)

        logger.loss_val_list.append(loss_val)
        logger.val_acc_list.append(acc_val)
    else:
        if tag =="SWAG":
            logger.swag_train_acc_list.append(acc_train)
            logger.swag_val_acc_list.append(acc_val) 
            logger.swag_test_acc_list.append(acc_test)            
        else:
            logger.swa_train_acc_list.append(acc_train)
            logger.swa_val_acc_list.append(acc_val) 
            logger.swa_test_acc_list.append(acc_test)   


def put_oracle_log(logger, ens_train_acc, ens_val_acc, ens_test_acc, iters=-1):    
    if iters>=0 and iters%args.log_ep!= 0:
        return
    logger.ens_train_acc_list.append(ens_train_acc)
    logger.ens_test_acc_list.append(ens_test_acc)
    logger.ens_val_acc_list.append(ens_val_acc)

    tag = "ens"
    if iters==0:
        open(os.path.join(args.acc_dir, tag+"_train_acc.txt"), "w")
        open(os.path.join(args.acc_dir, tag+"_val_acc.txt"), "w")
        open(os.path.join(args.acc_dir, tag+"_test_acc.txt"), "w")
    
    with open(os.path.join(args.acc_dir, tag+"_train_acc.txt"), "a") as f:
        f.write("%d %f\n"%(iters, ens_train_acc))
    with open(os.path.join(args.acc_dir, tag+"_test_acc.txt"), "a") as f:
        f.write("%d %f\n"%(iters, ens_test_acc))
    with open(os.path.join(args.acc_dir, tag+"_val_acc.txt"), "a") as f:
        f.write("%d %f\n"%(iters, ens_val_acc))