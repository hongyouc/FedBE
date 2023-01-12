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
import torchvision.utils as tv_utils
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

if __name__ == '__main__':
    args = args_parser()
    
    # make all the directories
    args.log_dir = os.path.join(args.log_dir)   
    model_dir = os.path.join(args.log_dir, "FedBE_wSWA_model")

    model = CNNCifar(args=args)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    args.num_classes = 10
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_test = datasets.CIFAR10('../data/mnist/', train=False, download=True, transform=trans_mnist)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_val)

    classes = ('0','1','2','3','4','5','6','7','8','9')

    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=True, num_workers=2)
    
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

    [acc_train, loss_train], [acc_test,  loss_test], [acc_val,  loss_val] = eval(model, tag="DIST", server_id=server_id)
    print("%f\n"%(acc_train))
    
    dataiter = iter(testloader)



    images, labels = next(dataiter)
    outputs = model(images)

    _, predicted = torch.max(outputs, 1)
    print(predicted)

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    print('Truth: ', ' '.join(f'{classes[labels[j]]:5s}'
                                for j in range(4)))
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
