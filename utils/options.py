#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--rounds', type=int, default=40, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users")
    parser.add_argument('--num_data', type=int, default=40000, help="number of data distributed to users")
    parser.add_argument('--num_server_data', type=int, default=-1, help="number of trans data to use in the server: -1 for using all - num_data in users.")
    
    parser.add_argument('--aug', action='store_true', help="aug")
    parser.add_argument('--ens', action='store_true', help="ensemble")
    parser.add_argument('--store_model', action='store_true', help="store_model") 
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients")

    # Local train
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs")
    parser.add_argument('--local_bs', type=int, default=40, help="local batch size")

    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--local_sch', type=str, default='step', help='step, adaptive')
    parser.add_argument('--adap_ep', type=int, default=40, help="epochs for warm up training")
    parser.add_argument('--local_loss', type=str, default='CE', help='CE')
    parser.add_argument('--server_sample_freq', type=int, default=1, help='o, resample')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="weight_decay")

    parser.add_argument('--num_layers', type=int, default=0, help="extra conv layer") 
    parser.add_argument('--use_SWA', action='store_true', help="use_SWA") 
    parser.add_argument('--use_oracle', action='store_true', help="use_oracle") 
    parser.add_argument('--dont_add_fedavg', action='store_true', help="add_fedavg") 
    
    parser.add_argument('--log_dir', type=str, default='log', help='model name')
    parser.add_argument('--log_ep', type=int, default=5, help='log_ep')
    parser.add_argument('--exp', type=str, default='', help='model name')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--dataset_trans', type=str, default='', help="Unsupervised dataset for server")
        
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--split_method', type=str, default='step', help='split_method, [step, dir]')
    
    # client regualrzation: FedProx
    parser.add_argument('--reg_type', type=str, default='', help='FedProx, scaffold')
    parser.add_argument('--mu', type=float, default=0.001, help="mu")

    # SWAG & Server
    parser.add_argument('--fedM', action='store_true', help="FedAvgM") 
    parser.add_argument('--teacher_type', type=str, default='SWAG', help='ensemble')
    parser.add_argument('--client_type', type=str, default='real', help='real, g')
    
    parser.add_argument('--swag_stepsize', type=float, default=1.0, help="swag_stepsize")
    parser.add_argument('--client_stepsize', type=float, default=1.0, help="client_stepsize")
    parser.add_argument('--var_scale', type=float, default=0.1, help="var_scale")
    parser.add_argument('--num_sample_teacher', type=int, default=10, help="number of teachers")
    parser.add_argument('--num_base', type=int, default=20, help="number of teachers")
    
    parser.add_argument('--use_client', action='store_true', help="use_client")
    parser.add_argument('--use_fake', action='store_true', help="use_fake")
    parser.add_argument('--sample_teacher', type=str, default="gaussian", help="use_client")
    
    parser.add_argument('--loss_type', type=str, default='KL', help='server loss')
    parser.add_argument('--temp', type=float, default=0.5, help="temp")
    
    parser.add_argument('--mom', type=float, default=0.9, help="teacher momentum")
    parser.add_argument('--server_bs', type=int, default=128, help="server batch size: B")    
    parser.add_argument('--server_lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--update', type=str, default='dist', help='Aggregation update strategy, [FedAvg, dist]')
    parser.add_argument('--server_ep', type=int, default=20, help="the number of center epochs")
    parser.add_argument('--warmup_ep', type=int, default=-1, help="the number of warmup rounds")
    
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--num_gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    
    if args.update =="FedAvg": args.use_SWA = False
    if args.teacher_type != "SWAG": args.dont_add_fedavg = True
    
    return args
