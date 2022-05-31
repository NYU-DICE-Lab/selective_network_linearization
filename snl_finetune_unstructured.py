# Selective Network Linearization unstructured method.
# Starting from the pretrained model. 

import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from architectures_unstructured import ARCHITECTURES, get_architecture
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
import datetime
import time
import numpy as np
import copy
import types
from math import ceil
from train_utils import AverageMeter, accuracy, accuracy_list, init_logfile, log
from utils import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('savedir', type=str, help='folder to load model')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--finetune_epochs', default=100, type=int,
                    help='number of total epochs for the finetuning')
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--logname', type=str, default='log.txt')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--alpha', default=1e-5, type=float,
                    help='Lasso coefficient')
parser.add_argument('--threshold', default=1e-2, type=float)
parser.add_argument('--relu_budget', default=50000, type=int)
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--gpu', default=0, type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--stride', type=int, default=1, help='conv1 stride')
args = parser.parse_args()


def num_relus(net):
    total = 0
    for name, param in net.named_parameters():
        if 'alpha' in name:
            total += param.numel()
    return total

def main():
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu)

    logfilename = os.path.join(args.outdir, args.logname)

    log(logfilename, "Hyperparameter List")
    log(logfilename, "Finetune Epochs: {:}".format(args.finetune_epochs))
    log(logfilename, "Learning Rate: {:}".format(args.lr))
    log(logfilename, "Alpha: {:}".format(args.alpha))
    log(logfilename, "ReLU Budget: {:}".format(args.relu_budget))

    print("Hyperparameter List")
    print("Finetune Epochs: {:}".format(args.finetune_epochs))
    print("Learning Rate: {:}".format(args.lr))
    print("Alpha: {:}".format(args.alpha))
    print("ReLU Budget: {:}".format(args.relu_budget))

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)


    # Loading the base_classifier
    base_classifier = get_architecture(args.arch, args.dataset, device, args)
    checkpoint = torch.load(args.savedir, map_location=device)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()

    log(logfilename, "Loaded the base_classifier")
    print("Loaded the base_classifier")

    # Calculating the loaded model's test accuracy.
    original_acc = model_inference(base_classifier, test_loader,
                                    device, display=True)
    
    log(logfilename, "Original Model Test Accuracy: {:.5}".format(original_acc))
    print("Original Model Test Accuracy, ", original_acc)

    # Creating a fresh copy of network not affecting the original network.
    net = copy.deepcopy(base_classifier)
    net = net.to(device)

    relu_count = relu_counting(net, args.arch, args)

    print("Original ReLU Count: {}".format(relu_count))

    for name, param in net.named_parameters():
        if 'alpha' in name:
            param.requires_grad = True
        
    criterion = nn.CrossEntropyLoss().to(device)  
    optimizer = Adam(net.parameters(), lr=args.lr)
    
    total = 0
    total = num_relus(net)

    # Corresponds to Line 4-9
    lowest_relu_count, relu_count = total, total
    for epoch in range(args.epochs):
        if epoch % 5 == 0:
            print("Current epochs: ", epoch)
            print("ReLU Count: {:}".format(relu_count))
        
        train_loss = mask_train_kd_unstructured(train_loader, net, base_classifier, criterion, optimizer,
                                epoch, device, alpha=args.alpha, display=False)
        acc = model_inference(net, test_loader, device, display=False)

        print("Epoch {:}, Mask Update Test Acc: {:.5}".format(epoch, acc))
        log(logfilename, "Epoch {:}, Mask Update Test Acc: {:.5}".format(epoch, acc))

        # counting ReLU in the neural network by using threshold.
        relu_count = 0
        for name, param in net.named_parameters():
            if 'alpha' in name:
                boolean_list = param.data > args.threshold
                relu_count += (boolean_list == 1).sum()
                
        if relu_count < lowest_relu_count:
            lowest_relu_count = relu_count 
        elif relu_count >= lowest_relu_count and epoch >= 5:
            args.alpha *= 1.1
            print("args.alpha = {}".format(args.alpha))

        if relu_count <= args.relu_budget:
            print("Current epochs breaking loop at {:}".format(epoch))
            break

    relu_count = 0
    for name, param in net.named_parameters():
        if 'alpha' in name:
            boolean_list = param.data > args.threshold
            relu_count += (boolean_list == 1).sum()

    print("After SNL Algorithm, the current ReLU Count: {}".format(relu_count))
    log(logfilename, "After SNL Algorithm, the current ReLU Count: {}".format(relu_count))

    # Line 11
    for name, param in net.named_parameters():
        if 'alpha' in name:
            boolean_list = param.data > args.threshold
            param.data = boolean_list.float()
            param.requires_grad = False

 
    # Line 12
    finetune_epoch = args.finetune_epochs

    optimizer = SGD(net.parameters(), lr=1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epoch)
    
    print("Finetuning the model")
    log(logfilename, "Finetuning the model")

    best_top1 = 0
    for epoch in range(finetune_epoch):
        train_loss, train_top1, train_top5 = train_kd(train_loader, net, base_classifier, optimizer, criterion, epoch, device)
        test_loss, test_top1, test_top5 = test(test_loader, net, criterion, device, 100, display=True)
        scheduler.step()
        
        if best_top1 < test_top1:
            best_top1 = test_top1
            is_best = True
        else:
            is_best = False

        if is_best:
            torch.save({
                    'arch': args.arch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'snl_best_checkpoint.pth.tar'))

    print("Final best Prec@1 = {}%".format(best_top1))
    log(logfilename, "Final best Prec@1 = {}%".format(best_top1))
        
if __name__ == "__main__":
    main()