import argparse
import os
import sys
from datasets import get_dataset, DATASETS, get_num_classes
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
import datetime
import time
import numpy as np
import shutil
import copy
import types
# from architectures import get_architecture
from math import ceil
from train_utils import AverageMeter, accuracy, accuracy_list

class Logits(nn.Module):
    def __init__(self):
        super(Logits, self).__init__()
    
    def forward(self, out_s, out_t):
        loss = F.mse_loss(out_s, out_t)
        return loss

class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss

def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, 
          epoch: int, device, print_freq=100, display=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
#     print("Entered training function")

    # switch to train mode
    model.train()
    
    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        targets = targets.to(device)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and display == True:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return (losses.avg, top1.avg, top5.avg)

def train_kd(train_loader, nets_student, nets_teacher, optimizer, criterion, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    nets_student.train()
    nets_teacher.eval()

    # criterion_kd = Logits().to(device)
    criterion_kd = SoftTarget(4.0).to(device)

    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs_s = nets_student(inputs)
        outputs_t = nets_teacher(inputs)

        loss = criterion(outputs_s, targets)
        loss_kd = criterion_kd(outputs_s, outputs_t.detach()) * 1.0

        total_loss = loss + loss_kd

        acc1, acc5 = accuracy(outputs_s, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    print(
        'Epoch: [{0}][{1}/{2}]\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5)
    )
    return (losses.avg, top1.avg, top5.avg)



def test(loader: DataLoader, model: torch.nn.Module, criterion, device, print_freq, display=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 and display == True:
                print('Test : [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        print(
            'Test Loss  ({loss.avg:.4f})\t'
            'Test Acc@1 ({top1.avg:.3f})\t'
            'Test Acc@5 ({top5.avg:.3f})'.format(
        loss=losses, top1=top1, top5=top5)
        )

        return (losses.avg, top1.avg, top5.avg)

def model_inference(base_classifier, loader, device, display=False, print_freq=100):
    print_freq = 100
    top1 = AverageMeter()
    top5 = AverageMeter()

    start = time.time()
    base_classifier.eval()
    # Regular dataset:
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = base_classifier(inputs)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            
            if i % print_freq == 0 and display == True:
                print("Test : [{0}/{1}]\t"
                      "Acc@1 {top1.avg:.3f}"
                      "Acc@5 {top5.avg:.3f}".format(
                      i, len(loader), top1=top1, top5=top5))
    end = time.time()
    if display == True:
        print("Inference Time: {0:.3f}".format(end-start))
        print("Final Accuracy: [{0}]".format(top1.avg))
        
    return top1.avg

def model_inference_imagenet(base_classifier, loader, device, display=False, print_freq=1000):
    print_freq = 100
    top1 = AverageMeter()
    top5 = AverageMeter()

    start = time.time()
    base_classifier.eval()
    # Regular dataset:
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = torch.tensor(targets)
            targets = targets.to(device, non_blocking=True)
            outputs = base_classifier(inputs)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            
            if i % print_freq == 0 and display == True:
                print("Test : [{0}/{1}]\t"
                      "Acc@1 {top1.avg:.3f}"
                      "Acc@5 {top5.avg:.3f}".format(
                      i, len(loader), top1=top1, top5=top5))
    end = time.time()
    if display == True:
        print("Inference Time: {0:.3f}".format(end-start))
        print("Final Accuracy: [{0}]".format(top1.avg))
        
    return top1.avg, top5.avg
        
def mask_train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, 
               epoch: int, device, alpha, display=False):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    
    for i, (inputs, targets) in enumerate(loader):

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        reg_loss = 0
        empty_tensor = []
        for name, param in model.named_parameters():
            if 'alpha' in name:
                empty_tensor.append(param)
        reg_loss += torch.norm(torch.cat(empty_tensor, dim=0), p=1)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets) + alpha * reg_loss

        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg

def mask_train_kd(loader: DataLoader, model: torch.nn.Module, model_teacher: torch.nn.Module, criterion, optimizer: Optimizer, 
               epoch: int, device, alpha, display=False):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    model_teacher.eval()

    criterion_kd = SoftTarget(4.0).to(device)
    
    for i, (inputs, targets) in enumerate(loader):

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        reg_loss = 0
        empty_tensor = []
        for name, param in model.named_parameters():
            if 'alpha' in name:
                empty_tensor.append(param)
        reg_loss += torch.norm(torch.cat(empty_tensor, dim=1), p=1)
        # compute output
        outputs = model(inputs)
        outputs_t = model_teacher(inputs)
        loss = criterion(outputs, targets) + criterion_kd(outputs, outputs_t) + alpha * reg_loss

        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg

def mask_train_kd_unstructured(loader: DataLoader, model: torch.nn.Module, model_teacher: torch.nn.Module, criterion, optimizer: Optimizer, 
               epoch: int, device, alpha, display=False):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    model_teacher.eval()

    criterion_kd = SoftTarget(4.0).to(device)
    
    for i, (inputs, targets) in enumerate(loader):

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        reg_loss = 0
        for name, param in model.named_parameters():
            if 'alpha' in name:
                reg_loss += torch.norm(param, p=1)
        # compute output
        outputs = model(inputs)
        outputs_t = model_teacher(inputs)
        loss = criterion(outputs, targets) + criterion_kd(outputs, outputs_t) + alpha * reg_loss

        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


class CosineAnnealingAlpha():
    def __init__(self, T_max, eta_min=0):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingAlpha, self).__init__()

