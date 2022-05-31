import torch
import os
import json
import logging
import time 


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_list(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = torch.tensor(output).cuda().topk(max(topk))
        pred = pred.t()
        correct = pred.eq(target)

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    

def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()
    
def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename)) 

    
def get_logger(name, logpath, filepath, package_files=[],
               displaying=True, saving=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_path = logpath + name + time.strftime("-%Y%m%d-%H%M%S")
    makedirs(log_path)
    if saving:
        info_file_handler = logging.FileHandler(log_path)
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
    logger.info(filepath)
    with open(filepath, 'r') as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger



