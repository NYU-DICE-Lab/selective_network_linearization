from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

# For ImageNet and Tiny-ImageNet, you need to define the directory in _imagenet() and _tinyimagenet() function.

# list of all datasets
DATASETS = ["imagenet", "cifar10", "cifar100", "mnist", "fashion_mnist", "tiny_imagenet"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "cifar100":
        return _cifar100(split)
    elif dataset == "mnist":
        return _mnist10(split)
    elif dataset == "tiny_imagenet":
        return _tinyimagenet(split)
    elif dataset == "fashion_mnist":
        return _fashion_mnist10(split)
    

def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STDDEV = (0.2023, 0.1994, 0.2010)

_MNIST_MEAN = (0.1307,)
_MNIST_STDDEV = (0.3081,)

_FASHION_MNIST_MEAN = (0.28604,)
_FASHION_MNIST_STDDEV = (0.35302,)

_TINY_MEAN = [0.480, 0.448, 0.398]
_TINY_STD = [0.277, 0.269, 0.282]

def _mnist10(split: str) -> Dataset:
    if split == "train":
        return datasets.MNIST("./dataset_cache", train=True, download=True, transform=transforms.Compose([
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)
        ]))
    
    elif split == "test":
        return datasets.MNIST("./dataset_cache", train=False, download=True, transform=transforms.Compose([
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)
        ]))
    

def _fashion_mnist10(split: str) -> Dataset:
    if split == "train":
        return datasets.FashionMNIST("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_FASHION_MNIST_MEAN, _FASHION_MNIST_STDDEV)
        ]))
    elif split == "test":
        return datasets.FashionMNIST("./dataset_cache", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_FASHION_MNIST_MEAN, _FASHION_MNIST_STDDEV)
        ]))

def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STDDEV)
            
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STDDEV)
        ]))
    
def _cifar100(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR100("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            
        ]))
    elif split == "test":
        return datasets.CIFAR100("./dataset_cache", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])) 
    


def _imagenet(split: str) -> Dataset:
    if split == "train":
        subdir = os.path.join("path/to/imagenet", "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STDDEV)
        ])
    elif split == "test":
        subdir = os.path.join("path/to/imagenet", "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STDDEV)
        ])
    return datasets.ImageFolder(subdir, transform)

def _tinyimagenet(split: str) -> Dataset:
    if split == "train":
        subdir = os.path.join("/scratch/mc8065/tiny-imagenet-200", "train")
        transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_TINY_MEAN, _TINY_STD)
        ])
    elif split == "test":
        subdir = os.path.join("/scratch/mc8065/tiny-imagenet-200", "val")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_TINY_MEAN, _TINY_STD)
        ])
    return datasets.ImageFolder(subdir, transform)
