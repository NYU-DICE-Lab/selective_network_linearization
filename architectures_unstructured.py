import torch
# from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from archs_unstructured.cifar_resnet import resnet34_in, resnet50_in, resnet18_in, resnet9_in
from archs_unstructured.cifar_resnet import wide_resnet_22_8, wide_resnet_28_10_drop02
# from archs.cifar_resnet import wide_resnet_22_8, wide_resnet_22_8_drop02, wide_resnet_28_10_drop02, wide_resnet_28_12_drop02, wide_resnet_16_8_drop02
from torch.nn.functional import interpolate


ARCHITECTURES = ["resnet50", "lenet300", "lenet5", "vgg19", "resnet32", "resnet50", "resnet34_in", "resnet50_in", "vgg16", "lenet_5_caffe", 
                 "resnet18_in", "wide_resnet_16_8_drop02", "resnet9_in",
                 "wide_resnet_22_8_drop02", "wide_resnet_22_8", "wide_resnet_28_10", "wide_resnet_28_12", "wide_resnet_28_10_drop02", "wide_resnet_28_12_drop02"]
def get_architecture(arch: str, dataset: str, device, args) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """

    if arch == "resnet9_in" and dataset == "cifar100":
        model = resnet9_in(num_classes=100, args=args).to(device)
    elif arch == "resnet18_in" and dataset == "cifar100":
        model = resnet18_in(num_classes=100, args=args).to(device)
    elif arch == "resnet18_in" and dataset == "cifar10":
        model = resnet18_in(num_classes=10, args=args).to(device)
    elif arch == "resnet18_in" and dataset == "tiny_imagenet":
        model = resnet18_in(num_classes=200, args=args).to(device)
    elif arch == "resnet34_in" and dataset == "cifar100":
        model = resnet34_in(num_classes=100, args=args).to(device)
    elif arch == "resnet34_in" and dataset == "cifar10":
        model = resnet34_in(num_classes=10, args=args).to(device)
    elif arch == "resnet34_in" and dataset == "tiny_imagenet":
        model = resnet34_in(num_classes=200, args=args).to(device)
    elif arch == "resnet50_in" and dataset == "cifar100":
        model = resnet50_in(num_classes=100, args=args).to(device)
    elif arch == "wide_resnet_22_8" and dataset == "cifar100":
        model = wide_resnet_22_8(num_classes=100, args=args).to(device)
    elif arch == "wide_resnet_22_8" and dataset == "cifar10":
        model = wide_resnet_22_8(num_classes=10, args=args).to(device)
    elif arch == "wide_resnet_22_8" and dataset == "tiny_imagenet":
        model = wide_resnet_22_8(num_classes=200, args=args).to(device)
    elif arch == "wide_resnet_28_10" and dataset == "cifar100":
        model = wide_resnet_28_10_drop02(num_classes=100, args=args).to(device)
    elif arch == "wide_resnet_28_10" and dataset == "cifar10":
        model = wide_resnet_28_10_drop02(num_classes=10, args=args).to(device)
    else:
        raise AssertionError("Your architecture is not in the list.")
    return model