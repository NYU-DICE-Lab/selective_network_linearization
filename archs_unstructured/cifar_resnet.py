from __future__ import absolute_import

'''
This file is from: https://raw.githubusercontent.com/bearpaw/pytorch-classification/master/models/cifar/resnet.py
by Wei Yang
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from .init_utils import weights_init

from torch.autograd import Variable


__all__ = ['ResNet', 'resnet32', 'resnet20', 'resnet34_in', 'resnet50_in', 'resnet50', 'resnet', 'lenet', 'wide_resnet', 'resnet18_in', 'resnet9_in',
           'wide_resnet22_8', 'wide_resnet_22_8_drop02', 'wide_resnet_28_10_drop02', 'wide_resnet_28_12_drop02', 'wide_resnet_16_8_drop02'
           'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 
           'vgg19_bn', 'vgg19', 'lenet_5_caffe']


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class LearnableAlpha(nn.Module):
    def __init__(self, out_channel, feature_size):
        super(LearnableAlpha, self).__init__()
        self.alphas = nn.Parameter(torch.ones(1, out_channel, feature_size, feature_size), requires_grad=True)

    def forward(self, x):
        out = F.relu(x) * self.alphas.expand_as(x) + (1-self.alphas.expand_as(x)) * x 
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.alpha1 = LearnableAlpha(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.alpha2 = LearnableAlpha(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(x))
        out = self.alpha1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = F.relu(out)
        out = self.alpha2(out)
        return out

class BasicBlock_IN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, feature_size):
        super(BasicBlock_IN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.alpha1 = LearnableAlpha(planes, feature_size)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.alpha2 = LearnableAlpha(planes, feature_size)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(x))
        out = self.alpha1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = F.relu(out)
        out = self.alpha2(out)
        return out

class ResNet_IN(nn.Module):
    def __init__(self, block, num_blocks, args, num_classes=10):
        super(ResNet_IN, self).__init__()
        print("Hello WOrld")
        self.in_planes = 64
        if args.dataset in ['cifar10', 'cifar100']:
            self.feature_size = 32
            self.last_dim = 4
            print("CIFAR10/100 Setting")
        elif args.dataset in ['tiny_imagenet']:
            self.feature_size = 64
            self.last_dim = 8
            print("Tiny_ImageNet Setting")
            print("num_classes: ", num_classes)
        else:
            raise ValueError("Dataset not implemented for ResNet_IN")
        self.args = args
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=args.stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.alpha = LearnableAlpha(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.apply(_weights_init)

        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            self.feature_size = self.feature_size // 2 if stride == 2 else self.feature_size
            layers.append(block(self.in_planes, planes, stride, self.feature_size))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, self.last_dim if self.args.stride == 1 else self.last_dim // 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def resnet9_in(num_classes, args):
    return ResNet_IN(BasicBlock_IN, [1, 1, 1, 1], num_classes=num_classes, args=args)

def resnet18_in(num_classes, args):
    return ResNet_IN(BasicBlock_IN, [2, 2, 2, 2], num_classes=num_classes, args=args)

def resnet34_in(num_classes, args):
    return ResNet_IN(BasicBlock_IN, [3, 4, 6, 3], num_classes=num_classes, args=args)

def resnet50_in(num_classes, args):
    return ResNet_IN(BasicBlock_IN, [3, 4, 14, 3], num_classes=num_classes, args=args)
    

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class LeNet300(nn.Module):
    
    def __init__(self, num_classes=10):
        super(LeNet300, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 784)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
def lenet300(**kwargs):
    return LeNet300(**kwargs)


class LeNet5(nn.Module):
    
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def lenet5(**kwargs):
    return LeNet5(**kwargs)


class LeNet_5_Caffe(nn.Module):
    """
    This is based on Caffe's implementation of Lenet-5 and is slightly different
    from the vanilla LeNet-5. Note that the first layer does NOT have padding
    and therefore intermediate shapes do not match the official LeNet-5.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, padding=0)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc3 = nn.Linear(50 * 4 * 4, 500)
        self.fc4 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.fc3(x.view(-1, 50 * 4 * 4)))

        return x
    
def lenet_5_caffe(**kwargs):
    return LeNet_5_Caffe(**kwargs)


class FCN(nn.Module):
    def __init__(self, num_classes=10):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(784, 10)
    
    def forward(self, x):
        x = (self.fc1(x.view(-1, 784)))
        x = F.log_softmax(x, dim=1)
        return x
    
def fcn(**kwargs):
    return FCN(**kwargs)


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, feature_size, dropRate=0.0):
        super(WideBasicBlock, self).__init__()
        self.equalInOut = (in_planes == out_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        # self.alpha1 = LearnableAlpha(in_planes, feature_size)
        if not self.equalInOut and stride != 1:
            self.alpha1 = LearnableAlpha(in_planes, 2*feature_size)
        else:
            self.alpha1 = LearnableAlpha(in_planes, feature_size)
        
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.alpha2 = LearnableAlpha(out_planes, feature_size)
        self.droprate = dropRate
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            # x = self.relu1(self.bn1(x))
            x = self.alpha1(self.bn1(x))
        else:
            # out = self.relu1(self.bn1(x))
            out = self.alpha1(self.bn1(x))
        # out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.alpha2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, feature_size, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, feature_size, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, feature_size, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, feature_size, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, args, depth=22, num_classes=10, widen_factor=8, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        if args.dataset in ['cifar10', 'cifar100']:
            self.feature_size = 32
            self.last_dim = 8
        elif args.dataset in ['tiny_imagenet']:
            self.feature_size = 64
            self.last_dim = 16
        else:
            raise ValueError("Dataset not implemented in WideResNet")
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = WideBasicBlock
        self.args = args
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=args.stride,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, self.feature_size, dropRate)
        # 2nd block
        self.feature_size = self.feature_size // 2
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, self.feature_size, dropRate)
        # 3rd block
        self.feature_size = self.feature_size // 2
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, self.feature_size, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        # self.alpha = LearnableAlpha(nChannels[3])
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)
        out = F.avg_pool2d(out, self.last_dim if self.args.stride == 1 else self.last_dim // 2)
        out = self.fc(out.view(-1, self.nChannels))
        return out
    
def wide_resnet_16_8_drop02(num_classes, args):
    return WideResNet(depth=16, num_classes=num_classes, widen_factor=8, dropRate=0.2, args=args)

def wide_resnet_22_8(num_classes, args):
    return WideResNet(depth=22, num_classes=num_classes, widen_factor=8, args=args)

def wide_resnet_22_8_drop02(num_classes, args):
    return WideResNet(depth=22, num_classes=num_classes, widen_factor=8, dropRate=0.2, args=args)

def wide_resnet_28_10_drop02(num_classes, args):
    return WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.2, args=args)

def wide_resnet_28_12_drop02(num_classes, args):
    return WideResNet(depth=28, num_classes=num_classes, widen_factor=12, dropRate=0.2, args=args)

def wide_resnet(**kwargs):
    return WideResNet(**kwargs)


defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=19, init_weights=True, cfg=None, affine=True, batchnorm=True):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine
        self.feature = self.make_layers(cfg, batchnorm)
        self.num_classes = num_classes
        
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self.apply(weights_init)
        # if pretrained:
        #     model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.num_classes == 200:
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        y = F.log_softmax(x, dim=1)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def vgg19(num_classes):
    """VGG 19-layer model (configuration "E")"""
    return VGG(num_classes)



# class VGG(nn.Module):
#     '''
#     VGG model 
#     '''
#     def __init__(self, features, num_classes=10):
#         super(VGG, self).__init__()
#         self.features = features
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, num_classes),
#         )
#          # Initialize weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 m.bias.data.zero_()


#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         x = F.log_softmax(x, dim=1)
#         return x


# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)


# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
#           512, 512, 512, 512, 'M'],
# }


# def vgg11():
#     """VGG 11-layer model (configuration "A")"""
#     return VGG(make_layers(cfg['A']))


# def vgg11_bn():
#     """VGG 11-layer model (configuration "A") with batch normalization"""
#     return VGG(make_layers(cfg['A'], batch_norm=True))


# def vgg13():
#     """VGG 13-layer model (configuration "B")"""
#     return VGG(make_layers(cfg['B']))


# def vgg13_bn():
#     """VGG 13-layer model (configuration "B") with batch normalization"""
#     return VGG(make_layers(cfg['B'], batch_norm=True))


# def vgg16():
#     """VGG 16-layer model (configuration "D")"""
#     return VGG(make_layers(cfg['D']))


# def vgg16_bn():
#     """VGG 16-layer model (configuration "D") with batch normalization"""
#     return VGG(make_layers(cfg['D'], batch_norm=True))


# def vgg19():
#     """VGG 19-layer model (configuration "E")"""
#     return VGG(make_layers(cfg['E']))


# def vgg19_bn(num_classes):
#     """VGG 19-layer model (configuration 'E') with batch normalization"""
#     return VGG(make_layers(cfg['E'], batch_norm=True), num_classes)

# def vgg(num_classes):
#     return vgg19_bn(num_classes)

# def vgg_16(**kwargs):
#     return vgg16_bn(**kwargs)