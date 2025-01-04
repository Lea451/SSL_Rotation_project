import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from github of article
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        super(BasicBlock, self).__init__()
        padding = (kernel_size-1)//2
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes, \
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))
        self.layers.add_module('ReLU',      nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)

#from github of article
class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, feat):
        assert(feat.size(2) == feat.size(3))
        feat_avg = F.avg_pool2d(feat, feat.size(2)).view(-1, feat.size(1))
        return feat_avg

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class ConvClassifier(nn.Module):
    def __init__(self, opt):
        super(ConvClassifier, self).__init__()
        num_couche = opt['num_couche']
        if num_couche == 0:
            nChannels = 512
        elif num_couche == 1:
            nChannels = 256
        elif num_couche == 2:
            nChannels = 128
        elif num_couche == 3:
            nChannels = 64
        num_classes    = opt['num_classes']

        self.classifier = nn.Sequential()
        self.classifier.add_module('Block3_ConvB1',  BasicBlock(nChannels, 192, 3))
        self.classifier.add_module('Block3_ConvB2',  BasicBlock(192, 192, 1))
        self.classifier.add_module('Block3_ConvB3',  BasicBlock(192, 192, 1))
        self.classifier.add_module('GlobalAvgPool',  GlobalAvgPool())
        self.classifier.add_module('Liniear_F',      nn.Linear(192, num_classes))
        
        self.initilize()

    def forward(self, feat):
        return self.classifier(feat)

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fin = m.in_features
                fout = m.out_features
                std_val = np.sqrt(2.0/fout)
                m.weight.data.normal_(0.0, std_val)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

def create_model(opt):
    return ConvClassifier(opt)