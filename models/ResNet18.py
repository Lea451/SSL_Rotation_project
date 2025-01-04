import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, opt):
        super(ResNet18, self).__init__()
        print("opt.keys() before Resnet", opt.keys())
        self.num_classes = opt['num_classes']

        # Initial layers : standard choice of kernel 7 stride 2 padding 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #pooling layer also standard

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_block = nn.Sequential(
            Flatten(),
            nn.Linear(512, self.num_classes)
        )

        self.all_feat_names = [
            'conv1',
            'pool1',
            'layer1',
            'layer2',
            'layer3',
            'layer4',
            'avgpool',
            'fc_block'
        ]
        

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _parse_out_keys_arg(self, out_feat_keys):
        out_feat_keys = [self.all_feat_names[-1]] if out_feat_keys is None else out_feat_keys

        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')
        for key in out_feat_keys:
            if key not in self.all_feat_names:
                raise ValueError(f'Feature with name {key} does not exist. Existing features: {self.all_feat_names}.')

        max_out_feat = max(self.all_feat_names.index(key) for key in out_feat_keys)
        return out_feat_keys, max_out_feat

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc_block(x)
        return x            

    #def forward(self, x, out_feat_keys=None):
    #    out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
    #    out_feats = [None] * len(out_feat_keys)

    #    feat = x
    #    for f in range(max_out_feat + 1):
    #        feat = self._feature_blocks[f](feat)
    #        key = self.all_feat_names[f]
    #       if key in out_feat_keys:
     #           out_feats[out_feat_keys.index(key)] = feat

    #    out_feats = out_feats[0] if len(out_feats) == 1 else out_feats
    #    return out_feats

def create_model(opt):
    return ResNet18(opt)

if __name__ == '__main__':
    opt = {'num_classes': 4}
    model = create_model(opt)

    #TODO: test the model