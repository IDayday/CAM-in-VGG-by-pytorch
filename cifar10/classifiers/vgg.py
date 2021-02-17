import torch.nn as nn

"""
因为是CIFAR10数据集，所以全连接层没有设置三层，仅保留一层，后面也没有接softmax
"""
_vgg_config = {
    'VGG9_avgpool': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512 ,'M', 512, 512],
    'VGG11_avgpool': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG17_avgpool': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512 , 512, 512, 'M', 512, 512, 512, 512],
    'VGG9_conv': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512 ,'M', 512, 512],
    'VGG11_conv': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG17_conv': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512 , 512, 512, 'M', 512, 512, 512, 512]
}

def _make_layers(cfg):
    layers = []
    in_channels = 3
    for layer_cfg in cfg:
        if layer_cfg == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=layer_cfg,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True))
            layers.append(nn.BatchNorm2d(num_features=layer_cfg))
            layers.append(nn.ReLU(inplace=True))
            in_channels = layer_cfg
    return nn.Sequential(*layers)

class _VGG_avgpool(nn.Module):
    """
    VGG module for 3x32x32 input, 10 classes
    """

    def __init__(self, name):
        super(_VGG_avgpool, self).__init__()
        cfg = _vgg_config[name]
        self.layers = _make_layers(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        flatten_features = 512
        self.final_layer = nn.Linear(flatten_features, 10)


    def forward(self, x):
        y = self.layers(x)
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.final_layer(y)
        return y

class _VGG_conv(nn.Module):
    """
    VGG module for 3x32x32 input, 10 classes
    """

    def __init__(self, name):
        super(_VGG_conv, self).__init__()
        cfg = _vgg_config[name]
        self.layers = _make_layers(cfg)
        flatten_features = 512
        self.final_layer = nn.Sequential(
            nn.Conv2d(flatten_features, 10, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        y = self.layers(x)
        # print(y)
        y = self.final_layer(y)
        # print(y)
        return y

def VGG9_avgpool():
    return _VGG_avgpool('VGG9_avgpool')

def VGG11_avgpool():
    return _VGG_avgpool('VGG11_avgpool')

def VGG17_avgpool():
    return _VGG_avgpool('VGG17_avgpool')

def VGG9_conv():
    return _VGG_conv('VGG9_conv')

def VGG11_conv():
    return _VGG_conv('VGG11_conv')

def VGG17_conv():
    return _VGG_conv('VGG17_conv')
