# pylint: disable=missing-function-docstring, missing-class-docstring, unused-argument
"""R2Plus1D, https://arxiv.org/abs/1711.11248. Code adapted from
https://github.com/pytorch/vision/blob/master/torchvision/models/video/resnet.py."""
import torch
import torch.nn as nn
from torch.nn import BatchNorm3d
import torchvision
from torchvision.models.video.resnet import VideoResNet, BasicBlock, R2Plus1dStem, Conv2Plus1D, BasicStem, Conv3DSimple
from torch.hub import load_state_dict_from_url

__all__ = ['r3d_18', 'r3d_34', 'r2plus1d_18', 'r2plus1d_34']

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}

def r3d_18(cfg):
    model = VideoResNet(block=BasicBlock, conv_makers=[Conv3DSimple] * 4, layers=[2, 2, 2, 2], stem=BasicStem)
    state_dict = load_state_dict_from_url(model_urls['r3d_18'], progress=True)
    model.load_state_dict(state_dict)
    model.fc = nn.Linear(model.fc.in_features, cfg.CONFIG.DATA.NUM_CLASSES)
    print('Pretrained Model Weight Loaded')
    return model

def r3d_34(cfg): 
    model = VideoResNet(block=BasicBlock, conv_makers=[Conv3DSimple] * 4, layers=[3, 4, 6, 3], stem=BasicStem)
    model.fc = nn.Linear(model.fc.in_features, cfg.CONFIG.DATA.NUM_CLASSES)
    return model

def r2plus1d_18(cfg):
    model = VideoResNet(block=BasicBlock, conv_makers=[Conv2Plus1D] * 4, layers=[2, 2, 2, 2], stem=R2Plus1dStem)
    state_dict = load_state_dict_from_url(model_urls['r2plus1d_18'], progress=True)
    model.load_state_dict(state_dict)
    model.fc = nn.Linear(model.fc.in_features, cfg.CONFIG.DATA.NUM_CLASSES)
    print('Pretrained Model Weight Loaded')
    return model

def r2plus1d_34(cfg):
    model = VideoResNet(block=BasicBlock, conv_makers=[Conv2Plus1D] * 4, layers=[3, 4, 6, 3], stem=R2Plus1dStem)
    
    
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)
    model.fc = nn.Linear(model.fc.in_features, 359)
    state_dict = torch.load('./logs/r2plus1d/pretrained.pth')
    model.load_state_dict(state_dict)
    model.fc = nn.Linear(model.fc.in_features, cfg.CONFIG.DATA.NUM_CLASSES)
    print('Pretrained Model Weight Loaded')
    return model

if __name__ == '__main__':

    model = VideoResNet(block=BasicBlock, conv_makers=[Conv3DSimple] * 4, layers=[2, 2, 2, 2], stem=BasicStem)
    state_dict = load_state_dict_from_url(model_urls['r3d_18'], progress=True)
    model.load_state_dict(state_dict)
    model.fc = nn.Linear(model.fc.in_features, 101)
    print('Pretrained Model Weight Loaded')
    input = torch.randn((2, 3, 4, 224, 224))
    output = model(input)
    print(output.shape)