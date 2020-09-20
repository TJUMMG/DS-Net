import torch
import torch.nn as nn
import torch.nn.functional as F

from model.aspp import ASPP
from model.resnet.resnet import ResNet101, ResNet18, ResNet34, ResNet50


INPUT_SIZE = 512


class ResNet_ASPP(nn.Module):
    def __init__(self, nInputChannels, n_classes, os, backbone_type):
        super(ResNet_ASPP, self).__init__()

        self.os = os
        self.backbone_type = backbone_type

        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8 or os == 32:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        if backbone_type == 'resnet18':
            self.backbone_features = ResNet18(nInputChannels, os, pretrained=False)
        elif backbone_type == 'resnet34':
            self.backbone_features = ResNet34(nInputChannels, os, pretrained=False)
        elif backbone_type == 'resnet50':
            self.backbone_features = ResNet50(nInputChannels, os, pretrained=False)
        elif self.backbone_type == 'resnet101':
            self.backbone_features = ResNet101(nInputChannels, os, pretrained=False)
        else:
            raise NotImplementedError

        asppInputChannels = 512
        asppOutputChannels = 256
        if backbone_type == 'resnet50' or backbone_type == 'resnet101': asppInputChannels = 2048

        self.aspp = ASPP(asppInputChannels, asppOutputChannels, rates)
        self.last_conv = nn.Sequential(
            nn.Conv2d(asppOutputChannels, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        )
    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, input):
        x, low_level_features, conv1_feat, layer2_feat, layer3_feat = self.backbone_features(input)
        layer4_feat = x
        if self.os == 32:
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.aspp(x)
        aspp_x = x
        x = self.last_conv(x)
        x = F.interpolate(x, input.shape[2:], mode='bilinear', align_corners=True)
        return layer4_feat, low_level_features, conv1_feat, layer2_feat, layer3_feat, aspp_x, x
