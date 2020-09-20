import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_aspp import ResNet_ASPP


class Model(nn.Module):
    def __init__(self, input_channel, mode):
        super(Model, self).__init__()
        self.ImageBone = ResNet_ASPP(input_channel, 1, 16, 'resnet34')
        self.FlowBone = ResNet_ASPP(input_channel, 1, 16, 'resnet34')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.conv1_i = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2_i = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv3_i = nn.Sequential(nn.Conv2d(256, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU())
        self.conv4_i = nn.Sequential(nn.Conv2d(512, 64, 7, 1, 3), nn.BatchNorm2d(64), nn.ReLU())
        self.convaspp_i = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.convp_i = nn.Sequential(nn.Conv2d(320, 5, 3, 1, 1), nn.BatchNorm2d(5), nn.ReLU())

        self.conv1_f = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2_f = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv3_f = nn.Sequential(nn.Conv2d(256, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU())
        self.conv4_f = nn.Sequential(nn.Conv2d(512, 64, 7, 1, 3), nn.BatchNorm2d(64), nn.ReLU())
        self.convaspp_f = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.convp_f = nn.Sequential(nn.Conv2d(320, 5, 3, 1, 1), nn.BatchNorm2d(5), nn.ReLU())

        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(512, 512, 7, 1, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 7, 1, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 7, 1, 3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.blockaspp = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.last_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if mode == 'train':
            self.ImageBone.backbone_features._load_pretrained_model('./model/resnet/pre_train/resnet34-333f7ec4.pth')
            self.FlowBone.backbone_features._load_pretrained_model('./model/resnet/pre_train/resnet34-333f7ec4.pth')

    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, image, flow):
        img_layer4_feat, img_layer1_feat, img_conv1_feat, img_layer2_feat, img_layer3_feat, img_aspp_feat, course_img = self.ImageBone(
            image)

        i1 = self.conv1_i(img_layer1_feat)
        i2 = self.conv2_i(img_layer2_feat)
        i2 = F.interpolate(i2, i1.shape[2:], mode='bilinear', align_corners=True)
        i3 = self.conv3_i(img_layer3_feat)
        i3 = F.interpolate(i3, i1.shape[2:], mode='bilinear', align_corners=True)
        i4 = self.conv4_i(img_layer4_feat)
        i4 = F.interpolate(i4, i1.shape[2:], mode='bilinear', align_corners=True)
        iaspp = self.convaspp_i(img_aspp_feat)
        iaspp = F.interpolate(iaspp, i1.shape[2:], mode='bilinear', align_corners=True)
        i = torch.cat([i1, i2, i3, i4, iaspp], dim=1)
        i = self.convp_i(i)
        imgfea_vec = self.avgpool(i)

        flow_layer4_feat, flow_layer1_feat, flow_conv1_feat, flow_layer2_feat, flow_layer3_feat, flow_aspp_feat, course_flo = self.FlowBone(
            flow)

        f1 = self.conv1_f(flow_layer1_feat)
        f2 = self.conv2_f(img_layer2_feat)
        f2 = F.interpolate(f2, f1.shape[2:], mode='bilinear', align_corners=True)
        f3 = self.conv3_f(flow_layer3_feat)
        f3 = F.interpolate(f3, f1.shape[2:], mode='bilinear', align_corners=True)
        f4 = self.conv4_f(flow_layer4_feat)
        f4 = F.interpolate(f4, f1.shape[2:], mode='bilinear', align_corners=True)
        faspp = self.convaspp_f(flow_aspp_feat)
        faspp = F.interpolate(faspp, f1.shape[2:], mode='bilinear', align_corners=True)
        f = torch.cat([f1, f2, f3, f4, faspp], dim=1)
        f = self.convp_f(f)
        flowfea_vec = self.avgpool(f)

        Vec = torch.cat([imgfea_vec, flowfea_vec], dim=2)
        Vec = nn.Softmax(2)(Vec)
        imgfea_vec = Vec[:, :, 0, :].unsqueeze(2)
        flowfea_vec = Vec[:, :, 1, :].unsqueeze(2)

        course_pre = (torch.sum(imgfea_vec, dim=1).unsqueeze(1) * course_img + torch.sum(flowfea_vec, dim=1).unsqueeze(
            1) * course_flo) / (torch.sum(imgfea_vec, dim=1).unsqueeze(1) + torch.sum(flowfea_vec, dim=1).unsqueeze(1))
        course_pre1 = nn.Sigmoid()(course_pre)

        reduce_vec = imgfea_vec - flowfea_vec
        imgfea_vec = torch.where(reduce_vec < -0.6, torch.full_like(imgfea_vec, 0), imgfea_vec)
        flowfea_vec = torch.where(reduce_vec > 0.6, torch.full_like(imgfea_vec, 0), flowfea_vec)

        fea1 = imgfea_vec[:, 0, :, :].unsqueeze(1) * img_layer1_feat + flowfea_vec[:, 0, :, :].unsqueeze(
            1) * flow_layer1_feat
        fea2 = imgfea_vec[:, 1, :, :].unsqueeze(1) * img_layer2_feat + flowfea_vec[:, 1, :, :].unsqueeze(
            1) * flow_layer2_feat
        fea3 = imgfea_vec[:, 2, :, :].unsqueeze(1) * img_layer3_feat + flowfea_vec[:, 2, :, :].unsqueeze(
            1) * flow_layer3_feat
        fea4 = imgfea_vec[:, 3, :, :].unsqueeze(1) * img_layer4_feat + flowfea_vec[:, 3, :, :].unsqueeze(
            1) * flow_layer4_feat
        feaaspp = imgfea_vec[:, 4, :, :].unsqueeze(1) * img_aspp_feat + flowfea_vec[:, 4, :, :].unsqueeze(
            1) * flow_aspp_feat

        fea = self.blockaspp(feaaspp)
        fea = F.interpolate(fea, fea4.shape[2:], mode='bilinear', align_corners=True)
        fea = fea + fea4

        fea = self.block4(fea)
        fea = F.interpolate(fea, fea3.shape[2:], mode='bilinear', align_corners=True)
        fea = fea + fea3

        fea = self.block3(fea)
        fea = F.interpolate(fea, fea2.shape[2:], mode='bilinear', align_corners=True)
        fea = fea + fea2

        fea = self.block2(fea)
        fea = F.interpolate(fea, fea1.shape[2:], mode='bilinear', align_corners=True)
        fea = fea + fea1

        fea = self.block1(fea)
        course_pre1 = F.interpolate(course_pre1, fea.shape[2:], mode='bilinear', align_corners=True)
        fea = fea * course_pre1 + fea
        pre = self.last_conv(fea)
        pre = F.interpolate(pre, image.shape[2:], mode='bilinear', align_corners=True)
        course_pre = F.interpolate(course_pre, image.shape[2:], mode='bilinear', align_corners=True)
        course_img = F.interpolate(course_img, image.shape[2:], mode='bilinear', align_corners=True)
        course_flo = F.interpolate(course_flo, image.shape[2:], mode='bilinear', align_corners=True)
        return pre, course_pre, course_img, course_flo