import torch
from collections import OrderedDict
from torch.optim import Adam, SGD
from model.Model import Model
import torchvision.utils as vutils
import torch.nn.functional as F
import os
import numpy as np
import cv2
from PIL import Image
from model.resnet_aspp import ResNet_ASPP

EPSILON = 1e-8
p = OrderedDict()

p['lr_bone'] = 5e-5  # Learning rate
p['lr_branch'] = 0.025
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
lr_decay_epoch = [9, 20]
nAveGrad = 10  # Update the weights once in 'nAveGrad' forward passes
showEvery = 50
tmp_path = 'tmp_out'

class Solver(object):
    def __init__(self, train_loader, test_loader, config, save_fold=None):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.save_fold = save_fold

        self.build_model()

        if config.mode == 'test':
            self.net_bone.eval()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  # 返回一个tensor变量内所有元素个数
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        print('mode: {}'.format(self.config.mode))
        print('------------------------------------------')
        self.net_bone = Model(3, self.config.mode)
        if self.config.cuda:
            self.net_bone = self.net_bone.cuda()

        if self.config.mode == 'train':
            if self.config.model_path != '':
                assert (os.path.exists(self.config.model_path)), ('please import correct pretrained model path!')
                self.net_bone.load_pretrain_model(self.config.model_path)
        else:
            assert (self.config.model_path != ''), ('Test mode, please import pretrained model path!')
            assert (os.path.exists(self.config.model_path)), ('please import correct pretrained model path!')
            self.net_bone.load_pretrain_model(self.config.model_path)

        self.lr_bone = p['lr_bone']
        self.lr_branch = p['lr_branch']
        self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()), lr=self.lr_bone,
                                   weight_decay=p['wd'])
        print('------------------------------------------')
        self.print_network(self.net_bone, 'DSNet')
        print('------------------------------------------')

    def test(self):

        if not os.path.exists(self.save_fold):
            os.makedirs(self.save_fold)
        for i, data_batch in enumerate(self.test_loader):
            image, flow, name, split, size = data_batch['image'], data_batch['flow'], data_batch['name'], data_batch[
                'split'], data_batch['size']
            dataset = data_batch['dataset']

            if self.config.cuda:
                image, flow = image.cuda(), flow.cuda()
            with torch.no_grad():

                pre, pre2, pre3, pre4 = self.net_bone(image, flow)

                for i in range(self.config.test_batch_size):
                    presavefold = os.path.join(self.save_fold, dataset[i], split[i])
                    if not os.path.exists(presavefold):
                        os.makedirs(presavefold)
                    pre1 = torch.nn.Sigmoid()(pre[i])
                    pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1))
                    pre1 = np.squeeze(pre1.cpu().data.numpy()) * 255
                    pre1 = cv2.resize(pre1, (size[0][1], size[0][0]))
                    cv2.imwrite(presavefold + '/' + name[i], pre1)

    def train(self):

        # 一个epoch中训练iter_num个batch
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        for epoch in range(self.config.epoch):
            r_img_loss,  r_flo_loss, r_pre_loss, r_sal_loss, r_sum_loss= 0, 0, 0, 0, 0
            self.net_bone.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                image, label, flow = data_batch['image'], data_batch['label'], data_batch['flow']
                if image.size()[2:] != label.size()[2:]:
                    print("Skip this batch")
                    continue
                if self.config.cuda:
                    image, label, flow = image.cuda(), label.cuda(), flow.cuda()

                sal_loss1 = []
                sal_loss2 = []
                sal_loss3 = []
                sal_loss4 = []

                pre1, pre2, pre3, pre4 = self.net_bone(image, flow)

                sal_loss1.append(F.binary_cross_entropy_with_logits(pre1, label, reduction='sum'))
                sal_loss2.append(F.binary_cross_entropy_with_logits(pre2, label, reduction='sum'))
                sal_loss3.append(F.binary_cross_entropy_with_logits(pre3, label, reduction='sum'))
                sal_loss4.append(F.binary_cross_entropy_with_logits(pre4, label, reduction='sum'))
                sal_img = sum(sal_loss3) / (nAveGrad * self.config.batch_size)
                sal_flo = sum(sal_loss4) / (nAveGrad * self.config.batch_size)
                sal_pre = sum(sal_loss2) / (nAveGrad * self.config.batch_size)
                sal_final = sum(sal_loss1) / (nAveGrad * self.config.batch_size)

                r_img_loss += sal_img.data
                r_flo_loss += sal_flo.data
                r_pre_loss += sal_pre.data
                r_sal_loss += sal_final.data

                sal_loss = (sum(sal_loss1) + sum(sal_loss2) + sum(sal_loss3) + sum(sal_loss4)) / (
                            nAveGrad * self.config.batch_size)
                r_sum_loss += sal_loss.data
                loss = sal_loss
                loss.backward()
                aveGrad += 1

                if aveGrad % nAveGrad == 0:
                    self.optimizer_bone.step()
                    self.optimizer_bone.zero_grad()
                    aveGrad = 0

                if i % showEvery == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  Loss ||  img : %10.4f  ||  flo : %10.4f ||  pre : %10.4f || sal : %10.4f || sum : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num,
                        r_img_loss * (nAveGrad * self.config.batch_size) / showEvery,
                        r_flo_loss * (nAveGrad * self.config.batch_size) / showEvery,
                        r_pre_loss * (nAveGrad * self.config.batch_size) / showEvery,
                        r_sal_loss * (nAveGrad * self.config.batch_size) / showEvery,
                        r_sum_loss * (nAveGrad * self.config.batch_size) / showEvery)  )

                    print('Learning rate: ' + str(self.lr_bone))
                    r_img_loss,  r_flo_loss, r_pre_loss, r_sal_loss, r_sum_loss= 0, 0, 0, 0, 0

                if i % 50 == 0:
                    vutils.save_image(torch.sigmoid(pre1.data), tmp_path + '/iter%d-sal-0.jpg' % i,
                                      normalize=True, padding=0)
                    # vutils.save_image(torch.sigmoid(edge_out.data), tmp_path + '/iter%d-edge-0.jpg' % i,
                    #                   normalize=True, padding=0)
                    vutils.save_image(image.data, tmp_path + '/iter%d-sal-data.jpg' % i, padding=0)
                    vutils.save_image(label.data, tmp_path + '/iter%d-sal-target.jpg' % i, padding=0)

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(),
                           '%s/models/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))

            if epoch in lr_decay_epoch:
                self.lr_bone = self.lr_bone * 0.2
                self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()),
                                           lr=self.lr_bone, weight_decay=p['wd'])

        torch.save(self.net_bone.state_dict(), '%s/models/final_bone.pth' % self.config.save_fold)



