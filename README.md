# DS-Net
DSNet: Dynamic Spatiotemporal Network for Video Salient Object Detection
### For training:
1. Download [pretrained ResNet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth) to './model/resnet/pre_train/'.

2. Organize each dataset according to the organization format in the './dataset/DAVIS'.

3. Install [FlowNet2.0](https://github.com/NVIDIA/flownet2-pytorch) and genrate correct optical flow images.

4. Start to train with `python main.py --mode train`.

### For testing:
1. Download [pretrained model](https://pan.baidu.com/s/1DdDIGhPYTT5_-cwvCdEhLA) (87xl) to './model/pretrained/'

2. Generate saliency maps for SOD dataset by `python main.py --mode test`.


