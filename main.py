
import argparse
import os
from dataset import Dataset
import torch
from solver import Solver
from torchvision import transforms
import transform
from torch.utils import data


def main(config):
    composed_transforms_ts = transforms.Compose([
        transform.FixedResize(size=(config.input_size, config.input_size)),
        transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform.ToTensor()])
    if config.mode == 'train':

        dataset = Dataset(datasets=['DAVIS'], transform=composed_transforms_ts, mode='train')
        train_loader = data.DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_thread, drop_last=True, shuffle=True)

        if not os.path.exists("%s/%s" % (config.save_fold, 'models')):
            os.mkdir("%s/%s" % (config.save_fold, 'models'))
        config.save_fold = "%s/%s" % (config.save_fold, 'models')
        train = Solver(train_loader, None, config)
        train.train()

    elif config.mode == 'test':

        dataset = Dataset(datasets=config.test_dataset, transform=composed_transforms_ts, mode='test')

        test_loader = data.DataLoader(dataset, batch_size=config.test_batch_size, num_workers=config.num_thread, drop_last=True, shuffle=False)
        test = Solver(train_loader=None, test_loader=test_loader, config=config, save_fold=config.testsavefold)
        test.test()


    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser()

    # Hyper-parameters11111111111111111111
    print(torch.cuda.is_available())

    parser.add_argument('--cuda', type=bool, default=True)  # 是否使用cuda

    # train
    parser.add_argument('--epoch', type=int, default=25)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='')

    parser.add_argument('--save_fold', type=str, default='./result')  # 训练过程中输出的保存路径
    parser.add_argument('--input_size', type=int, default=512)

    # test
    parser.add_argument('--test_dataset', type=list, default=['ViSal', 'DAVIS', 'FBMS', 'SegV2', 'VOS'])
    parser.add_argument('--testsavefold', type=str, default='./prediction')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    config = parser.parse_args()

    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
