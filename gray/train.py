import os
import argparse
import glob
import re
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from models_rddcnn import DnCNN
import utils
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset


# Params
parser = argparse.ArgumentParser(description='DnCNN')
parser.add_argument('--model', default='RDDCNN_', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--patch_size', default=40, type=int, help='batch size')
parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
parser.add_argument('--sigma', default=50, type=int, help='noise level')
parser.add_argument('--noise_range_low', default=0, type=int, help='noise lower bound in blind mode')
parser.add_argument('--noise_range_high', default=55, type=int, help='noise higher bound in blind mode')
parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--mode', default='S', type=str, help='training mode S for known noise level, B for unknown')
args = parser.parse_args()

batch_size = args.batch_size
patch_size = args.patch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma


class sum_squared_error(_Loss):
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def main():
    print('===> Building model')
    model = DnCNN()
    
    model.train()
    criterion_mse = sum_squared_error()
    if cuda:
        model = model.cuda()
        device_ids = [0]
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
        criterion_mse = criterion_mse.cuda()
    
    initial_epoch = utils.findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        # model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    
    for epoch in range(initial_epoch, n_epoch):
        xs = dg.datagenerator(data_dir=args.train_data, mode=args.mode)
        xs = xs.astype('float32')/255.0
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW

        DDataset = DenoisingDataset(xs)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, batch_x in enumerate(DLoader):
            optimizer.zero_grad()

            if args.mode == 'S':
                noise = torch.randn(batch_x.size()).mul_(args.sigma/255.0)
            if args.mode == 'B':
                noise = torch.zeros(batch_x.size())
                stdN = np.random.uniform(args.noise_range_low, args.noise_range_high, size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0, :, :, :].size()
                    noise[n, :, :, :] = torch.randn(sizeN).mul_(stdN[n]/255.0)
            batch_y = batch_x + noise

            batch_x, batch_y = Variable(batch_x, requires_grad=True), Variable(batch_y, requires_grad=True)
            if cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            loss = criterion_mse(model(batch_y), batch_x) / batch_size
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            #if n_count % 10 == 0:
                #print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.size(0)//batch_size, loss.item()/batch_size))
        elapsed_time = time.time() - start_time
        scheduler.step(epoch)  # step to the learning rate in this epoch

        utils.log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        # torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))


if __name__ == "__main__":
    save_dir = os.path.join('models', args.model + 'sigma' + str(args.sigma))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    main()
