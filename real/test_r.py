import cv2
import os
import argparse
import glob
import numpy as np
import torch
import utils
import torch.nn as nn
from torch.autograd import Variable
from models_rddcnn_r import DnCNN
from skimage.io import imread, imsave
import time
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['RealTest'], help='directory of test dataset')
    parser.add_argument('--model_dir', default='models', help='directory of the model')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def main():
    args = parse_args()
    model_name = 'real.pth'
    model = torch.load(os.path.join(args.model_dir, model_name))
    utils.log('load ' + model_name)
    model.eval()  # evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        ans = []
        for set_cur in args.set_names:
            psnrs = []
            for im in os.listdir(os.path.join(args.set_dir, set_cur)):
                if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                    if 'mean' in im:
                        real_im = im[:-8] + 'real.png'
                        label = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32) / 255.0
                        real = np.array(imread(os.path.join(args.set_dir, set_cur, real_im)),
                                        dtype=np.float32) / 255.0
                        real_temp = np.expand_dims(np.transpose(real, (2, 0, 1)), axis=0)
                        real_ = torch.from_numpy(real_temp)

                        torch.cuda.synchronize()
                        start_time = time.time()
                        real_ = real_.cuda()
                        label_ = model(real_)  # inference
                        label_ = label_[0, :, :, :]
                        label_ = label_.cpu()
                        label_ = label_.detach().numpy().astype(np.float32)
                        label_ = np.transpose(label_, (1, 2, 0))
                        torch.cuda.synchronize()
                        elapsed_time = time.time() - start_time
                        # print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))

                        # show(x_)
                        # print(x.shape, y.shape)
                        # psnr_y = peak_signal_noise_ratio(label, real)
                        # print(psnr_y)
                        psnr_x_ = peak_signal_noise_ratio(label, label_)
                        psnrs.append(psnr_x_)
                        # print(im, psnr_x_)
            psnr_avg = np.mean(psnrs)
            ans.append(psnr_avg)
        # print(psnrs)
        utils.log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB'.format(set_cur, round(psnr_avg, 2)))


if __name__ == "__main__":
    main()
