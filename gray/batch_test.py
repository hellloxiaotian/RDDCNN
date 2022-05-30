import cv2
import os
import argparse
import glob
import numpy as np
import torch
import utils
import torch.nn as nn
from torch.autograd import Variable
from models_rddcnn import DnCNN
from skimage.io import imread, imsave
import time
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/', type=str, help='directory of test dataset')
    # parser.add_argument('--set_names', default=['Set68', 'Set12'], help='directory of test dataset')
    # parser.add_argument('--set_names', default=['cut68'], help='directory of test dataset')
    parser.add_argument('--set_names', default=['running_time'], help='directory of test dataset')
    parser.add_argument('--sigma', default=15, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('models', 'RDDCNN_blindsigma25'), help='directory of the model')
    parser.add_argument('--model_name', default='model_180.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=0, type=int, help='save the de-noised image, 1 or 0')
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
    model = DnCNN().cuda()
    device_ids = [0]
    model = nn.DataParallel(model, device_ids=device_ids)
    for i in range(1):
        model_name = 'model_' + '%03d' % (95 - i) + '.pth'
        model.load_state_dict(torch.load(os.path.join(args.model_dir, model_name)))
        # model = torch.load(os.path.join(args.model_dir, model_name))
        print('load ' + model_name)
        model.eval()  # evaluation mode
        if torch.cuda.is_available():
            model = model.cuda()
        with torch.no_grad():
            ans = []
            for set_cur in args.set_names:
                psnrs = []
                for im in os.listdir(os.path.join(args.set_dir, set_cur)):
                    if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):

                        x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32)/255.0
                        np.random.seed(seed=8)  # for reproducibility
                        y = x + np.random.normal(0, args.sigma/255.0, x.shape)  # Add Gaussian noise without clipping
                        y = y.astype(np.float32)
                        start_time = time.time()
                        y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

                        torch.cuda.synchronize()
                        y_ = y_.cuda()
                        x_ = model(y_)  # inference
                        x_ = x_.view(y.shape[0], y.shape[1])
                        x_ = x_.cpu()
                        x_ = x_.detach().numpy().astype(np.float32)
                        torch.cuda.synchronize()
                        elapsed_time = time.time() - start_time
                        print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))
                        
                        psnr_x_ = peak_signal_noise_ratio(x, x_)
                        psnr_y_ = peak_signal_noise_ratio(x, y)
                        if args.save_result:
                            name, ext = os.path.splitext(im)
                            show(np.hstack((y, x_)))  # show the image
                            save_result(x_, path=os.path.join(args.result_dir, set_cur, name+'_%.3f'% psnr_x_ +ext))  # save the denoised image
                            save_result(y, path=os.path.join(args.result_dir, set_cur, name +'_%.3f'% psnr_y_+ ext))
                        psnrs.append(psnr_x_)
                psnr_avg = np.mean(psnrs)
                ans.append(psnr_avg)
                # print(psnrs)
            for index in range(len(args.set_names)):
                print("%10s: %.3fdB " % (args.set_names[index], ans[index]), end='')
            print('')


if __name__ == "__main__":
    main()
