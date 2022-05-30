import glob
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch
import h5py
from scipy.io import loadmat

patch_size, stride = 40, 10
aug_times = 1
scales = [1]
batch_size = 128


class DenoisingDataset(Dataset):
    def __init__(self):
        super(DenoisingDataset, self).__init__()
        h5f_real = h5py.File('data_real.h5', 'r')
        h5f_label = h5py.File('data_label.h5', 'r')
        self.real_keys = list(h5f_real.keys())
        self.label_keys = list(h5f_label.keys())
        self.data_len = len(h5f_real.keys())
        h5f_real.close()
        h5f_label.close()

    def __getitem__(self, index):
        h5f_real = h5py.File('data_real.h5', 'r')
        h5f_label = h5py.File('data_label.h5', 'r')
        real_key = self.real_keys[index]
        label_key = self.label_keys[index]
        item_real = np.array(h5f_real[real_key], np.float32)
        item_label = np.array(h5f_label[label_key], np.float32)
        h5f_real.close()
        h5f_label.close()
        batch_x = torch.from_numpy(item_real)
        batch_y = torch.from_numpy(item_label)
        return batch_x, batch_y

    def __len__(self):
        disregard = self.data_len - self.data_len // batch_size * batch_size  # because of batch loading
        return self.data_len - disregard


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    img_real = cv2.imread(file_name[:-8] + 'Real.JPG', cv2.IMREAD_COLOR)
    img_label = cv2.imread(file_name, cv2.IMREAD_COLOR)
    w, h, c = img_real.shape
    real_patches, label_patches = [], []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        real_scaled = cv2.resize(img_real, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        label_scaled = cv2.resize(img_label, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, w_scaled-patch_size+1, 3 * stride):
            for j in range(0, h_scaled-patch_size+1, 3 * stride):
                x_real = real_scaled[i:i+patch_size, j:j+patch_size]
                x_label = label_scaled[i:i + patch_size, j:j + patch_size]
                for k in range(0, aug_times):
                    mode = np.random.randint(0, 8)
                    real_aug = data_aug(x_real, mode)
                    label_aug = data_aug(x_label, mode)
                    real_aug = np.transpose(real_aug, (2, 0, 1))
                    label_aug = np.transpose(label_aug, (2, 0, 1))
                    real_patches.append(real_aug)
                    label_patches.append(label_aug)
    return real_patches, label_patches


def datagenerator(data_dir='data/RealTrain', verbose=False):
    file_list = glob.glob(data_dir + '/*')
    # print('file_list', len(file_list))
    h5f_real = h5py.File('data_real.h5', 'w')
    h5f_label = h5py.File('data_label.h5', 'w')
    for i in range(len(file_list)):
        label_data, real_data = list(), list()
        if 'mean' in file_list[i]:
            real_patches, label_patches = gen_patches(file_list[i])
            print("len_patches:", len(label_patches))
            for j in range(len(label_patches)):
                real_data.append(real_patches[j])
                label_data.append(label_patches[j])
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
        real_data = np.array(real_data, dtype='uint8')
        label_data = np.array(label_data, dtype='uint8')
        real_data = real_data.astype('float32') / 255.0
        label_data = label_data.astype('float32') / 255.0
        data_len = len(h5f_real.keys())
        print("data_len", data_len)
        for i in range(len(real_data)):
            h5f_real.create_dataset(str(data_len+i), data=real_data[i])
            h5f_label.create_dataset(str(data_len+i), data=label_data[i])

    h5f_real.close()
    h5f_label.close()

    print('^_^-training data finished-^_^')


if __name__ == '__main__':
    datagenerator(data_dir='data/RealTrain')
    h5f_real = h5py.File('data_real.h5', 'r')
    print(len(h5f_real.keys()))