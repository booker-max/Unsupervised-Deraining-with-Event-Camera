import h5py
import numpy as np
import torch
import os
import sys
from spatial_transform import Random_crop, ToTorchFormatTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import time
import torch.distributed as dist

def make_dataset(source, mode):
    #root:'./datasets/hmdb51_frames'
    #source:'./datasets/settings/hmdb51/train_rgb_split1.txt'
    if not os.path.exists(source):
        print("Setting file %s for hmdb51 dataset doesn't exist." % (source))
        sys.exit()
    else:
        rgb_samples = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()[0]
                rgb_samples.append(line_info)
    
        print('{}: {} sequences have been loaded'.format(mode, len(rgb_samples)))
    return rgb_samples

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SequenceDataset(Dataset):

    def __init__(self, h5_file, train_txt, seq_len):

        # self.h5 = h5py.File(h5_file, "r")

        self.h5_file = h5_file

        self.crop_size = 128

        self.seq_len = seq_len

        self.sequence_samples = make_dataset(train_txt, "train")

        self.num_sequences = len(self.sequence_samples)

        self.gan_list = list(range(self.num_sequences))

        np.random.shuffle(self.gan_list)

        ###
        self.transform = torchvision.transforms.Compose([
                                Random_crop(self.crop_size, self.seq_len),
                                ToTorchFormatTensor(),
                                ])

        # print(self.gan_list)
        # import pdb
        # pdb.set_trace()
    
    def open_h5(self):

        self.h5 = h5py.File(self.h5_file, "r")

    def get_sequence(self, A_idx, B_idx, transform=True):

        key, sub_key, img_idx = self.sequence_samples[A_idx].split('/')
        gan_key, _, gan_img_idx = self.sequence_samples[B_idx].split('/')
        
        # if dist.get_rank() == 0:
        # print("A is {}, B is {}".format(self.sequence_samples[A_idx], self.sequence_samples[B_idx]))
        #     start = time.time()

        rainy_frame, clean_frame, gan_clean_frame, rainy_event, gan_clean_event = [], [], [], [], []

        # start = time.time()
        
        for i in range(self.seq_len):

            rainy_frame.append(self.h5["{}/{}/{}/{:05d}".format(key, sub_key, "rainy", int(img_idx) + i)][:][:,:,::-1])
            clean_frame.append(self.h5["{}/{}/{:05d}".format(key, "gt", int(img_idx) + i)][:][:,:,::-1])
            gan_clean_frame.append(self.h5["{}/{}/{:05d}".format(gan_key, "gt", int(gan_img_idx) + i)][:][:,:,::-1])
        
        # print("Access Frame time is {:f}".format(time.time() - start))
        # start = time.time()

        for i in range(self.seq_len -1):
            rainy_event.append(self.h5["{}/{}/{}/{:05d}".format(key, sub_key, "voxel", int(img_idx) + i)][:])
            gan_clean_event.append(self.h5["{}/{}/{:05d}".format(gan_key, "clean_voxel", int(gan_img_idx) + i)][:])
            
        # print("Access Event time is {:f}".format(time.time() - start))

        item = {"rainy": rainy_frame,
                "gt": clean_frame,
                "rainy_events": rainy_event,
                "gan_gt": gan_clean_frame,
                "gan_events": gan_clean_event}

        # print("Access time is {:f}".format(time.time() - start))
        # start = time.time()

        if transform:

            item = self.transform(item)

        # if dist.get_rank() == 0:
        #     print("Transform time is {:f}".format(time.time() - start))
        # print("Transform time is {:f}".format(time.time() - start))

        return item

    def __len__(self):

        return self.num_sequences
    
    def __getitem__(self, idx):

        if not hasattr(self, "h5"):
            self.open_h5()

        A_idx = idx
        B_idx = self.gan_list[idx]

        # print("A_idx is {}, B_idx is {}".format(A_idx, B_idx))

        sequence = self.get_sequence(A_idx, B_idx)

        return sequence
