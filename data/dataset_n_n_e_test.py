import h5py
import numpy as np
import torch
import os
import sys
from spatial_transform import Random_crop, ToTorchFormatTensor, Center_crop
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

    def __init__(self, h5_file, seq_len, scene_type, transform_flag=True):

        self.h5_file = h5_file

        self.seq_len = seq_len

        self.crop_size = 128

        self.scene_type = scene_type

        self.h5 = h5py.File(self.h5_file, "r")
        
        self.num_sequences = len(self.h5[scene_type + "/1/rainy"].keys()) - seq_len + 1
        # self.num_sequences = len(self.h5[scene_type + "/1/rainy"].keys()) // seq_len
        # self.num_sequences = 10
        
        if transform_flag:
            self.transform = torchvision.transforms.Compose([
                                Center_crop(self.crop_size, self.seq_len),
                                ToTorchFormatTensor()
                                ])
        else:
            self.transform = torchvision.transforms.Compose([
                                ToTorchFormatTensor()
                                ])

        

        # print("{} has {} sequences".format(self.scene_type, self.num_sequences))
    
    def open_h5(self):

        self.h5 = h5py.File(self.h5_file, "r")

    def get_sequence(self, A_idx):

        rainy_frame, clean_frame, rainy_event = [], [], []

        # start = time.time()
        
        for i in range(self.seq_len):
            
            rainy_frame.append(self.h5["{}/{}/{}/{:05d}".format(self.scene_type, "1", "rainy", A_idx + i)][:][:,:,::-1])
            clean_frame.append(self.h5["{}/{}/{:05d}".format(self.scene_type, "gt", A_idx + i)][:][:,:,::-1])
        
        # print("Access Frame time is {:f}".format(time.time() - start))
        # start = time.time()

        for i in range(self.seq_len -1):
            rainy_event.append(self.h5["{}/{}/{}/{:05d}".format(self.scene_type, "1", "voxel", A_idx + i)][:])
            
        # print("Access Event time is {:f}".format(time.time() - start))

        item = {"rainy": rainy_frame,
                "gt": clean_frame,
                "rainy_events": rainy_event}

        # print("Access time is {:f}".format(time.time() - start))
        # start = time.time()
        item = self.transform(item)

        # if dist.get_rank() == 0:
        #     print("Transform time is {:f}".format(time.time() - start))
        # print("Transform time is {:f}".format(time.time() - start))

        return item

    def __len__(self):

        return self.num_sequences
    
    def __getitem__(self, idx):

        A_idx = idx

        sequence = self.get_sequence(A_idx)

        return sequence


class SequenceDataset_04(Dataset):

    def __init__(self, h5_file, seq_len, scene_type):

        self.h5_file = h5_file

        self.seq_len = seq_len

        scene_len = {"a1":168, "a2":116, "a3":125 ,"a4":298, "b1":256, "b2":250, "b3":219, "b4":250}

        self.crop_size = 128

        self.scene_type = scene_type
        
        self.num_sequences = scene_len[scene_type] - seq_len + 1
        # self.num_sequences = len(self.h5[scene_type + "/1/rainy"].keys()) // seq_len
        # self.num_sequences = 10

        self.transform = torchvision.transforms.Compose([
                                Center_crop(self.crop_size, self.seq_len),
                                ToTorchFormatTensor(),
                                ])

        # print("{} has {} sequences".format(self.scene_type, self.num_sequences))
    
    def open_h5(self):

        self.h5 = h5py.File(self.h5_file, "r")

    def get_sequence(self, A_idx, transform=True):

        rainy_frame, clean_frame, rainy_event = [], [], []

        # start = time.time()
        
        for i in range(self.seq_len):
            
            rainy_frame.append(self.h5["{}/{}/{}/{:05d}".format(self.scene_type, "1", "rainy", A_idx + i)][:][:,:,::-1])
            clean_frame.append(self.h5["{}/{}/{:05d}".format(self.scene_type, "gt", A_idx + i)][:][:,:,::-1])
        
        # print("Access Frame time is {:f}".format(time.time() - start))
        # start = time.time()

        for i in range(self.seq_len -1):
            rainy_event.append(self.h5["{}/{}/{}/{:05d}".format(self.scene_type, "1", "voxel", A_idx + i)][:])
            
        # print("Access Event time is {:f}".format(time.time() - start))

        item = {"rainy": rainy_frame,
                "gt": clean_frame,
                "rainy_events": rainy_event}

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

        sequence = self.get_sequence(A_idx)

        return sequence
