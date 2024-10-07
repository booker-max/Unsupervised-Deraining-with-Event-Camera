import torchvision
import numpy.random as random
import torch
import numpy as np
import cv2

class Random_crop(object):

    def __init__(self, input_size, seq_len):

        self.input_size = input_size
        self.seq_len = seq_len

    def __call__(self, sequence):
        ####img_group是concat后的矩阵

        h, w, _ = sequence["rainy"][0].shape

        h1 = random.randint(0, h - self.input_size)
        w1 = random.randint(0, w - self.input_size)
        
        # print("*"*80)
        # print("h1 is {},w1 is {}".format(h1, w1))
        # print("*"*80)
        # print("h is {},w is {}".format(h, w))

        for key in sequence.keys():

            if key.split('_')[-1] != "events":

                for idx in range(self.seq_len):

                    sequence[key][idx] = sequence[key][idx][h1:(h1 + self.input_size), w1:(w1 + self.input_size), :]
            else:
                
                for idx in range(self.seq_len-1):

                    sequence[key][idx] = sequence[key][idx][:, h1:(h1 + self.input_size), w1:(w1 + self.input_size)]

        return sequence

#单独为seq=1时设计的
class Random_crop_S(object):

    def __init__(self, input_size, seq_len):

        self.input_size = input_size
        self.seq_len = seq_len

    def __call__(self, sequence):
        ####img_group是concat后的矩阵

        h, w, _ = sequence["rainy"].shape

        h1 = random.randint(0, h - self.input_size)
        w1 = random.randint(0, w - self.input_size)

        for key in sequence.keys():
            sequence[key] = sequence[key][h1:(h1 + self.input_size), w1:(w1 + self.input_size), :]

        return sequence

class Random_crop_list(object):

    def __init__(self, input_size, seq_len):

        self.input_size = input_size
        self.seq_len = seq_len

    def __call__(self, sequence):
        ####img_group是concat后的矩阵

        h, w, _ = sequence["rainy"][0].shape

        h1 = random.randint(0, h - self.input_size)
        w1 = random.randint(0, w - self.input_size)
        
        # print("*"*80)
        # print("h1 is {},w1 is {}".format(h1, w1))
        # print("*"*80)
        # print("h is {},w is {}".format(h, w))

        for key in sequence.keys():

            for idx in range(len(sequence[key])):

                if key.split('_')[-1] != "events":

                    sequence[key][idx] = sequence[key][idx][h1:(h1 + self.input_size), w1:(w1 + self.input_size), :]
                
                else:

                    sequence[key][idx] = sequence[key][idx][:, h1:(h1 + self.input_size), w1:(w1 + self.input_size)]

        return sequence

class Center_crop(object):

    def __init__(self, input_size, seq_len):

        self.input_size = input_size
        self.seq_len = seq_len

    def __call__(self, sequence):
        ####img_group是concat后的矩阵

        h, w, _ = sequence["rainy"][0].shape
        
        if not isinstance(self.input_size, list):

            h1 = int(round((h - self.input_size) / 2.))
            w1 = int(round((w - self.input_size) / 2.))

            for key in sequence.keys():

                if key.split('_')[-1] != "events":

                    for idx in range(self.seq_len):

                        sequence[key][idx] = sequence[key][idx][h1:(h1 + self.input_size), w1:(w1 + self.input_size), :]
                else:
                    
                    for idx in range(self.seq_len-1):

                        sequence[key][idx] = sequence[key][idx][:, h1:(h1 + self.input_size), w1:(w1 + self.input_size)]

        else:

            h1 = int(round((h - self.input_size[0]) / 2.))
            w1 = int(round((w - self.input_size[1]) / 2.))

            for key in sequence.keys():

                if key.split('_')[-1] != "events":

                    for idx in range(self.seq_len):

                        sequence[key][idx] = sequence[key][idx][h1:(h1 + self.input_size[0]), w1:(w1 + self.input_size[1]), :]
                else:
                    
                    for idx in range(self.seq_len-1):

                        sequence[key][idx] = sequence[key][idx][:, h1:(h1 + self.input_size[0]), w1:(w1 + self.input_size[1])]

        # print("h1 is {},w1 is {}".format(h1, w1))

        return sequence

class ToTorchFormatTensor_Norm(object):

    def __init__(self, mean, std, mean_E, std_E):

        self.transform = torchvision.transforms.ToTensor()
        self.mean, self.std, self.mean_E, self.std_E = mean, std, mean_E, std_E

    def __call__(self, sequence):

        for key in sequence.keys():

            if key.split('_')[-1] != "events":

                sequence[key] = self.transform(np.concatenate(sequence[key], -1))
                
                if True:

                    rep_mean = self.mean * (sequence[key].size()[0]//len(self.mean))
                    rep_std = self.std * (sequence[key].size()[0]//len(self.std))

                    for t, m, s in zip(sequence[key], rep_mean, rep_std):
                        t.sub_(m).div_(s)
            else:

                sequence[key] = torch.from_numpy(np.concatenate(sequence[key], 0))

                if True:

                    rep_mean = self.mean_E * (sequence[key].size()[0]//len(self.mean_E))
                    rep_std = self.std_E * (sequence[key].size()[0]//len(self.std_E))

                    for t, m, s in zip(sequence[key], rep_mean, rep_std):
                        t.sub_(m).div_(s)

        return sequence

class ToTorchFormatTensor(object):

    def __init__(self):

        self.transform = torchvision.transforms.ToTensor()

    def __call__(self, sequence):

        for key in sequence.keys():

            if key.split('_')[-1] != "events":

                sequence[key] = self.transform(np.concatenate(sequence[key], -1))
            
            else:

                sequence[key] = torch.from_numpy(np.concatenate(sequence[key], 0))

        return sequence

class ToTorchFormatTensor_S(object):

    def __init__(self):

        self.transform = torchvision.transforms.ToTensor()

    def __call__(self, sequence):

        for key in sequence.keys():

            sequence[key] = self.transform(sequence[key])

        return sequence


class ToTensorList(object):

    def __init__(self):

        self.transform = torchvision.transforms.ToTensor()

    def __call__(self, sequence):

        for key in sequence.keys():

            for idx in range(len(sequence[key])):

                if key.split('_')[-1] != "events":

                    sequence[key][idx] = cv2.cvtColor(sequence[key][idx], cv2.COLOR_BGR2RGB)
                    sequence[key][idx] = self.transform(sequence[key][idx])
                else:
                    sequence[key][idx] = torch.from_numpy(sequence[key][idx])
        
            # sequence[key] = torch.cat(sequence[key],0)

        return sequence


"""
ToTensor_list表示不会concatenate之后再totensor, 而是逐列表元素
"""

class ToTorchFormatTensor_list(object):

    def __init__(self):

        self.transform = torchvision.transforms.ToTensor()

    def __call__(self, sequence):

        for key in sequence.keys():

            if key.split('_')[-1] != "events":

                for idx in range(len(sequence[key])):

                    sequence[key][idx] = self.transform(np.ascontiguousarray(sequence[key][idx]))
            
            else:
                
                for idx in range(len(sequence[key])):

                    sequence[key] = torch.from_numpy(sequence[key][idx])

        return sequence

if True:

    class ToTorchFormatTensor_1(object):

        def __init__(self):

            self.transform = torchvision.transforms.ToTensor()

        def __call__(self, sequence):

            for key in sequence.keys():

                if key.split('_')[-1] != "events":

                    sequence[key] = self.transform(np.concatenate(sequence[key], -1))
                
                else:

                    sequence[key] = torch.from_numpy(np.concatenate(sequence[key], 0))

            return sequence


    class Random_crop_1(object):

        def __init__(self, input_size, seq_len):

            self.input_size = input_size
            self.seq_len = seq_len

        def __call__(self, sequence):
            ####img_group是concat后的矩阵

            _, h, w = sequence["rainy"].shape

            h1 = random.randint(0, h - self.input_size)
            w1 = random.randint(0, w - self.input_size)

            # print("h1 is {},w1 is {}".format(h1, w1))

            for key in sequence.keys():

                sequence[key] = sequence[key][:, h1:(h1 + self.input_size), w1:(w1 + self.input_size)]

            return sequence

    class ToTorchFormatTensor_2(object):

        def __init__(self, seq_len):

            self.transform = torchvision.transforms.ToTensor()
            self.seq_len = seq_len

        def __call__(self, sequence):

            for key in sequence.keys():

                if key.split('_')[-1] != "events":

                    for idx in range(self.seq_len):

                        sequence[key][idx] = self.transform(sequence[key][idx].copy())
                
                else:

                    for idx in range(self.seq_len-1):

                        sequence[key][idx] = torch.from_numpy(sequence[key][idx])

            return sequence

    class RandomCrop_fix(object):

        def __init__(self, image_size, crop_size):
            self.ch, self.cw = crop_size, crop_size
            ih, iw = image_size

            self.h1 = random.randint(0, ih - self.ch)
            self.w1 = random.randint(0, iw - self.cw)

            self.h2 = self.h1 + self.ch
            self.w2 = self.w1 + self.cw
            
        def __call__(self, sequence):

            for key in sequence.keys():

                if key.split('_')[-1] != "events":

                    sequence[key] = sequence[key][self.h1 : self.h2, self.w1 : self.w2, :]
            
                else:

                    sequence[key] = sequence[key][:, self.h1 : self.h2, self.w1 : self.w2]

            return sequence
           
    class ToTorchFormatTensor_fix(object):

        def __init__(self):

            self.transform = torchvision.transforms.ToTensor()

        def __call__(self, sequence):

            for key in sequence.keys():

                if key.split('_')[-1] != "events":

                    sequence[key] = self.transform(sequence[key].copy())
            
                else:

                    sequence[key] = torch.from_numpy(sequence[key])
            
            return sequence