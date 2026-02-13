import glob
import os
import pickle as pkl
import re
from skimage import measure
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
import torch

class Random_erasing2:
    """with a probability prob, erases a proportion prop of the image"""

    def __init__(self, prob, prop):
        self.prob = prob
        self.prop = prop

    def __call__(self, x):
        if np.random.rand() < self.prob:
            labels = measure.label(x.numpy() > 0, connectivity=1)
            regions = measure.regionprops(labels)
            pics_suppr = np.random.rand(len(regions)) < self.prop
            for k in range(len(regions)):
                if pics_suppr[k]:
                    try:
                        _, y1, x1, _, y2, x2 = regions[k].bbox
                    except:
                        raise Exception(regions[k].bbox)
                    x[:, y1:y2, x1:x2] *= regions[k].image == False
            return x

        return x


class Random_int_noise:
    """With a probability prob, adds a gaussian noise to the image """

    def __init__(self, prob, maximum):
        self.prob = prob
        self.minimum = 1 / maximum
        self.delta = maximum - self.minimum

    def __call__(self, x):
        if np.random.rand() < self.prob:
            return x * (self.minimum + torch.rand_like(x) * self.delta)
        return x


class Random_shift_rt:
    """With a probability prob, shifts verticaly the image depending on a gaussian distribution"""

    def __init__(self, prob, mean, std):
        self.prob = prob
        self.mean = torch.tensor(float(mean))
        self.std = float(std)

    def __call__(self, x):
        if np.random.rand() < self.prob:
            shift = torch.normal(self.mean, self.std)
            return transforms.functional.affine(x, 0, [0, shift], 1, [0, 0])
        return x

class Log_normalisation:
    """Log normalisation of intensities"""

    def __init__(self, eps=1e-5):
        self.epsilon = eps

    def __call__(self, x):
        return torch.log(x+1+self.epsilon)/torch.log(torch.max(x)+1+self.epsilon)


class To_float:
    "Repeat tensor along new dim"

    def __init__(self):
        pass

    def __call__(self, x):
        return x.to(torch.float32)

def pkl_loader(path):
    with open(path, 'rb') as f:
        sample = pkl.load(f)
    return sample


class Antibio_Dataset(DatasetFolder):
    def __init__(self, root,label_path,label_col,augment=False, model_type=None):
        self.root = root
        self.instances, self.labels , self.sample_name= self.make_dataset('.pkl',label_path,label_col)
        self.loader = pkl_loader
        self.type = model_type
        if self.type == 'ms1':
            if augment:
                self.transform_img = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Resize((256,256)),
                                                         transforms.Normalize((0.246), (0.210)),
                                                         Random_shift_rt(1,0,3),
                                                         Random_int_noise(1, 2)])

            else :
                self.transform_img = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Resize((256,256)),
                                                         transforms.Normalize((0.246), (0.210)),])
        elif self.type == 'ms2':
            if augment:#augment 3D a définir
                self.transform_img = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Resize((256, 256)),
                                                         transforms.Normalize((1.0207), (1.0011)),
                                                         Random_shift_rt(1, 0, 3),
                                                         Random_int_noise(1, 2)])

            else:
                self.transform_img = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Resize((256, 256)),
                                                         transforms.Normalize((1.0207), (1.0011)),
                                                        ])
        else :
            self.transform_img = None
        self.label_col = label_col
        self.classes = list(set(self.labels))
        self.classes.sort()


    def __getitem__(self, index: int):
        label = self.labels[index]
        path = self.instances[index]
        name = self.sample_name[index]
        sample = self.loader(path)
        if self.type == 'ms2':
            sample = sample["image"]
            sample = np.stack(sample, axis=0)
        if self.transform_img is not None:
            sample = self.transform_img(sample)
        label_id = self.classes.index(label)
        print(name)
        return sample, label_id, name


    def __len__(self):
        return len(self.instances)


    def make_dataset(self,valid_ext,label_path,label_col):
        instances,labels,sample_name=[],[],[]
        df_label = pd.read_csv(label_path)
        file_names = glob.glob(os.path.join(self.root, '*'))
        for file_name in file_names:
            if file_name.endswith(valid_ext):
                m = re.match(r'([A-Z]+)-(\d+)-([A-Z]+)',  os.path.basename(file_name))
                if m:
                    label = df_label[df_label['sample_name'] == f"{m.group(1)}{m.group(2)}"][label_col].tolist()
                    if len(label) > 0:
                        instances.append(file_name)
                        labels.append(label[0])
                        sample_name.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
                else:
                    raise ValueError(f"Label not found for: {file_name}")
        assert(len(instances)==len(labels))
        print(len(instances), ' image detected. \n Dataset loading: done')
        return instances,labels,sample_name
