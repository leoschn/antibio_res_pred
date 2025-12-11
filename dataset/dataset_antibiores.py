import glob
import os
import pickle as pkl
import re

import pandas as pd
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
import torch

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
        sample = torch.from_numpy(pkl.read(f))
    sample = sample.unsqueeze(0)
    return sample


class Antibio_Dataset(DatasetFolder):
    def __init__(self, root,label_path,label_col):
        self.root = root
        self.instances, self.labels = self.make_dataset('.pkl',label_path,label_col)
        self.loader = pkl_loader
        self.transform_img = transforms.Compose([transforms.ToTensor(),
                                             Log_normalisation(),
                                             transforms.Resize((256,256))])

    def __getitem__(self, index: int):
        label = self.labels[index]
        path = self.instances[index]
        sample = self.loader(path)
        if self.transform_img is not None:
            sample = self.transform_img(sample)
        return sample, label

    def __len__(self):
        return len(self.instances)


    def make_dataset(self,valid_ext,label_path,label_col):
        instances,labels=[],[]
        df_label = pd.read_csv(label_path)
        file_names = glob.glob(os.path.join(self.root, '*'))
        for file_name in file_names:
            if file_name.endswith(valid_ext):
                instances.append(file_name)
                m = re.match(r'([A-Z]+)-(\d+)-([A-Z]+)',  os.path.basename(file_name))
                if m:
                    label = df_label['sample_name' == f"{m.group(1)}{m.group(2)}"][self.label_col]
                    labels.append(label)
                else:
                    raise ValueError(f"Label not found for: {file_name}")
        return instances,labels
