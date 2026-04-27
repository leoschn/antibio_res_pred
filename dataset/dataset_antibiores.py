import glob
import os
import pickle as pkl
import re
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
import torch


def pkl_loader(path):
    with open(path, 'rb') as f:
        sample = pkl.load(f)
    return sample

class CropTransform:
    def __init__(self, top: int, left: int, height: int, width: int):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:  # (H, W)
            return image[self.top:self.top + self.height,self.left:self.left + self.width]
        elif image.ndim == 3:  # (C, H, W)
            return image[:,self.top:self.top + self.height,self.left:self.left + self.width]
        else:
            raise ValueError("Unsupported image shape")


class SpeciesDataset(DatasetFolder):

    def __init__(self, root, origin, im_size = (256,512)):
        self.root = root
        self.samples, self.targets , self.sample_name= self.build_dataset('.pkl')
        self.loader = pkl_loader
        self.classes = list(set(self.targets))
        self.classes.sort()
        self.origin = origin
        if origin == 'real' :
            self.transform_img = transforms.Compose([
                                                 CropTransform(top=90, left=0, height=422, width=1024),
                                                 transforms.ToTensor(),
                                                 transforms.Resize(im_size),
                                                 transforms.Normalize((3.04),(3.04)), #same as the transform used in diffusion
            ])
        elif origin == 'synth' :
            self.transform_img = None
        else :
            raise NotImplementedError


    def __getitem__(self, index: int):
        label = self.targets[index]
        path = self.samples[index]
        name = self.sample_name[index]
        sample = self.loader(path)
        sample = sample["image"]
        if self.origin == 'real':
            assert len(sample)==101
            sample = sample[1:101] #only keep ms2 data
        else :
            assert len(sample==100)

        if self.transform_img:
                sample = [self.transform_img(s) for s in sample]
        sample = torch.cat(sample, dim=0)
        label = self.classes.index(label)
        return sample, label, name


    def __len__(self):
        return len(self.samples)


    def build_dataset(self, valid_ext):
        instances,labels,sample_name=[],[],[]
        file_names = glob.glob(os.path.join(self.root, '*'))
        for file_name in file_names:
            if file_name.endswith(valid_ext):
                m = re.match(r'([A-Z]+)-(\d+)-([A-Z]+)',  os.path.basename(file_name))
                if m and m.group(1) :
                    instances.append(file_name)
                    labels.append(m.group(1))
                    sample_name.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        assert(len(instances)==len(labels))
        print(len(instances), ' image detected. \n Dataset loading: done')
        return instances,labels,sample_name

