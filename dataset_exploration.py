import numpy as np

from dataset.dataset_antibiores import Antibio_Dataset

train_data = Antibio_Dataset(root='data/img_ms1/train_data',label_path='data/antibiores_labels.csv',label_col='GEN (mic) cat')
x_cond = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])

train_mean_cond = np.mean(x_cond, axis=(0, 1, 2)) #0.25
train_std_cond  = np.std(x_cond, axis=(0, 1, 2)) #0.21