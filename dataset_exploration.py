import numpy as np

from dataset.dataset_antibiores import Antibio_Dataset

train_data = Antibio_Dataset(root='/lustre/fsn1/projects/rech/bun/ucg81ws/dataset/img_train',
                             label_path='/lustre/fswork/projects/rech/bun/ucg81ws/these/antibio_res_pred/data/antibiores_labels.csv',
                             label_col='GEN (mic) cat',
                             model_type='ms2')
x_cond = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])

train_mean_cond = np.mean(x_cond, axis=(0, 1, 2))
train_std_cond  = np.std(x_cond, axis=(0, 1, 2))
print('mean : ',train_mean_cond, 'std : ', train_std_cond)