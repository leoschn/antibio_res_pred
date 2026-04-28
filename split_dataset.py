import glob
import os
import numpy as np
import shutil

DIR_IMG = '/lustre/fsn1/projects/rech/bun/ucg81ws/image'
DIR_PAIRS = '/lustre/fsn1/projects/rech/bun/ucg81ws/pairs'

OUT_DIR_PAIRS = '/lustre/fsn1/projects/rech/bun/ucg81ws/synt_class_dataset/eval_1/generation_dataset'
OUT_DIR_IMG = '/lustre/fsn1/projects/rech/bun/ucg81ws/synt_class_dataset/eval_1/classification_dataset'

MAJOR_SPECIES = ['CITFRE','ESCCOL','KLEPNE','ENTHOR','PRTMIR']

img_f_name_list = glob.glob(os.path.join(DIR_IMG,'*.pkl'))

img_base_name_list = [os.path.basename(f_name).split('.')[0] for f_name in img_f_name_list]
img_base_name_array = np.array(img_base_name_list)


def is_major_species(name,species):
    for spe in species:
        if spe in name:
            return True

    return False

img_is_major = np.array([is_major_species(f_name,MAJOR_SPECIES) for f_name in img_base_name_list])

path_major_img_array = img_base_name_array[img_is_major]

ind_major_array = np.linspace(0,len(path_major_img_array)-1,len(path_major_img_array),dtype=int)

np.random.shuffle(ind_major_array)

idx_A = int(len(path_major_img_array)*0.4)
idx_B = idx_A + int(len(path_major_img_array)*0.1)
idx_C = idx_B + int(len(path_major_img_array)*0.2)
idx_D = idx_C + int(len(path_major_img_array)*0.05)
idx_E = len(path_major_img_array)


path_minor_img_array = img_base_name_array[~img_is_major]

ind_minor_array = np.linspace(0,len(path_minor_img_array)-1,len(path_minor_img_array),dtype=int)

np.random.shuffle(ind_minor_array)

idx_C_minor = int(len(path_minor_img_array)*0.8)
idx_D_minor = len(path_minor_img_array)

path_array_A = path_major_img_array[ind_major_array[:idx_A]]
path_array_B = path_major_img_array[ind_major_array[idx_A:idx_B]]
path_array_C = np.concat([path_major_img_array[ind_major_array[idx_B:idx_C]],path_minor_img_array[ind_minor_array[:idx_C_minor]]], axis=0)
path_array_D = np.concat([path_major_img_array[ind_major_array[idx_C:idx_D]],path_minor_img_array[ind_minor_array[idx_C_minor:idx_D_minor]]], axis=0)
path_array_E = path_major_img_array[ind_major_array[idx_D:idx_E]]

print('Split A : ', len(path_array_A), 'images')
print('Split B : ', len(path_array_B), 'images')
print('Split C : ', len(path_array_C), 'images')
print('Split D : ', len(path_array_D), 'images')
print('Split E : ', len(path_array_E), 'images')

#COPY IMG
for base_name in path_array_A:
    shutil.copy(os.path.join(DIR_IMG,base_name+'.pkl'),os.path.join(OUT_DIR_IMG,'split_A',base_name+'.pkl'))
for base_name in path_array_B:
    shutil.copy(os.path.join(DIR_IMG,base_name+'.pkl'),os.path.join(OUT_DIR_IMG,'split_B',base_name+'.pkl'))
for base_name in path_array_E:
    shutil.copy(os.path.join(DIR_IMG,base_name+'.pkl'),os.path.join(OUT_DIR_IMG,'split_E',base_name+'.pkl'))

#COPY PAIRS

for base_name in path_array_C:
    for i in range(100):
        try:
            shutil.copy(os.path.join(DIR_PAIRS, base_name+ f'_ms2_{i}.pkl'),
                        os.path.join(OUT_DIR_PAIRS, 'split_C', base_name+ f'_ms2_{i}.pkl'))
        except:
            pass
for base_name in path_array_D:
    for i in range(100):
        try:
            shutil.copy(os.path.join(DIR_PAIRS, base_name+ f'_ms2_{i}.pkl'),
                        os.path.join(OUT_DIR_PAIRS, 'split_D', base_name+ f'_ms2_{i}.pkl'))
        except:
            pass
for base_name in path_array_E:
    for i in range(100):
        try:
            shutil.copy(os.path.join(DIR_PAIRS, base_name+ f'_ms2_{i}.pkl'),
                        os.path.join(OUT_DIR_PAIRS, 'split_E', base_name+ f'_ms2_{i}.pkl'))
        except:
            pass




