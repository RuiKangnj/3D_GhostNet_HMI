###convert envi format to npy
import spectral.io.envi as envi
from tqdm import tqdm
from data.preprocessing import spectra_slice,img_fill,aug_hyp
import os
import numpy as np


def envi_open(hyp_path):
    hdr_path=hyp_path.replace('.hyp','.hdr')
    img = envi.open(hdr_path,hyp_path)
    data=img.load()
    return data

def convert_hyp_to_npy(folder_path,target_path):
    for root, dirs,_ in os.walk(folder_path):
        for dir in tqdm(sorted(dirs), desc="Processing folders"):
            dir_path = os.path.join(root, dir)
            for file in tqdm(sorted(os.listdir(dir_path)), desc="Processing files in '{}'".format(dir)):
                if file.endswith('.hyp'):
                    hyp_path = os.path.join(dir_path, file)
                    raw_data=envi_open(hyp_path)
                    data_slice = spectra_slice(raw_data)        #299 to 256
                    if raw_data.shape[0]<64 and raw_data.shape[1]<64:
                        data_fill = img_fill(64, 64, data_slice)   ####standard 64,64,256
                  #  data_rotated, data_flipped_lr, data_flipped_ud = aug_hyp(data_fill)
                    # 生成新的文件名
                        origin_name=dir+'_'+file[:-4]+'_origin.npy'
                    #rotated_name=dir+'_'+file[:-4]+'_ro.npy'
                    #flipped_lr_name, flipped_ud_name=dir+'_'+file[:-4]+'_fllr.npy',dir+'_'+file[:-4]+'_flup.npy'

                        new_npy_ori_path = os.path.join(target_path, origin_name)
                        np.save(new_npy_ori_path, data_fill)
                    else:
                        continue


                   # new_npy_rot_path = os.path.join(target_path, rotated_name)
                  #  np.save(new_npy_rot_path, data_rotated)
                    # 保存为 .npy 文件
                   # new_npy_fllr_path = os.path.join(target_path, flipped_lr_name)
                  #  np.save(new_npy_fllr_path, data_flipped_lr)

                   # new_npy_flud_path = os.path.join(target_path, flipped_ud_name)
                   # np.save(new_npy_flud_path, data_flipped_ud)


def collect_file_paths(folder_path):
    file_paths = []
    labels = []

    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)
        if os.path.isdir(label_path):
            label = int(label_folder)

            for file in os.listdir(label_path):
                if file.endswith('.hyp'):
                    hyp_path = os.path.join(label_path, file)
                    file_paths.append(hyp_path)
                    labels.append(label)

    return file_paths, labels