import numpy as np

def spectra_slice(data):     ####ensure channel is 256
    data_reduced=data[:,:,23:-20]
    assert data_reduced.shape[2] == 256, "The number of channels is not 256."
    return(data_reduced)

def img_fill(target_w,target_h,raw_img):
    padding_height = (target_w - raw_img.shape[0]) // 2
    padding_width = (target_h - raw_img.shape[1]) // 2
    padded_img = np.pad(raw_img,((padding_height, target_w - raw_img.shape[0] - padding_height), (padding_width, target_h - raw_img.shape[1] - padding_width),
                            (0,0)),'constant')
    return(padded_img)

def aug_hyp(data):
    data_rotated=np.rot90(data)
    data_flipped_lr = np.fliplr(data)
    data_flipped_ud = np.flipud(data)
    return(data_rotated,data_flipped_lr,data_flipped_ud)