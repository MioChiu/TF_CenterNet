import numpy as np
import os
import cfg
import cv2
import math
from utils.utils import image_preporcess
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.data_aug import random_horizontal_flip, random_crop, random_translate, random_color_distort

def process_data(line, use_aug):
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.split()
    image_path = s[0]
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " %image_path)
    image = np.array(cv2.imread(image_path))
    labels = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in s[1:]])
    
    if use_aug:
        image, labels = random_horizontal_flip(image, labels)
        image, labels = random_crop(image, labels)
        image, labels = random_translate(image, labels)

    image, labels = image_preporcess(np.copy(image), [cfg.input_image_h,cfg.input_image_w], np.copy(labels))
    
    output_h = cfg.input_image_h//cfg.down_ratio
    output_w = cfg.input_image_w//cfg.down_ratio
    hm = np.zeros((output_h, output_w, cfg.num_classes),dtype=np.float32)
    wh = np.zeros((cfg.max_objs, 2),dtype=np.float32)
    reg = np.zeros((cfg.max_objs, 2),dtype=np.float32)
    ind = np.zeros((cfg.max_objs),dtype=np.float32)
    reg_mask = np.zeros((cfg.max_objs),dtype=np.float32)

    for idx, label in enumerate(labels):
        bbox = label[:4]/cfg.down_ratio
        class_id = label[4]
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        radius = gaussian_radius((math.ceil(h),math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_umich_gaussian(hm[:,:,class_id], ct_int, radius)
        wh[idx] = 1. * w, 1. * h
        ind[idx] = ct_int[1] * output_w + ct_int[0]
        reg[idx] = ct - ct_int
        reg_mask[idx] = 1
    
    return image, hm, wh, reg, reg_mask, ind

def get_data(batch_lines, use_aug):

    batch_image = np.zeros((cfg.batch_size, cfg.input_image_h, cfg.input_image_w, 3),dtype=np.float32)

    batch_hm = np.zeros((cfg.batch_size, cfg.input_image_h//cfg.down_ratio, cfg.input_image_w//cfg.down_ratio, cfg.num_classes),dtype=np.float32)
    batch_wh = np.zeros((cfg.batch_size, cfg.max_objs, 2),dtype=np.float32)
    batch_reg = np.zeros((cfg.batch_size, cfg.max_objs, 2),dtype=np.float32)
    batch_reg_mask = np.zeros((cfg.batch_size, cfg.max_objs),dtype=np.float32)
    batch_ind = np.zeros((cfg.batch_size, cfg.max_objs),dtype=np.float32)
    # batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes= [], [], [], [], [], [], []
    for num, line in enumerate(batch_lines):
        image, hm, wh, reg, reg_mask, ind = process_data(line, use_aug)
        batch_image[num, :, :, :] = image
        batch_hm[num, :, :, :] = hm
        batch_wh[num, :, :] = wh
        batch_reg[num, :, :] = reg
        batch_reg_mask[num, :] = reg_mask
        batch_ind[num, :] = ind

    return batch_image, batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind
