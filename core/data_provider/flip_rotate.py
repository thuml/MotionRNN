# -*- coding: utf-8 -*-
# @File  : flip_rotate.py
# @Author: GZF
# @Date  : 2017/10/29
# @Desc  :

import os
from PIL import Image
import numpy as np
import cv2
import random


def read_batch(img_path="../../input/1"):
    batch_size = 1
    seq_len = 10
    width = 100
    channel = 1
    input_batch = np.zeros((batch_size, seq_len, width, width, channel)).astype(np.float32)
    filename_list = os.listdir(img_path)
    for filename, index in zip(filename_list, range(seq_len)):
        file = Image.open(os.path.join(img_path, filename))
        file = np.array(file, dtype=np.float32)
        file = cv2.resize(file, (width, width))
        input_batch[0, index, :, :, 0] = file
        # plt.imshow(file, cmap="gray")
        # plt.show()
        print(file.shape)

    return input_batch


def augment_data(batch):
    rand = random.random()

    if rand < 0.5:
        batch = np.flip(batch, 1)
    elif rand < 0.6:
        w = batch.shape[2]
        angle = 90
        rotate_img(batch, w, angle)
    elif rand < 0.7:
        w = batch.shape[2]
        angle = 270
        rotate_img(batch, w, angle)
    elif rand < 0.8:
        flip_img(batch, 0)
    elif rand < 0.9:
        flip_img(batch, 1)
    elif rand < 1.0:
        flip_img(batch, -1)
    return batch


def flip_img(batch, flipCode):
    for batch_ind in range(batch.shape[0]):
        for seq_ind in range(batch.shape[1]):
            img_arr = batch[batch_ind, seq_ind, :, :, 0]
            batch[batch_ind, seq_ind, :, :, 0] = cv2.flip(img_arr, flipCode)


def rotate_img(batch, w, angle):
    center = (w / 2, w / 2)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)

    for batch_ind in range(batch.shape[0]):
        for seq_ind in range(batch.shape[1]):
            img_arr = batch[batch_ind, seq_ind, :, :, 0]
            batch[batch_ind, seq_ind, :, :, 0] = cv2.warpAffine(img_arr, M, (w, w))

# if __name__ == '__main__':

#     batch = read_batch()
#     batch = augment_data(batch)

#     for batch_ind in range(batch.shape[0]):
#         for seq_ind in range(batch.shape[1]):
#             img_arr = batch[batch_ind, seq_ind, :, :, 0]
#             plt.imshow(img_arr, cmap="gray")
#             plt.show()
