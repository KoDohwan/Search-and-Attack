import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import json
import csv
import h5py
import random
import os
import os.path

import cv2
from torchvision import datasets, transforms
import videotransforms

from PIL import Image
from tqdm import tqdm

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

def load_rgb_frames(root, vid, start_frame, clip_len, frame_rate):
    frames = torch.zeros((clip_len, 3, 256, 342))
    for i in range(clip_len):
        idx = start_frame + i * frame_rate
        frame_id = 'frame' + str(idx).zfill(6) + '.jpg'
        img = cv2.imread(os.path.join(root, vid, frame_id))[:, :, [2, 1, 0]]
        img = cv2.resize(img, dsize=(342, 256), interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        frames[i] = img
    return frames.transpose(0, 1)

def make_dataset(cfg, split_file, root):
    dataset = []
    data = []
    with open(split_file, 'r') as f:
        for line in f:
            data.append((line.split()[0].split('/')[1].split('.')[0], int(line.split()[1])))

    for vid, label_index in data:
        if vid[2:18] == 'HandStandPushups':
            vid = vid.replace('HandStandPushups', 'HandstandPushups')
        num_frames = len(os.listdir(os.path.join(root, vid)))
        if num_frames <= cfg.CONFIG.DATA.CLIP_LEN + 1:
            continue
        label = np.array([label_index-1])
        dataset.append((vid, label, num_frames))
    return dataset

class UCF(data_utl.Dataset):
    def __init__(self, cfg, split_file, root, transforms=None):
        self.cfg = cfg
        self.data = make_dataset(cfg, split_file, root)
        self.transforms = transforms
        self.root = root

    def __getitem__(self, index):
        clip_len = self.cfg.CONFIG.DATA.CLIP_LEN
        vid, label, num_frames = self.data[index]

        frame_rate = int(np.floor(float(num_frames) / float(clip_len)))

        # start_frame = 1
        # start_frame = random.randint(1, num_frames - clip_len - 1)
        start_frame = random.randint(1, num_frames - clip_len * frame_rate + 1)

        imgs = load_rgb_frames(self.root, vid, start_frame, clip_len, frame_rate)
        frames = self.transforms(imgs)

        return frames, torch.from_numpy(label)

    def __len__(self):
        return len(self.data)