import numpy as np
import torch
import random
import os
import decord
decord.bridge.set_bridge('torch')

def load_frames(vid, clip_len, root): 
    vname = os.path.join(root, f'{vid}')
    with open(vname, 'rb') as f:
        vr = decord.VideoReader(f, width=342, height=256, num_threads=1)
    frame_rate = int(np.floor(float(len(vr)) / float(clip_len)))
    start_frame = 1
    # start_frame = random.randint(1, num_frames - clip_len * frame_rate + 1)
    idx_list = [start_frame + i * frame_rate for i in range(clip_len)]
    frames = vr.get_batch(idx_list)
    frames = frames.type(torch.float32) / 255.

    return frames.permute((3, 0, 1, 2))

def make_dataset(cfg, split_file, root):
    data = []
    with open(split_file, 'r') as f:
        for line in f:
            data.append((line.split()[0], np.array(int(line.split()[1]) - 1)))
    return data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split_file, root, transforms=None):
        self.cfg = cfg
        self.split_file = split_file
        self.transforms = transforms
        self.root = root
        self.data = make_dataset(cfg, split_file, root)

    def __getitem__(self, index):
        vid, label = self.data[index]
        frames = load_frames(vid, self.cfg.CONFIG.DATA.CLIP_LEN, self.root)
        frames = self.transforms(frames)
        return frames, torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
