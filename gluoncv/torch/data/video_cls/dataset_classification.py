"""Customized dataloader for general video classification tasks."""
import os
import warnings
import numpy as np
from decord import VideoReader, cpu

import torch
from torchvision import datasets, transforms

from ..transforms.videotransforms import video_transforms

from dataset import Dataset

__all__ = ['build_dataloader', 'build_dataloader_val', 'build_dataloader_test']

def build_dataloader(cfg):
    train_transforms = transforms.Compose([video_transforms.ResizeVideo((128, 171)),
                                        video_transforms.RandomHorizontalFlipVideo(),
                                        video_transforms.RandomResizedCropVideo(112, (0.75, 1.25)),
                                        video_transforms.NormalizeVideo(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])], )
    train_dataset = Dataset(cfg, cfg.CONFIG.DATA.TRAIN_ANNO_PATH, cfg.CONFIG.DATA.DATA_PATH, train_transforms)

    val_transforms = transforms.Compose([video_transforms.ResizeVideo((128, 171)),
                                        video_transforms.CenterCropVideo((112, 112)),
                                        video_transforms.NormalizeVideo(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])], )
    val_dataset = Dataset(cfg, cfg.CONFIG.DATA.VAL_ANNO_PATH, cfg.CONFIG.DATA.DATA_PATH, val_transforms)

    if cfg.DDP_CONFIG.DISTRIBUTED:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    mg_sampler = None
    if cfg.CONFIG.DATA.MULTIGRID:
        mg_sampler = MultiGridBatchSampler(train_sampler, batch_size=cfg.CONFIG.TRAIN.BATCH_SIZE, drop_last=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=9, pin_memory=True, batch_sampler=mg_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.CONFIG.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None),
            num_workers=9, sampler=train_sampler, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=(val_sampler is None),
        num_workers=9, sampler=val_sampler, pin_memory=True)
    return train_loader, val_loader, train_sampler, val_sampler, mg_sampler

def build_dataloader_val(cfg):
    val_transforms = transforms.Compose([video_transforms.ResizeVideo((128, 171)),
                                        video_transforms.CenterCropVideo((112, 112)),
                                        video_transforms.NormalizeVideo(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])], )
    val_dataset = Dataset(cfg, cfg.CONFIG.DATA.VAL_ANNO_PATH, cfg.CONFIG.DATA.DATA_PATH, val_transforms)

    if cfg.DDP_CONFIG.DISTRIBUTED:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=False,
        num_workers=9, sampler=val_sampler, pin_memory=True)
    return val_loader, val_sampler

def build_dataloader_test(cfg):
    """Build dataloader for testing"""
    test_dataset = VideoClsDataset(anno_path=cfg.CONFIG.DATA.VAL_ANNO_PATH,
                                   data_path=cfg.CONFIG.DATA.VAL_DATA_PATH,
                                   mode='test',
                                   clip_len=cfg.CONFIG.DATA.CLIP_LEN,
                                   frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                   test_num_segment=cfg.CONFIG.DATA.TEST_NUM_SEGMENT,
                                   test_num_crop=cfg.CONFIG.DATA.TEST_NUM_CROP,
                                   keep_aspect_ratio=cfg.CONFIG.DATA.KEEP_ASPECT_RATIO,
                                   crop_size=cfg.CONFIG.DATA.CROP_SIZE,
                                   short_side_size=cfg.CONFIG.DATA.SHORT_SIDE_SIZE,
                                   new_height=cfg.CONFIG.DATA.NEW_HEIGHT,
                                   new_width=cfg.CONFIG.DATA.NEW_WIDTH)

    if cfg.DDP_CONFIG.DISTRIBUTED:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=(test_sampler is None),
        num_workers=9, sampler=test_sampler, pin_memory=True)

    return test_loader
