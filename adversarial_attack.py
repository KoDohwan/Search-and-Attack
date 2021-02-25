import os
import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
from tensorboardX import SummaryWriter

from gluoncv.torch.model_zoo import get_model
from gluoncv.torch.data import build_dataloader_val
from gluoncv.torch.utils.model_utils import deploy_model, load_model, save_model
from gluoncv.torch.utils.task_utils import adversarial_classification, validation_classification
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.engine.launch import spawn_workers
from gluoncv.torch.utils.utils import build_log_dir
from gluoncv.torch.utils.lr_policy import GradualWarmupScheduler

from torchvision import datasets, transforms
import videotransforms
from ucf_dataset import UCF as Dataset

import warnings

def main_worker(cfg):
    warnings.filterwarnings("ignore")
    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tb_logdir = build_log_dir(cfg)
        writer = SummaryWriter(log_dir=tb_logdir)
    else:
        writer = None
    cfg.freeze()

    # create model
    model = get_model(cfg)
    model = deploy_model(model, cfg)

    # create dataset and dataloader
    val_loader, _ = build_dataloader_val(cfg)

    if cfg.CONFIG.MODEL.LOAD:
        model, _ = load_model(model, cfg, load_fc=True)

    criterion = nn.CrossEntropyLoss().cuda()
    if cfg.CONFIG.MODEL.NAME == 'lrcn':
        criterion = nn.NLLLoss().cuda()

    # adversarial_classification(model, val_loader, -1, criterion, cfg, writer)
    validation_classification(model, val_loader, -1, criterion, cfg, writer)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video action recognition models.')
    parser.add_argument('--config-file', type=str, help='path to config file.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    spawn_workers(main_worker, cfg)