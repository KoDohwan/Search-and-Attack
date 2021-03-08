import torch
import torch.nn as nn

import cv2

import math
import numpy as np
import random
import copy
from itertools import combinations
from tqdm import tqdm

from ..attack import Attack
from .pgd import PGD2
from .fgsm import FGSM2

class LEAST_FRAME_ATTACK(Attack):
    def __init__(self, cfg, model):
        super(LEAST_FRAME_ATTACK, self).__init__("LEAST_FRAME_ATTACK", cfg, model)

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        batch_size = images.shape[0]
        clip_len = self.cfg.CONFIG.DATA.CLIP_LEN
        adv_num_frame = self.cfg.CONFIG.ADV.FRAME

        loss = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss(reduction='none')

        if self.cfg.CONFIG.ADV.METHOD == 'FGSM':
            atk = FGSM2(self.cfg, self.model, eps=2/255)
        elif self.cfg.CONFIG.ADV.METHOD == 'PGD':
            atk = PGD2(self.cfg, self.model, eps=2/255, alpha=0.5/255, steps=4, random_start=True)

        adv_images = images.clone().detach()

        idx_set = [set() for _ in range(batch_size)]
        idx_count = [[0 for _ in range(clip_len)] for _ in range(batch_size)]


        adv_images.requires_grad = True
        outputs = self.model(adv_images)
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        grad_sign = grad.sign()
        grad = torch.abs(grad)
        grad_list = torch.sum(grad, (1, 3, 4)).cpu().detach().numpy()
        grad_sum = torch.sum(grad).cpu().detach().numpy()
        grad_var = np.mean(np.var(grad_list, 1))

        # with open('idx.txt', 'a') as f:
        #     f.write(str(grad_list) + '\n')

        if self.cfg.CONFIG.ADV.TYPE == 'All':
            idx_list = [[i for i in range(clip_len)] for _ in range(batch_size)]
        elif self.cfg.CONFIG.ADV.TYPE == 'Random':
            idx_list = [random.sample(range(clip_len), adv_num_frame) for _ in range(batch_size)]
        elif self.cfg.CONFIG.ADV.TYPE == 'Even':
            idx_list = [[int(clip_len / adv_num_frame * i) for i in range(adv_num_frame)] for _ in range(batch_size)]
        elif self.cfg.CONFIG.ADV.TYPE == 'L1':
            idx_list = np.argsort(grad_list, 1)[:, -adv_num_frame:].tolist()
        elif self.cfg.CONFIG.ADV.TYPE == 'Evenly L1':
            idx_list = []
            rate = int(clip_len / adv_num_frame)
            for i in range(batch_size):
                idx = []
                for j in range(adv_num_frame):
                    idx.append(np.argmax(grad_list[i, j*rate:(j+1)*rate]) + j * rate)
                idx_list.append(idx)
        elif self.cfg.CONFIG.ADV.TYPE == 'Loss':
            idx_list = [[] for _ in range(batch_size)]
            loss_array = np.zeros((clip_len, batch_size))
            adv_images = images.clone().detach()
            for i in range(clip_len):
                temp_images, _, _ = atk(adv_images, labels, [[i] for _ in range(batch_size)])
                loss_array[i] = criterion(self.model(temp_images), labels).cpu().detach().numpy()
            loss_array = np.transpose(loss_array, (1, 0))
            idx_list = np.argsort(loss_array)[:, -adv_num_frame:].tolist()
        elif self.cfg.CONFIG.ADV.TYPE == 'Loss-I':
            idx_list = [[] for _ in range(batch_size)]
            idx_remain = [list(range(clip_len)) for _ in range(batch_size)]
            adv_images = images.clone().detach()
            temp_list = [[] for _ in range(batch_size)]
            for i in range(adv_num_frame):
                max_idx = [-1 for _ in range(batch_size)]
                max_list = [-math.inf for _ in range(batch_size)]
                for j in range(clip_len - i):
                    for k in range(batch_size):
                        temp_list[k] = idx_list[k] + [idx_remain[k][j]]
                    temp_images, _, _ = atk(adv_images, labels, temp_list)
                    loss = criterion(self.model(temp_images), labels)
                    for k in range(batch_size):
                        if (max_list[k] < loss[k].item()):
                            max_idx[k] = idx_remain[k][j]
                            max_list[k] = loss[k].item()
                for j in range(batch_size):
                    idx_list[j].append(max_idx[j])
                    idx_remain[j].remove(max_idx[j])
                atk_images, _, _ = atk(atk_images, labels, idx_list)
        elif self.cfg.CONFIG.ADV.TYPE == 'Logit':
            idx_list = [[] for _ in range(batch_size)]
            cos = nn.CosineSimilarity(dim=0)
            max_idx = [-1 for _ in range(batch_size)]
            max_list = [-math.inf for _ in range(batch_size)]
            logits_list = torch.Tensor(batch_size, clip_len, self.cfg.CONFIG.DATA.NUM_CLASSES)
            logits_sim = torch.Tensor(batch_size, clip_len)
            for i in range(clip_len):
                temp_list = [[i] for _ in range(batch_size)]
                adv_images, _, _ = atk(images, labels, temp_list)
                logits = self.model(adv_images)
                logits_list[:, i, :] = logits.detach().cpu() - self.model(images).detach().cpu()
                loss = criterion(logits, labels)
                for j in range(batch_size):
                    if max_list[j] < loss[j].item():
                        max_idx[j] = i
                        max_list[j] = loss[j].item()
            for i in range(batch_size):
                for j in range(clip_len):
                    logits_sim[i, j] = cos(logits_list[i, max_idx[i]], logits_list[i, j])
            topk_idx = torch.topk(logits_sim, adv_num_frame, dim=1).indices.tolist()
            for i in range(batch_size):
                idx_list[i] = topk_idx[i]
        elif self.cfg.CONFIG.ADV.TYPE == 'Gradient':
            idx_list = [[] for _ in range(batch_size)]
            max_list = np.argmax(grad_list, 1)
            print(max_list.shape)
        

        atk_grad = sum([sum(grad_list[i, idx]) for i, idx in enumerate(idx_list)])
        print(idx_list)

        adv_images, l1_grad, num_frames = atk(images, labels, idx_list)
        ratio = float(atk_grad / grad_sum * 100) if grad_sum != 0. else 0.
        return adv_images, l1_grad, num_frames, ratio, grad_var