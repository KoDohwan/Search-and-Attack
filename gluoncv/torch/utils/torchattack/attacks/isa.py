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

class ISA(Attack):
    def __init__(self, cfg, model):
        super(ISA, self).__init__("ISA", cfg, model)

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        loss = nn.CrossEntropyLoss()
        if self.cfg.CONFIG.MODEL.NAME == 'lrcn':
            loss = nn.NLLLoss().cuda()

        adv_images = images.clone().detach()

        idx_set = [set() for _ in range(images.shape[0])]
        idx_count = [[0 for _ in range(32)] for _ in range(images.shape[0])]

        if self.cfg.CONFIG.MODEL.NAME == 'lrcn':
            self.model.module.Lstm.reset_hidden_state()
            self.model.module.Lstm.train()

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
            idx_list = [[i for i in range(32)] for _ in range(images.shape[0])]
        elif self.cfg.CONFIG.ADV.TYPE == 'Random':
            idx_list = [random.sample(range(32), self.cfg.CONFIG.ADV.FRAME) for _ in range(images.shape[0])]
        elif self.cfg.CONFIG.ADV.TYPE == 'Even':
            idx_list = [[int(32 / self.cfg.CONFIG.ADV.FRAME * i) for i in range(self.cfg.CONFIG.ADV.FRAME)] for _ in range(images.shape[0])]
        elif self.cfg.CONFIG.ADV.TYPE == 'L1':
            idx_list = np.argsort(grad_list, 1)[:, -self.cfg.CONFIG.ADV.FRAME:].tolist()
        elif self.cfg.CONFIG.ADV.TYPE == 'Evenly L1':
            idx_list = []
            rate = int(32 / self.cfg.CONFIG.ADV.FRAME)
            for i in range(images.shape[0]):
                idx = []
                for j in range(self.cfg.CONFIG.ADV.FRAME):
                    idx.append(np.argmax(grad_list[i, j*rate:(j+1)*rate]) + j * rate)
                idx_list.append(idx)
        elif self.cfg.CONFIG.ADV.TYPE == 'Greedy Loss':
            idx_list = [[] for _ in range(images.shape[0])]
            idx_left = [list(range(32)) for _ in range(images.shape[0])]
            atk = FGSM2(self.cfg, self.model, eps=2/255)
            criterion = nn.CrossEntropyLoss(reduction='none')
            atk_images = images.clone().detach()
            temp_list = [[] for _ in range(images.shape[0])]
            for i in range(self.cfg.CONFIG.ADV.FRAME):
                max_idx = [-1 for _ in range(images.shape[0])]
                max_list = [-math.inf for _ in range(images.shape[0])]
                for j in range(32-i):   # 31-i: num of indices left
                    for k in range(images.shape[0]):
                        temp_list[k] = idx_list[k] + [idx_left[k][j]]
                    temp_images, _, _ = atk(atk_images, labels, temp_list)
                    loss = criterion(self.model(temp_images), labels)
                    for k in range(images.shape[0]):
                        if (max_list[k] < loss[k].item()):
                            max_idx[k] = idx_left[k][j]
                            max_list[k] = loss[k].item()
                for j in range(images.shape[0]):
                    idx_list[j].append(max_idx[j])
                    idx_left[j].remove(max_idx[j])
                atk_images, _, _ = atk(atk_images, labels, idx_list)
                print(idx_list)
        elif self.cfg.CONFIG.ADV.TYPE == 'Greedy Logit':
            idx_left = [list(range(32)) for _ in range(images.shape[0])]
            atk = FGSM2(self.cfg, self.model, eps=2/255)
            criterion = nn.CrossEntropyLoss(reduction='none')
            cos = nn.CosineSimilarity(dim=0)
            batch_size = images.shape[0]
            #atk_images = images.clone().detach()
            #temp_list = [[] for _ in range(batch_size)]
            max_idx = [-1 for _ in range(batch_size)]
            max_list = [-math.inf for _ in range(batch_size)]
            logits_list = torch.Tensor(batch_size, 32, 101)
            logits_diff = torch.Tensor(batch_size, 31)
            for i in range(32):     
                temp_list = [[i] for _ in range(batch_size)]
                atk_images, _, _ = atk(images, labels, temp_list)
                logits = self.model(atk_images)
                logits_list[:,i,:] = logits - self.model(images)
                loss = criterion(logits, labels)
                for b in range(batch_size):
                    if max_list[b] < loss[b].item():
                        max_idx[b] = i
                        max_list[b] = loss[b].item()
            for b in range(batch_size):
                idx_left[b].remove(max_idx[b])
            for i in range(31):
                for b in range(batch_size):
                    logits_diff[b,i] = cos(logits_list[b,max_idx[b]],logits_list[b, idx_left[b][i]])
            topk_idx = torch.topk(logits_diff, self.cfg.CONFIG.ADV.FRAME-1, dim=1).indices.tolist()
            idx_list = [[] for _ in range(batch_size)]
            for b in range(batch_size):
                idx_list[b] = [max_idx[b]] + topk_idx[b]
            print(idx_list)




        atk_grad = sum([sum(grad_list[i, idx]) for i, idx in enumerate(idx_list)])
        print(idx_list)

        if self.cfg.CONFIG.ADV.METHOD == 'FGSM':
            atk = FGSM2(self.cfg, self.model, eps=2/255)
        elif self.cfg.CONFIG.ADV.METHOD == 'PGD':
            atk = PGD2(self.cfg, self.model, eps=2/255, alpha=0.5/255, steps=4, random_start=True)

        adv_images, l1_grad, num_frames = atk(images, labels, idx_list)
        ratio = float(atk_grad / grad_sum * 100) if grad_sum != 0. else 0.
        return adv_images, l1_grad, num_frames, ratio, grad_var