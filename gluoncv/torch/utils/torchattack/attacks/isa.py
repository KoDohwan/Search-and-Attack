import torch
import torch.nn as nn

import cv2

import numpy as np
import random
import copy

from ..attack import Attack
from .pgd import PGD2
from .fgsm import FGSM2

class ISA(Attack):
    def __init__(self, cfg, model, eps=0.3, alpha=2/255, steps=1, random_start=False):
        super(ISA, self).__init__("ISA", cfg, model)
        self.eps = (eps / self.std).view((3, 1, 1, 1)).to(self.device)
        self.alpha = (alpha / self.std).view((3, 1, 1, 1)).to(self.device)
        self.steps = steps
        self.random_start = random_start

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        loss = nn.CrossEntropyLoss()
        if self.cfg.CONFIG.MODEL.NAME == 'lrcn':
            loss = nn.NLLLoss().cuda()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=-1, max=1).detach()

        idx_set = [set() for _ in range(images.shape[0])]
        idx_count = [[0 for _ in range(32)] for _ in range(images.shape[0])]

        for i in range(self.steps):
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
                
        if self.cfg.CONFIG.ADV.TYPE == 'All':
            idx_list = [[j for j in range(32)] for _ in range(images.shape[0])]
        elif self.cfg.CONFIG.ADV.TYPE == 'Random':
            idx_list = [random.sample(range(32), self.cfg.CONFIG.ADV.FRAME) for _ in range(images.shape[0])]
        elif self.cfg.CONFIG.ADV.TYPE == 'Even':
            idx_list = [[int(32 / self.cfg.CONFIG.ADV.FRAME * i) for i in range(self.cfg.CONFIG.ADV.FRAME)] for _ in range(images.shape[0])]
        elif self.cfg.CONFIG.ADV.TYPE == 'L1':
            idx_list = np.argsort(grad_list, 1)[:, -self.cfg.CONFIG.ADV.FRAME:].tolist()
        elif self.cfg.CONFIG.ADV.TYPE == 'Evenly L1':
            idx_list = []
            rate = int(32 / self.cfg.CONFIG.ADV.FRAME)
            for j in range(images.shape[0]):
                idx = []
                for k in range(self.cfg.CONFIG.ADV.FRAME):                
                    idx.append(np.argmax(grad_list[j, k*rate:(k+1)*rate]) + k * rate)
                idx_list.append(idx)

            # for j in range(images.shape[0]):
            #     for num, k in enumerate(idx_list[j]):
            #         idx_set[j].add(k)
            #         idx_count[j][k] += 1
            #     # idx_list[j] = list(idx_set[j])

            # diff = torch.zeros(images.shape, device=self.device)
            # adv_images = adv_images.detach()
            # for j in range(images.shape[0]):
            #     diff[j, :, idx_list[j], :, :] = self.alpha * grad_sign[j, :, idx_list[j], :, :]
            # adv_images += diff
            # delta = torch.min(torch.max(adv_images - images, -self.eps), self.eps)
            # adv_images = torch.min(torch.max(images + delta, self.min), self.max).detach()

        atk_grad = sum([sum(grad_list[i, idx]) for i, idx in enumerate(idx_list)])
        print(idx_list)

        if self.cfg.CONFIG.ADV.METHOD == 'FGSM':
            atk = FGSM2(self.cfg, self.model, eps=2/255)
        elif self.cfg.CONFIG.ADV.METHOD == 'PGD':
            atk = PGD2(self.cfg, self.model, eps=2/255, alpha=0.5/255, steps=8, random_start=True)

        adv_images, l1_grad, num_frames = atk(images, labels, idx_list)
        ratio = float(atk_grad / grad_sum * 100) if grad_sum != 0. else 0.
        return adv_images, l1_grad, num_frames, ratio, grad_var