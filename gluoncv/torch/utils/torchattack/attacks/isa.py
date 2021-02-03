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
    def __init__(self, model, eps=0.3, alpha=2/255, steps=40, random_start=False, iterative=True):
        super(ISA, self).__init__("ISA", model)
        self.eps = (eps / self.std).view((3, 1, 1, 1)).to(self.device)
        self.alpha = (alpha / self.std).view((3, 1, 1, 1)).to(self.device)
        self.steps = steps
        self.random_start = random_start
        self.iterative = iterative

    def forward(self, images, labels):
        if self.iterative:
            images = images.clone().detach().to(self.device)
            labels = labels.clone().detach().to(self.device)
            labels = self._transform_label(images, labels)

            loss = nn.CrossEntropyLoss()

            adv_images = images.clone().detach()

            if self.random_start:
                # Starting at a uniformly random point
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
                adv_images = torch.clamp(adv_images, min=-1, max=1).detach()

            idx_set = [set() for _ in range(images.shape[0])]
            idx_count = [[0 for _ in range(64)] for _ in range(images.shape[0])]

            for i in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.model(adv_images)
                cost = loss(outputs, labels)

                grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
                grad_sign = grad.sign()
                grad = torch.abs(grad)
                grad_list = torch.sum(grad, (1, 3, 4)).cpu().detach().numpy()

                idx_list = np.argsort(grad_list, 1)[:, -4:].tolist()                     # gradient

                for j in range(images.shape[0]):
                    for num, k in enumerate(idx_list[j]):
                        idx_set[j].add(k)
                        idx_count[j][k] += 1
                    # idx_list[j] = list(idx_set[j])

                diff = torch.zeros(images.shape, device=self.device)
                adv_images = adv_images.detach()
                for j in range(images.shape[0]):
                    diff[j, :, idx_list[j], :, :] = self.alpha * grad_sign[j, :, idx_list[j], :, :]
                adv_images += diff
                delta = torch.min(torch.max(adv_images - images, -self.eps), self.eps)
                adv_images = torch.min(torch.max(images + delta, self.min), self.max).detach()

            for i in range(images.shape[0]):
                idx_list[i] = np.argsort(idx_count[i])[-4:].tolist()
        else:
            # idx_list = [[j for j in range(32)] for _ in range(images.shape[0])]         # all
            # idx_list = [random.sample(range(32), 4) for _ in range(images.shape[0])]    # random
            idx_list = [[8 * i for i in range(4)] for _ in range(images.shape[0])]      # even
            # idx_list = [[28 + i for i in range(8)] for _ in range(images.shape[0])]     # head

        print(idx_list)

        # atk = FGSM2(self.model, eps=4/255)
        atk = PGD2(self.model, eps=2/255, alpha=1/255, steps=8, random_start=True)

        adv_images, l1_grad, num_frames = atk(images, labels, idx_list)

        return adv_images, l1_grad, num_frames