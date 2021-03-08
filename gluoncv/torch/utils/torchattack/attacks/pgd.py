import torch
import torch.nn as nn

import cv2

import numpy as np
import random
import copy

from ..attack import Attack

class PGD(Attack):
    def __init__(self, cfg, model, eps=0.3, alpha=2/255, steps=40, random_start=False):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, images, labels):
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
        # idx_list = [random.sample(range(64), 8) for _ in range(images.shape[0])]    # random

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs, att = self.model(adv_images)
            cost = loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            grad_sign = grad.sign()
            grad = torch.abs(grad)
            grad_list = torch.sum(grad, (1, 3, 4)).cpu().detach().numpy()

            # idx_list = [[j for j in range(64)] for _ in range(images.shape[0])]      # all
            idx_list = np.argsort(grad_list, 1)[:, -8:].tolist()                     # gradient

            idx_count = [[0 for _ in range(64)] for _ in range(images.shape[0])]

            for j in range(images.shape[0]):
                for num, k in enumerate(idx_list[j]):
                    idx_set[j].add(k)
                    idx_count[j][k] += 1
                idx_list[j] = list(idx_set[j])

            if idx_list == None:
                adv_images = adv_images.detach() + self.alpha * grad_sign
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=-1, max=1).detach()
            else:
                diff = torch.zeros(images.shape, device=self.device)
                adv_images = adv_images.detach()
                for j in range(images.shape[0]):
                    diff[j, :, idx_list[j], :, :] = self.alpha * grad_sign[j, :, idx_list[j], :, :]
                adv_images += diff
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

        for i in range(images.shape[0]):
            idx_list[i] = np.argsort(idx_count[i])[-8:].tolist()
        idx_set = [set() for _ in range(images.shape[0])]
        adv_images = images.clone().detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            cost = loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            grad_sign = grad.sign()
            grad = torch.abs(grad)
            grad_list = torch.sum(grad, (1, 3, 4)).cpu().detach().numpy()

            for j in range(images.shape[0]):
                for k in idx_list[j]:
                    idx_set[j].add(k)

            if idx_list == None:
                adv_images = adv_images.detach() + self.alpha * grad_sign
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=-1, max=1).detach()
            else:
                diff = torch.zeros(images.shape, device=self.device)
                adv_images = adv_images.detach()
                for j in range(images.shape[0]):
                    diff[j, :, idx_list[j], :, :] = self.alpha * grad_sign[j, :, idx_list[j], :, :]
                adv_images += diff
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

        num_frames = 0
        aap = 0.
        for i in range(images.shape[0]):
            num_frames += len(idx_set[i])
            aap += torch.norm(adv_images[i] - images[i], p=1).cpu().item() / len(idx_set[i])

        random_idx = random.sample(idx_set[0], 1)
        img = (images[0, :, random_idx[0], :, :].transpose(0, 1).transpose(1, 2).cpu().detach().numpy() + 1) / 2 * 255
        adv_img = (adv_images[0, :, random_idx[0], :, :].transpose(0, 1).transpose(1, 2).cpu().detach().numpy() + 1) / 2 * 255
        cv2.imwrite('./images/test1.png', img[:, :, [2, 1, 0]])
        cv2.imwrite('./images/test2.png', adv_img[:, :, [2, 1, 0]])
        cv2.imwrite('./images/test3.png', adv_img[:, :, [2, 1, 0]] - img[:, :, [2, 1, 0]])

        return adv_images, torch.norm(adv_images - images, p=1).cpu().item(), num_frames, aap

class PGD2(Attack):
    def __init__(self, cfg, model, eps=0.3, alpha=2/255, steps=40, random_start=False):
        super(PGD2, self).__init__("PGD2", cfg, model)
        self.eps = (eps / self.std).view((3, 1, 1, 1)).to(self.device)
        self.alpha = (alpha / self.std).view((3, 1, 1, 1)).to(self.device)
        self.steps = steps
        self.random_start = random_start

    def forward(self, images, labels, idx_list):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            for i in range(images.shape[0]):
                # diff = torch.empty_like(adv_images).uniform_(-self.eps, self.eps)[i, :, idx_list[i], :, :]
                diff = (2 * torch.rand(images.shape, device=self.device) * self.eps - self.eps)[i, :, idx_list[i], :, :]
                adv_images[i, :, idx_list[i], :, :] += diff
            adv_images = torch.min(torch.max(adv_images, self.min), self.max).detach()

        idx_set = [set() for _ in range(images.shape[0])]

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            cost = loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            grad_sign = grad.sign()
            grad = torch.abs(grad)
            grad_list = torch.sum(grad, (1, 3, 4)).cpu().detach().numpy()

            for j in range(images.shape[0]):
                for k in idx_list[j]:
                    idx_set[j].add(k)


            diff = torch.zeros(images.shape, device=self.device)
            adv_images = adv_images.detach()
            for j in range(images.shape[0]):
                diff[j, :, idx_list[j], :, :] = self.alpha * grad_sign[j, :, idx_list[j], :, :]
            adv_images += diff
            delta = torch.min(torch.max(adv_images - images, -self.eps), self.eps)
            adv_images = torch.min(torch.max(images + delta, self.min), self.max).detach()

        num_frames = 0
        for i in range(images.shape[0]):
            num_frames += len(idx_set[i])

        random_idx = random.sample(idx_set[0], 1)
        img = (images[0, :, random_idx[0], :, :].transpose(0, 1).transpose(1, 2).cpu().detach().numpy() + 1) / 2 * 255
        adv_img = (adv_images[0, :, random_idx[0], :, :].transpose(0, 1).transpose(1, 2).cpu().detach().numpy() + 1) / 2 * 255
        cv2.imwrite('./images/test1.png', img[:, :, [2, 1, 0]])
        cv2.imwrite('./images/test2.png', adv_img[:, :, [2, 1, 0]])
        cv2.imwrite('./images/test3.png', adv_img[:, :, [2, 1, 0]] - img[:, :, [2, 1, 0]])

        return adv_images, torch.norm(adv_images - images, p=1).cpu().item(), num_frames