import torch
import torch.nn as nn
import numpy as np
import random
import cv2

from ..attack import Attack

class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT: 0.007)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps, weight):
        super(FGSM, self).__init__("FGSM", model)
        self.eps = eps
        self.weight = weight

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs, latent_rep= self.model(images)
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
        grad_sign = grad.sign()
        grad = nn.ReLU()(grad)
        # grad = torch.abs(grad)
        grad_list = torch.sum(grad, (1, 3, 4)).cpu().detach().numpy()

        # grad_cam = GradCam2(model=self.model.module, feature_module=self.model.module.res_layers, target_layer_names=["3"], use_cuda=True)
        # input = torch.zeros_like(images)
        # temp = images.cpu().detach().numpy()
        # temp = (temp + 1) / 2
        # temp = np.transpose(temp, (0, 2, 3, 4, 1))
        # # temp = temp[:, :, :, :, [2, 1, 0]]
        # # temp = temp.copy()[:, :, :, :, ::-1]
        # temp = np.ascontiguousarray(np.transpose(temp, (0, 4, 1, 2, 3)))
        # temp = torch.from_numpy(temp)
        # input = temp.requires_grad_(True)
        # target_index = None

        # idx_list = None                                                           # all
        # idx_list = [random.sample(range(64), 16) for _ in range(images.shape[0])]  # random
        idx_list = np.argsort(grad_list, 1)[:, -4:].tolist()                      # gradient
        # idx_list, mask_list = grad_cam(input, target_index)                       # gradcam

        if idx_list == None:
            adv_images = images + self.eps * grad_sign
            adv_images = torch.clamp(adv_images, min=-1, max=1).detach()
        else:
            adv_images = torch.zeros(images.shape, device=self.device)
            if self.weight:
                for i in range(images.shape[0]):
                    for j, idx in enumerate(idx_list[i]):
                        adv_images[i, :, idx, :, :] = (j + 1) / 255 * grad_sign[i, :, idx, :, :]
            else:
                for i in range(images.shape[0]):
                    adv_images[i, :, idx_list[i], :, :] = self.eps * grad_sign[i, :, idx_list[i], :, :]
            adv_images += images
            adv_images = torch.clamp(adv_images, min=-1, max=1).detach()

        return adv_images, torch.norm(adv_images - images, p=1)

class FGSM2(Attack):
    def __init__(self, model, eps):
        super(FGSM2, self).__init__("FGSM2", model)
        self.eps = (eps / self.std).view((3, 1, 1, 1)).to(self.device)

    def forward(self, images, labels, idx_list):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
        grad_sign = grad.sign()

        idx_set = [set() for _ in range(images.shape[0])]
        for i in range(images.shape[0]):
            for j in idx_list[i]:
                idx_set[i].add(j)

        adv_images = torch.zeros(images.shape, device=self.device)
        for i in range(images.shape[0]):
            adv_images[i, :, idx_list[i], :, :] = self.eps * grad_sign[i, :, idx_list[i], :, :]
        adv_images += images
        adv_images = torch.min(torch.max(adv_images, self.min), self.max).detach()

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