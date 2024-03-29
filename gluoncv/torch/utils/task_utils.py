# pylint: disable=line-too-long
"""
Utility functions for task
"""
import os
import time
import numpy as np
import random
import json

import torch
import torch.nn as nn

import cv2

from .torchattack import FGSM, PGD, VFA

from .utils import AverageMeter, accuracy

def train_classification(base_iter, model, dataloader, epoch, criterion, optimizer, cfg, writer=None):
    """Task of training video classification"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for step, data in enumerate(dataloader):
        base_iter = base_iter + 1
        train_batch = data[0].cuda()
        train_label = data[1].cuda()
        train_label = train_label.long().view(-1)
        data_time.update(time.time() - end)
        outputs = model(train_batch)

        loss = criterion(outputs, train_label)
        prec1, prec5 = accuracy(outputs.data, train_label, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), train_label.size(0))
        top1.update(prec1.item(), train_label.size(0))
        top5.update(prec5.item(), train_label.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if step % cfg.CONFIG.LOG.DISPLAY_FREQ == 0 and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                lr = param['lr']
            print('lr: ', lr)
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(data_time=data_time.val, batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(top1_acc=top1.avg, top5_acc=top5.avg)
            print(print_string)
            iteration = base_iter
            writer.add_scalar('train_loss_iteration', losses.avg, iteration)
            writer.add_scalar('train_top1_acc_iteration', top1.avg, iteration)
            writer.add_scalar('train_top5_acc_iteration', top5.avg, iteration)
            writer.add_scalar('train_batch_size_iteration', train_label.size(0), iteration)
            writer.add_scalar('learning_rate', lr, iteration)
    return base_iter

def validation_classification(model, val_dataloader, epoch, criterion, cfg, writer):
    """Task of validating video classification"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, data in enumerate(val_dataloader):
            data_time.update(time.time() - end)
            val_batch = data[0].cuda()
            val_label = data[1].cuda()
            val_label = val_label.long().view(-1)

            outputs = model(val_batch)

            loss = criterion(outputs, val_label)
            prec1a, prec5a = accuracy(outputs.data, val_label, topk=(1, 5))

            losses.update(loss.item(), val_batch.size(0))
            top1.update(prec1a.item(), val_batch.size(0))
            top5.update(prec5a.item(), val_batch.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if step % cfg.CONFIG.LOG.DISPLAY_FREQ == 0 and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
                print('----validation----')
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_dataloader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(data_time=data_time.val, batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)
                print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(top1_acc=top1.avg, top5_acc=top5.avg)
                print(print_string)

        eval_path = cfg.CONFIG.LOG.EVAL_DIR
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)

        with open(os.path.join(eval_path, "{}.txt".format(cfg.DDP_CONFIG.GPU_WORLD_RANK)), 'w') as f:
            f.write("{} {} {}\n".format(losses.avg, top1.avg, top5.avg))
        torch.distributed.barrier()

        loss_lst, top1_lst, top5_lst = [], [], []
        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 and writer is not None:
            print("Collecting validation numbers")
            for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE):
                data = open(os.path.join(eval_path, "{}.txt".format(x))).readline().strip().split(" ")
                data = [float(x) for x in data]
                loss_lst.append(data[0])
                top1_lst.append(data[1])
                top5_lst.append(data[2])
            print("Global result:")
            print_string = 'loss: {loss:.5f}'.format(loss=np.mean(loss_lst))
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(top1_acc=np.mean(top1_lst), top5_acc=np.mean(top5_lst))
            print(print_string)
            writer.add_scalar('val_loss_epoch', np.mean(loss_lst), epoch)
            writer.add_scalar('val_top1_acc_epoch', np.mean(top1_lst), epoch)
            writer.add_scalar('val_top5_acc_epoch', np.mean(top5_lst), epoch)

def adversarial_classification(model, val_dataloader, epoch, criterion, cfg, writer):
    """Task of validating video classification"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    TAP = AverageMeter()
    frames = AverageMeter()
    grad_ratio = AverageMeter()
    grad_var = AverageMeter()
    suc_var = AverageMeter()
    fail_var = AverageMeter()
    tot_var = AverageMeter()
    suc_dis_var = AverageMeter()
    fail_dis_var = AverageMeter()
    tot_dis_var = AverageMeter()    
    suc_var_ratio = AverageMeter()
    fail_var_ratio = AverageMeter()
    tot_var_ratio = AverageMeter()
    model.eval()

    random_list = []
    with open('idx.json', 'r') as f:
        var_dict = json.load(f)
    for idx, (var, distance_var) in list(var_dict.items()):
        if distance_var > 1.0 and distance_var <= 1.5:
            temp_list = []
            for i in idx[1:-1].split(','):
                temp_list.append(int(i))
            random_list.append(temp_list)
    atk = VFA(cfg, model, random_list)

    perturbation = 0.
    sum_frames = 0
    total = 0
    end = time.time()
    for step, data in enumerate(val_dataloader):
        data_time.update(time.time() - end)
        val_batch = data[0].cuda()
        val_label = data[1].cuda()
        val_label = val_label.long().view(-1)
        total += val_batch.shape[0]

        adv_batch, pert, num_frames, ratio, var, idx_list = atk(val_batch, val_label)

        outputs = model(adv_batch)
        _outputs = model(val_batch).detach().cpu()
        for i in range(val_batch.shape[0]):
            var = np.var(idx_list[i])
            dis_var = np.var([np.sort(idx_list[i])[j+1] - np.sort(idx_list[i])[j] for j in range(cfg.CONFIG.ADV.FRAME-1)] + [np.sort(idx_list[i])[0] + cfg.CONFIG.DATA.CLIP_LEN - np.sort(idx_list[i])[cfg.CONFIG.ADV.FRAME-1]])
            var_ratio = dis_var / var
            if torch.argmax(outputs[i].detach().cpu()).item() != torch.argmax(_outputs[i]).item():
                suc_var.update(var, 1)
                suc_dis_var.update(dis_var, 1)
                suc_var_ratio.update(var_ratio, 1)
            else:
                fail_var.update(var, 1)
                fail_dis_var.update(dis_var, 1)
                fail_var_ratio.update(var_ratio, 1)
            tot_var.update(var, 1)
            tot_dis_var.update(dis_var, 1)
            tot_var_ratio.update(var_ratio, 1)

        loss = criterion(outputs, val_label)
        prec1a, prec5a = accuracy(outputs.data, val_label, topk=(1, 5))

        losses.update(loss.item(), val_batch.size(0))
        top1.update(prec1a.item(), val_batch.size(0))
        top5.update(prec5a.item(), val_batch.size(0))
        grad_ratio.update(ratio, val_batch.size(0) if ratio != 0 else 0)
        grad_var.update(var, val_batch.size(0))
        TAP.update(pert / val_batch.size(0), val_batch.size(0))
        frames.update(num_frames / val_batch.size(0), val_batch.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if step % cfg.CONFIG.LOG.DISPLAY_FREQ == 0 and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            print('----adversarial----')
            print_string = f'Epoch: [{epoch}][{step + 1}/{len(val_dataloader)}]'
            print(print_string)
            print_string = f'data_time: {data_time.val:.3f}, batch time: {batch_time.val:.3f}'
            print(print_string)
            print_string = f'loss: {losses.avg:.5f}'
            print(print_string)
            print_string =  f'TAP: {TAP.sum}, avg_frames: {frames.avg:.2f}, grad_var: {grad_var.avg:.4f}, grad_ratio: {grad_ratio.avg:.2f}%'
            print(print_string)
            print_string = f'suc_var: {suc_var.avg:.4f}, fail_var: {fail_var.avg:.4f}, tot_var: {tot_var.avg:.4f}'
            print(print_string)
            print_string = f'suc_dis_var: {suc_dis_var.avg:.4f}, fail_dis_var: {fail_dis_var.avg:.4f}, tot_dis_var: {tot_dis_var.avg:.4f}'
            print(print_string)
            print_string = f'suc_var_ratio: {suc_var_ratio.avg:.4f}, fail_var_ratio: {fail_var_ratio.avg:.4f}, tot_var_ratio: {tot_var_ratio.avg:.4f}'
            print(print_string)
            print_string = f'Top-1 accuracy: {top1.avg:.2f}%, Top-5 accuracy: {top5.avg:.2f}%'
            print(print_string)

    eval_path = cfg.CONFIG.LOG.EVAL_DIR
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    with open(os.path.join(eval_path, "{}.txt".format(cfg.DDP_CONFIG.GPU_WORLD_RANK)), 'w') as f:
        f.write(f'{losses.avg} {top1.avg} {top5.avg} {TAP.sum} {frames.sum} {frames.avg} {grad_ratio.avg} {grad_var.avg} {suc_var.avg} {fail_var.avg} {tot_var.avg} {suc_dis_var.avg} {fail_dis_var.avg} {tot_dis_var.avg} {suc_var_ratio.avg} {fail_var_ratio.avg} {tot_var_ratio.avg}\n')
    torch.distributed.barrier()

    loss_lst, top1_lst, top5_lst, tap_lst, frames_sum_lst, frames_avg_lst, grad_ratio_lst, grad_var_lst = [], [], [], [], [], [], [], []
    suc_var_lst, fail_var_lst, tot_var_lst, suc_dis_var_lst, fail_dis_var_lst, tot_dis_var_lst, suc_var_ratio_lst, fail_var_ratio_lst, tot_var_ratio_lst = [], [], [], [], [], [], [], [], []
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 and writer is not None:
        print("Collecting validation numbers")
        for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE):
            data = open(os.path.join(eval_path, "{}.txt".format(x))).readline().strip().split(" ")
            data = [float(x) for x in data]
            loss_lst.append(data[0])
            top1_lst.append(data[1])
            top5_lst.append(data[2])
            tap_lst.append(data[3])
            frames_sum_lst.append(data[4])
            frames_avg_lst.append(data[5])
            grad_ratio_lst.append(data[6])
            grad_var_lst.append(data[7])
            suc_var_lst.append(data[8])
            fail_var_lst.append(data[9])
            tot_var_lst.append(data[10])
            suc_dis_var_lst.append(data[11])
            fail_dis_var_lst.append(data[12])
            tot_dis_var_lst.append(data[13])
            suc_var_ratio_lst.append(data[14])
            fail_var_ratio_lst.append(data[15])
            tot_var_ratio_lst.append(data[16])

        fout = open('temp.txt', 'a')
        print("Global result:")
        fout.write(f'{cfg.CONFIG.MODEL.NAME}/{cfg.CONFIG.ADV.METHOD}/{cfg.CONFIG.ADV.TYPE}/{cfg.CONFIG.ADV.FRAME}\n')
        print_string = f'loss: {np.mean(loss_lst):.5f}'
        print(print_string)
        print_string = f'TAP: {np.sum(tap_lst)}, avg_frames: {np.mean(frames_avg_lst):.2f}, grad_var: {np.mean(grad_var_lst):.4f}, grad_ratio: {np.mean(grad_ratio_lst):.2f}%'
        print(print_string)
        fout.write(print_string + '\n')
        print_string = f'suc_var: {np.mean(suc_var_lst):.4f}, fail_var: {np.mean(fail_var_lst):.4f}, tot_var: {np.mean(tot_var_lst):.4f}'
        print(print_string)
        fout.write(print_string + '\n')
        print_string = f'suc_dis_var: {np.mean(suc_dis_var_lst):.4f}, fail_dis_var: {np.mean(fail_dis_var_lst):.4f}, tot_dis_var: {np.mean(tot_dis_var_lst):.4f}'
        print(print_string)
        fout.write(print_string + '\n')
        print_string = f'suc_var_ratio: {np.mean(suc_var_ratio_lst):.4f}, fail_var_ratio: {np.mean(fail_var_ratio_lst):.4f}, tot_var_ratio: {np.mean(tot_var_ratio_lst):.4f}'
        print(print_string)
        fout.write(print_string + '\n')
        print_string = f'Top-1 accuracy: {np.mean(top1_lst):.2f}%, Top-5 accuracy: {np.mean(top5_lst):.2f}%'
        print(print_string)
        fout.write(print_string + '\n\n')
        writer.add_scalar('val_loss_epoch', np.mean(loss_lst), epoch)
        writer.add_scalar('val_top1_acc_epoch', np.mean(top1_lst), epoch)
        writer.add_scalar('val_top5_acc_epoch', np.mean(top5_lst), epoch)

def test_classification(model, test_loader, criterion, cfg, file):
    """Task of testing video classification"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    final_result = []

    with torch.no_grad():
        for step, (inputs, labels, ids, chunk_nb, split_nb) in enumerate(test_loader):
            data_time.update(time.time() - end)
            val_batch = inputs.cuda()
            val_label = labels.cuda()
            outputs = model(val_batch)
            loss = criterion(outputs, val_label)

            for i in range(outputs.size(0)):
                string = "{} {} {} {} {}\n".format(ids[i], \
                                                   str(outputs.data[i].cpu().numpy().tolist()), \
                                                   str(int(labels[i].cpu().numpy())), \
                                                   str(int(chunk_nb[i].cpu().numpy())), \
                                                   str(int(split_nb[i].cpu().numpy())))
                final_result.append(string)

            prec1, prec5 = accuracy(outputs.data, val_label, topk=(1, 5))
            losses.update(loss.item(), val_batch.size(0))
            top1.update(prec1.item(), val_batch.size(0))
            top5.update(prec5.item(), val_batch.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if step % cfg.CONFIG.LOG.DISPLAY_FREQ == 0 and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
                print('----Testing----')
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(0, step + 1, len(test_loader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)
                print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg,
                    top5_acc=top5.avg)
                print(print_string)
    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(top1.avg, top5.avg))
        for line in final_result:
            f.write(line)
