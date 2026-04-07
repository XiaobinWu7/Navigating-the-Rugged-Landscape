from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import sys
import time
import ipdb
import numpy as np
# import open3d as o3d
from pytorch3d.ops import knn_points, knn_gather
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients
import torch.backends.cudnn as cudnn
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

sys.path.append(os.path.join(ROOT_DIR, 'Lib'))
from Lib.utility import estimate_perpendicular, _compare, farthest_points_sample, pad_larger_tensor_with_index_batch
from Lib.loss_utils import norm_l2_loss, chamfer_loss, pseudo_chamfer_loss, hausdorff_loss, curvature_loss, uniform_loss, _get_kappa_ori, _get_kappa_adv
from Lib.dist_utils import ChamferkNNDist


def _run_binary_search(k, net, ori_pc, adv_pc, target, gt_target, targeted):
    epsilon = 1
    best_attack_pc = adv_pc

    def try_epsilon(epsilon):
        bi_se_pc = (epsilon * ori_pc + (1 - epsilon) * adv_pc).unsqueeze(0)
        bi_se_output = net(bi_se_pc)
        output_label = torch.argmax(bi_se_output).item()
        attack_success = _compare(output_label, target, gt_target.cuda(), targeted).item()
        return attack_success, bi_se_pc.squeeze(0)

    bad = 0
    good = epsilon

    for i in range(k):
        epsilon = (good + bad) / 2
        atk_success, bi_se_pc = try_epsilon(epsilon)
        if atk_success:
            bad = epsilon
            best_attack_pc = bi_se_pc
        else:
            good = epsilon

    return best_attack_pc


def offset_proj(offset, ori_pc, ori_normal, project='dir'):
    # offset: shape [b, 3, n], perturbation offset of each point
    # normal: shape [b, 3, n], normal vector of the object
    condition_inner = torch.zeros(offset.shape).cuda().byte()

    intra_KNN = knn_points(offset.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    normal = knn_gather(ori_normal.permute(0,2,1), intra_KNN.idx).permute(0,3,1,2).squeeze(3).contiguous() # [b, 3, n]
    normal_len = (normal**2).sum(1, keepdim=True).sqrt()
    normal_len_expand = normal_len.expand_as(offset) #[b, 3, n]

    # add 1e-6 to avoid dividing by zero
    offset_projected = (offset * normal / (normal_len_expand + 1e-6)).sum(1,keepdim=True) * normal / (normal_len_expand + 1e-6)

    # let perturb be the projected ones
    offset = torch.where(condition_inner, offset, offset_projected)

    return offset


def find_offset(ori_pc, adv_pc):
    intra_KNN = knn_points(adv_pc.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    knn_pc = knn_gather(ori_pc.permute(0,2,1), intra_KNN.idx).permute(0,3,1,2).squeeze(3).contiguous() # [b, 3, n]
    real_offset = adv_pc - knn_pc

    return real_offset


def lp_clip(offset, cc_linf):
    lengths = (offset**2).sum(1, keepdim=True).sqrt() #[b, 1, n]
    lengths_expand = lengths.expand_as(offset) # [b, 3, n]

    condition = lengths > 1e-6
    offset_scaled = torch.where(condition, offset / lengths_expand * cc_linf, torch.zeros_like(offset))

    condition = lengths < cc_linf
    offset = torch.where(condition, offset, offset_scaled)

    return offset


def _forward_step(net, pc_ori, input_curr_iter, normal_ori, ori_kappa, target, scale_const, cfg, targeted, dist_func):
    #needed cfg:[arch, classes, cls_loss_type, confidence, dis_loss_type, is_cd_single_side, dis_loss_weight, hd_loss_weight, curv_loss_weight, curv_loss_knn]
    b, _, n = input_curr_iter.size()

    output_curr_iter = net(input_curr_iter)

    if cfg.cls_loss_type == 'Margin':
        target_onehot = torch.zeros(target.size() + (cfg.classes,)).cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)

        fake = (target_onehot * output_curr_iter).sum(1)
        other = ((1. - target_onehot) * output_curr_iter - target_onehot * 10000.).max(1)[0]

        if targeted:
            # if targeted, optimize for making the other class most likely
            cls_loss = torch.clamp(other - fake + cfg.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            cls_loss = torch.clamp(fake - other + cfg.confidence, min=0.)  # equiv to max(..., 0.)

    elif cfg.cls_loss_type == 'CE':

        if targeted:
            cls_loss = nn.CrossEntropyLoss(reduction='none').cuda()(output_curr_iter, Variable(target, requires_grad=False))
        else:
            cls_loss = - nn.CrossEntropyLoss(reduction='none').cuda()(output_curr_iter, Variable(target, requires_grad=False))
    elif cfg.cls_loss_type == 'None':
        cls_loss = torch.FloatTensor(b).zero_().cuda()
    else:
        assert False, 'Not support such clssification loss'

    info = 'cls_loss: {0:6.4f}\t'.format(cls_loss.mean().item())

    if cfg.dis_loss_type == 'CD':
        if cfg.is_cd_single_side:
            dis_loss = pseudo_chamfer_loss(input_curr_iter, pc_ori)
        else:
            dis_loss = chamfer_loss(input_curr_iter, pc_ori)

        constrain_loss = cfg.dis_loss_weight * dis_loss
        info = info + 'cd_loss: {0:6.4f}\t'.format(dis_loss.mean().item())
    elif cfg.dis_loss_type == 'L2':
        assert cfg.hd_loss_weight ==0
        dis_loss = norm_l2_loss(input_curr_iter, pc_ori)
        constrain_loss = cfg.dis_loss_weight * dis_loss
        info = info + 'l2_loss: {0:6.4f}\t'.format(dis_loss.mean().item())
    elif cfg.dis_loss_type == 'KNN':
        assert cfg.hd_loss_weight == 0
        dis_loss = dist_func(input_curr_iter.transpose(1, 2).contiguous(), pc_ori.transpose(1, 2).contiguous()) * n
        constrain_loss = cfg.dis_loss_weight * dis_loss
        info = info + 'cd_loss: {0:6.4f}\t'.format(dis_loss.mean().item())
    elif cfg.dis_loss_type == 'None':
        dis_loss = 0
        constrain_loss = 0
    else:
        assert False, 'Not support such distance loss'

    # hd_loss
    if cfg.hd_loss_weight !=0:
        hd_loss = hausdorff_loss(input_curr_iter, pc_ori)
        constrain_loss = constrain_loss + cfg.hd_loss_weight * hd_loss

        info = info+'hd_loss : {0:6.4f}\t'.format(hd_loss.mean().item())
    else:
        hd_loss = 0

    # nor loss
    if cfg.curv_loss_weight !=0:
        adv_kappa, normal_curr_iter = _get_kappa_adv(input_curr_iter, pc_ori, normal_ori, cfg.curv_loss_knn)
        curv_loss = curvature_loss(input_curr_iter, pc_ori, adv_kappa, ori_kappa)
        constrain_loss = constrain_loss + cfg.curv_loss_weight * curv_loss
        info = info+'curv_loss : {0:6.4f}\t'.format(curv_loss.mean().item())
    else:
        normal_curr_iter = torch.zeros(b, 3, n).cuda()
        curv_loss = 0

    # uniform loss
    if cfg.uniform_loss_weight != 0:
        uniform = uniform_loss(input_curr_iter)
        constrain_loss = constrain_loss + cfg.uniform_loss_weight * uniform
        info = info+'uniform : {0:6.4f}\t'.format(uniform.mean().item())
    else:
        uniform = 0

    scale_const = scale_const.float().cuda()
    loss_n = cls_loss + scale_const * constrain_loss
    loss = loss_n.mean()

    return output_curr_iter, normal_curr_iter, loss, loss_n, cls_loss, dis_loss, hd_loss, curv_loss, constrain_loss, info


def attack(net, input_data, cfg, i, loader_len, saved_dir=None):

    if cfg.attack_label == 'Untarget':
        targeted = False
    else:
        targeted = True

    step_print_freq = 1

    pc = input_data[0]
    normal = input_data[1]
    gt_labels = input_data[2]
    if pc.size(3) == 3:
        pc = pc.permute(0,1,3,2)
    if normal.size(3) == 3:
        normal = normal.permute(0,1,3,2)

    bs, l, _, n = pc.size()
    b = bs*l

    pc_ori = pc.view(b, 3, n).cuda()
    normal_ori = normal.view(b, 3, n).cuda()
    gt_target = gt_labels.view(-1)

    if cfg.attack_label == 'Untarget':
        target = gt_target.cuda()
    else:
        target = input_data[3].view(-1).cuda()

    if cfg.curv_loss_weight !=0:
        kappa_ori = _get_kappa_ori(pc_ori, normal_ori, cfg.curv_loss_knn)
    else:
        kappa_ori = None

    lower_bound = torch.ones(b) * 0
    scale_const = torch.ones(b) * cfg.initial_const
    upper_bound = torch.ones(b) * 1e10

    best_loss = [1e10] * b
    best_attack = pc_ori.clone()
    bi_best_attack = pc_ori.clone()
    best_attack_step = [cfg.iter_max_steps] * b
    best_attack_BS_idx = [-1] * b
    dist_func = ChamferkNNDist(chamfer_method='adv2ori',
                               knn_k=5, knn_alpha=1.05,
                               chamfer_weight=5., knn_weight=3.)

    for search_step in range(cfg.binary_max_steps):
        iter_best_score = [-1] * b
        attack_success = torch.zeros(b).cuda()

        input_all = None

        for step in range(cfg.iter_max_steps):
            if cfg.is_partial_var:
                if step % 50 == 0:
                    with torch.no_grad():
                        init_point_idx = np.random.randint(n)
                        intra_KNN = knn_points(pc_ori[:, :, init_point_idx].unsqueeze(2).permute(0,2,1), pc_ori.permute(0,2,1), K=cfg.knn_range+1) #[dists:[b,n,cfg.knn_range+1], idx:[b,n,cfg.knn_range+1]]
                    part_offset = torch.zeros(b, 3, cfg.knn_range).cuda()
                    part_offset.requires_grad_()

                    if cfg.optim == 'adam':
                        optimizer = torch.optim.Adam([part_offset], lr=cfg.lr)
                    elif cfg.optim == 'sgd':
                        optimizer = torch.optim.SGD([part_offset], lr=cfg.lr, momentum=0.9)
                    else:
                        assert False, 'Wrong optimizer!'

                    try:
                        periodical_pc = input_all.detach().clone()
                    except:
                        periodical_pc = pc_ori.clone()
            else:
                if step == 0:
                    offset = torch.zeros(b, 3, n).cuda()

                    offset.requires_grad_()
                    if cfg.optim == 'adam':
                        optimizer = optim.Adam([offset], lr=cfg.lr)
                    elif cfg.optim == 'sgd':
                        optimizer = optim.SGD([offset], lr=cfg.lr)
                    else:
                        assert False, 'Not support such optimizer.'

                    periodical_pc = pc_ori.clone()

            if cfg.is_partial_var:
                offset = pad_larger_tensor_with_index_batch(part_offset, intra_KNN.idx.tolist(), n)
            input_all = periodical_pc + offset

            if (input_all.size(2) > cfg.npoint) and (not cfg.is_partial_var) and cfg.is_subsample_opt:
                input_curr_iter = farthest_points_sample(input_all, cfg.npoint)
            else:
                input_curr_iter = input_all

            with torch.no_grad():
                for k in range(b):
                    if input_curr_iter.size(2) < input_all.size(2):
                        batch_k_pc = farthest_points_sample(torch.cat([input_all[k].unsqueeze(0)]*cfg.eval_num), cfg.npoint)

                        if cfg.log_dir == "dgcnn":
                            batch_k_adv_output = net(batch_k_pc)
                        else:
                            batch_k_adv_output, _ = net(batch_k_pc, )

                        attack_success[k] = _compare(torch.max(batch_k_adv_output, 1)[1].data, target[k], gt_target[k], targeted).sum() > 0.5 * cfg.eval_num
                        output_label = torch.max(batch_k_adv_output,1)[1].mode().values.item()
                    else:
                        adv_output = net(input_curr_iter[k].unsqueeze(0))
                        output_label = torch.argmax(adv_output).item()

                        attack_success[k] = _compare(output_label, target[k], gt_target[k].cuda(), targeted).item()

                    if attack_success[k] :
                        best_loss[k] = 1
                        best_attack[k] = input_all.data[k].clone()
                        best_attack_BS_idx[k] = search_step
                        best_attack_step[k] = step
                        bi_best_attack[k] = _run_binary_search(20, net, pc_ori[k].clone(), best_attack[k], target[k], gt_target[k].cuda(), targeted)

                        return best_attack, bi_best_attack, target, (np.array(best_loss) < 1e10), best_attack_step

            if cfg.is_pre_jitter_input:
                if step % cfg.calculate_project_jitter_noise_iter == 0:
                    project_jitter_noise = estimate_perpendicular(input_curr_iter, cfg.jitter_k, sigma=cfg.jitter_sigma, clip=cfg.jitter_clip)
                else:
                    project_jitter_noise = project_jitter_noise.clone()
                input_curr_iter.data = input_curr_iter.data + project_jitter_noise

            _, normal_curr_iter, loss, loss_n, cls_loss, dis_loss, hd_loss, nor_loss, constrain_loss, info = _forward_step(net, pc_ori, input_curr_iter, normal_ori, kappa_ori, target, scale_const, cfg, targeted, dist_func)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for saving
            if (step % 50 == 0) and cfg.is_debug:
                fout = open(os.path.join(saved_dir, 'Obj', str(step)+'bf.xyz'), 'w')
                k = -1
                for m in range(input_curr_iter.shape[2]):
                    fout.write('%f %f %f %f %f %f\n' % (input_curr_iter[k, 0, m], input_curr_iter[k, 1, m], input_curr_iter[k, 2, m], normal_curr_iter[k, 0, m], normal_curr_iter[k, 1, m], normal_curr_iter[k, 2, m]))
                fout.close()

            if cfg.is_pro_grad:
                with torch.no_grad():
                    if cfg.is_real_offset:
                        offset.data = find_offset(pc_ori, periodical_pc + offset).data

                    proj_offset = offset_proj(offset, pc_ori, normal_ori)
                    offset.data = proj_offset.data

            if cfg.cc_linf != 0:
                with torch.no_grad():
                    proj_offset = lp_clip(offset, cfg.cc_linf)
                    offset.data = proj_offset.data

            if cfg.is_debug:
                info = '[{5}/{6}][{0}/{1}][{2}/{3}] \t loss: {4:6.4f}\t output:{7}\t'.format(search_step+1, cfg.binary_max_steps, step+1, cfg.iter_max_steps, loss.item(), i, loader_len, output_label) + info
            else:
                info = '[{5}/{6}][{0}/{1}][{2}/{3}] \t loss: {4:6.4f}\t'.format(search_step+1, cfg.binary_max_steps, step+1, cfg.iter_max_steps, loss.item(), i, loader_len) + info

            if step % step_print_freq == 0 or step == cfg.iter_max_steps - 1:
                print(info)

        if cfg.is_debug:
            ipdb.set_trace()

        # adjust the scale constants
        for k in range(b):
            if _compare(output_label, target[k], gt_target[k].cuda(), targeted).item() and iter_best_score[k] != -1:
                lower_bound[k] = max(lower_bound[k], scale_const[k])
                if upper_bound[k] < 1e9:
                    scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5
                else:
                    scale_const[k] *= 2
            else:
                upper_bound[k] = min(upper_bound[k], scale_const[k])
                if upper_bound[k] < 1e9:
                    scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5

    return best_attack, bi_best_attack, target, (np.array(best_loss) < 1e10), best_attack_step