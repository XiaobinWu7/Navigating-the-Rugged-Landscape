from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import time

import numpy as np
import scipy.io as sio
import open3d as o3d
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import importlib

from Lib.loss_utils import curvature_loss, _get_kappa_ori, _get_kappa_adv, chamfer_loss, hausdorff_loss
from Attacker_AOF import Attack_GGS_CGC_NGS
from Attacker_AOF import Attack_Default
from Lib.utility import (Average_meter, Count_converge_iter, Count_loss_iter, SaveGradStats,
                         _compare, accuracy, estimate_normal_via_ori_normal,
                         farthest_points_sample)

cudnn.deterministic = True
cudnn.benchmark = False


def main(cfg):
    if cfg.attack_label == 'Untarget':
        targeted = False
    else:
        targeted = True

    print('=>Creating dir')

    # 简化保存路径命名，使其看起来更干净专业
    saved_root = os.path.join('Attack_Results', cfg.dataset, cfg.log_dir + '_' + str(cfg.strategy))
    saved_dir = os.path.join(saved_root, 'Run_01')

    print('==>Successfully created {}'.format(saved_dir))

    trg_dir = os.path.join(saved_dir, 'Mat')
    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)
    pc_dir = os.path.join(saved_dir, 'PC')
    if not os.path.exists(pc_dir):
        os.makedirs(pc_dir)
    bi_trg_dir = os.path.join(saved_dir, 'Mat_Bi')
    if not os.path.exists(bi_trg_dir):
        os.makedirs(bi_trg_dir)

    seed = 0 if cfg.id == 0 else int(time.time())
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    '''MODEL LOADING'''
    if cfg.dataset == 'ModelNet':
        num_class = 40
        cfg.classes = 40
        experiment_dir = os.path.join('log', 'classification_modelnet40', cfg.log_dir)
        data_file = cfg.data_dir_file1
    elif cfg.dataset == 'ShapeNetPart':
        num_class = 16
        cfg.classes = 16
        experiment_dir = os.path.join('log', 'classification_shapenet', cfg.log_dir)
        data_file = cfg.data_dir_file2

    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    if cfg.log_dir == "dgcnn":
        net = model.get_model(cfg, output_channels=num_class)
    elif cfg.log_dir == "curvenet_cls":
        net = model.CurveNet(num_class)
    else:
        net = model.get_model(num_class, normal_channel=cfg.use_normals)

    if not cfg.use_cpu:
        net = net.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')

    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()
    print('\nSuccessfully load pretrained-model from {}\n'.format(cfg.log_dir))

    '''DATA LOADING'''
    from Provider.modelnet10_instance250 import ModelNet40
    test_dataset = ModelNet40(data_mat_file=data_file, resample_num=-1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                                              drop_last=False, num_workers=cfg.num_workers, pin_memory=True)

    '''RECORDERS'''
    num_attack_success = 0
    cnt_ins = test_dataset.start_index
    cnt_all = 0
    num_attack_classes = 1 if cfg.attack_label in ['Untarget', 'Random'] else 9

    average_step = 0.0
    median_step_list = []

    l2_dist, bi_l2_dist = 0, 0
    median_l2_list, bi_median_l2_list = [], []
    chamfer_dist, hd_dist, curv_value = 0, 0, 0
    bi_chamfer_dist, bi_hd_dist, bi_curv_value = 0, 0, 0
    attack_time = 0

    for i, data in enumerate(test_loader):
        pc, normal, gt_labels = data[0], data[1], data[2]
        if pc.size(3) == 3:
            pc = pc.permute(0, 1, 3, 2)
        if normal.size(3) == 3:
            normal = normal.permute(0, 1, 3, 2)

        bs, l, _, n = pc.size()
        b = bs * l

        pc = pc.view(b, 3, n).cuda()
        ori_pc = pc.clone()
        normal = normal.squeeze(0).cuda()
        gt_target = gt_labels.view(-1).cuda()

        attack1_time = time.time()

        if cfg.strategy == 'GGS_CGC_NGU':
            adv_pc, bi_adv_pc, targeted_label, attack_success_indicator, best_attack_step = Attack_GGS_CGC_NGS.attack(
                net, data, cfg, i, len(test_loader), saved_dir)
        elif cfg.strategy == 'Default':
            adv_pc, bi_adv_pc, targeted_label, attack_success_indicator, best_attack_step = Attack_Default.attack(
                    net, data,
                    cfg, i,
                    len(test_loader),
                    saved_dir)

        attack2_time = time.time()
        attack_time += attack2_time - attack1_time

        average_step += sum(best_attack_step)
        median_step_list.append(best_attack_step[0])
        eval_num = 1

        # Distance metrics
        dist = torch.sqrt(torch.sum((adv_pc - ori_pc) ** 2, dim=[1, 2]) + torch.tensor(1e-7))
        median_l2_list.append(dist[0])
        l2_dist += dist.sum()

        bi_dist = torch.sqrt(torch.sum((bi_adv_pc - ori_pc) ** 2, dim=[1, 2]) + torch.tensor(1e-7))
        bi_median_l2_list.append(bi_dist[0])
        bi_l2_dist += bi_dist.sum()

        if cfg.eval_complex_metrics:
            kappa_ori = _get_kappa_ori(ori_pc, normal, cfg.curv_loss_knn)

            adv_kappa, normal_curr_iter = _get_kappa_adv(adv_pc, ori_pc, normal, cfg.curv_loss_knn)
            curv_value += curvature_loss(adv_pc, ori_pc, adv_kappa, kappa_ori)

            bi_adv_kappa, bi_normal_curr_iter = _get_kappa_adv(bi_adv_pc, ori_pc, normal, cfg.curv_loss_knn)
            bi_curv_value += curvature_loss(bi_adv_pc, ori_pc, bi_adv_kappa, kappa_ori)

            chamfer_dist += chamfer_loss(adv_pc, ori_pc)
            hd_dist += hausdorff_loss(adv_pc, ori_pc)
            bi_chamfer_dist += chamfer_loss(bi_adv_pc, ori_pc)
            bi_hd_dist += hausdorff_loss(bi_adv_pc, ori_pc)

        # Evaluate model accuracy and save results
        for _ in range(0, eval_num):
            with torch.no_grad():
                eval_points = farthest_points_sample(adv_pc, cfg.npoint) if adv_pc.size(2) > cfg.npoint else adv_pc
                bi_eval_points = bi_adv_pc
                test_adv_output = net(eval_points)

            output_label = torch.argmax(test_adv_output).item()
            attack_success_iter = _compare(output_label, targeted_label, gt_target.cuda(), targeted)
            try:
                attack_success += attack_success_iter
            except:
                attack_success = attack_success_iter

        saved_pc = eval_points.cpu().numpy()
        bi_saved_pc = bi_eval_points.cpu().numpy()

        for k in range(b):
            if attack_success_indicator[k].item():
                num_attack_success += 1

            name = 'adv_' + str(cnt_ins + k // num_attack_classes) + '_gt' + str(gt_target[k].item()) + '_attack' + str(torch.max(test_adv_output, 1)[1].data[k].item())

            sio.savemat(os.path.join(saved_dir, 'Mat', name + '.mat'),
                        {"adversary_point_clouds": saved_pc[k], 'gt_label': gt_target[k].item(),
                         'attack_label': torch.max(test_adv_output, 1)[1].data[k].item()})

            # Save PLY for visualization
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(saved_pc[k].transpose(1, 0))
            o3d.io.write_point_cloud(os.path.join(saved_dir, 'PC', name + '.ply'), pcd)

        cnt_ins = cnt_ins + bs
        cnt_all = cnt_all + b

    print('\n================ Results ================')
    print('Attack Success Rate: {0:.2f}%'.format(num_attack_success / float(cnt_all) * 100))
    print('Average Attack Steps: {0:.3f}'.format(average_step / float(cnt_all)))

    median_l2_list.sort()
    bi_median_l2_list.sort()
    median_l2 = median_l2_list[cnt_all // 2]
    bi_median_l2 = bi_median_l2_list[cnt_all // 2]

    print('\n[Standard Adversarial Stats]')
    print('Average L2 Distance: {0:.3f}'.format(l2_dist.item() / float(cnt_all)))
    print('Median L2 Distance: {0:.3f}'.format(median_l2))

    print('\n[Binary Search Projected Stats]')
    print('Average Bi-L2 Distance: {0:.3f}'.format(bi_l2_dist.item() / float(cnt_all)))
    print('Median Bi-L2 Distance: {0:.3f}'.format(bi_median_l2))

    if cfg.eval_complex_metrics:
        print('\n[Complex Geometric Metrics]')
        print('Average Chamfer Distance: {0:.3f}'.format(chamfer_dist.item() * 1e3 / float(cnt_all)))
        print('Average Hausdorff Distance: {0:.3f}'.format(hd_dist.item() * 1e2 / float(cnt_all)))
        print('Average Curvature Distance: {0:.3f}'.format(curv_value.item() * 1e2 / float(cnt_all)))

    print('\nTotal Attack Time: {0:.3f} S'.format(attack_time))
    print('=========================================')
    print('Finish!')

    return saved_dir


if __name__ == '__main__':
    stat_time = time.time()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'models'))

    parser = argparse.ArgumentParser(description='Point Cloud Attacking')

    parser.add_argument('--strategy', default='GGS_CGC_NGU', type=str, choices=['Default', 'GGS_CGC_NGU'], help='Attack strategy')
    parser.add_argument('--eval_complex_metrics', action='store_true', default=False, help='Enable Chamfer, Hausdorff and Curvature evaluation (requires PyTorch3D)')

    parser.add_argument('--id', type=int, default=0, help='')
    parser.add_argument('--log_dir', type=str, default=r'dgcnn', choices=['pointnet', 'dgcnn', 'pointnet2_msg','curvenet_cls'])
    parser.add_argument('--dataset', type=str, default='ShapeNetPart', choices=['ModelNet', 'ShapeNetPart'])

    parser.add_argument('--data_dir_file1', default=r'./data/modelnet10_250instances1024_all.mat', type=str)
    parser.add_argument('--data_dir_file2', default=r'./data/shapenet10_226instances1024_all.mat', type=str)

    parser.add_argument('-c', '--classes', default=40, type=int)
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('--npoint', default=1024, type=int)

    # Algorithm Parameters (GGS & CGC)
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer: adam | sgd')
    parser.add_argument('--attack_label', type=str, default='Untarget', help='Attack label setting')
    parser.add_argument('--binary_max_steps', type=int, default=1, help='Number of binary search steps')
    parser.add_argument('--iter_max_steps', type=int, default=100, help='Maximum optimization iterations')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--step_a', type=float, default=0.15, help='Inner update step size')
    parser.add_argument('--tau', type=float, default=1.0, help='Fusion coefficient lambda')

    parser.add_argument('--task_num', type=int, default=1, help='Number of calibration cycles K')
    parser.add_argument('--sample_size', type=int, default=10, help='GGS sample size N')
    parser.add_argument('--gaus_noise', type=float, default=0.003, help='Gaussian noise radius sigma')

    parser.add_argument('--initial_const', type=float, default=1, help='Initial trade-off constant')
    parser.add_argument('--cc_linf', type=float, default=0.18, help='Perturbation budget epsilon')
    parser.add_argument('--eval_num', type=int, default=1, help='Number of repeated evaluations')

    # Additional attack controls
    parser.add_argument('--low_pass', type=int, default=100, help='Low-pass number for AOF only')
    parser.add_argument('--step_alpha', type=float, default=5, help='Step size for PGD-like attack')
    parser.add_argument('--mu', type=float, default=1, help='Momentum factor for MI-FGSM-like attack')

    # Loss Config
    parser.add_argument('--cls_loss_type', type=str, default='Margin', choices=['Margin', 'CE'], help='Classification loss type')
    parser.add_argument('--confidence', type=float, default=15.0)
    parser.add_argument('--dis_loss_type', type=str, default='None',
                        choices=['CD', 'L2', 'KNN', 'None'], help='Distance loss type')
    parser.add_argument('--dis_loss_weight', type=float, default=0.0, help='Weight of distance loss')
    parser.add_argument('--is_cd_single_side', action='store_true', default=False, help='Use single-side Chamfer distance')

    parser.add_argument('--hd_loss_weight', type=float, default=0.0, help='Weight of Hausdorff loss')
    parser.add_argument('--curv_loss_weight', type=float, default=0.0, help='Weight of curvature loss')
    parser.add_argument('--uniform_loss_weight', type=float, default=0.0, help='Weight of uniform loss')

    parser.add_argument('--knn_smoothing_loss_weight', type=float, default=5.0, help='Weight of KNN smoothing loss')
    parser.add_argument('--laplacian_loss_weight', type=float, default=0, help='Weight of Laplacian loss for mesh')
    parser.add_argument('--edge_loss_weight', type=float, default=0, help='Weight of edge loss for mesh')

    # Evaluation Config
    parser.add_argument('--curv_loss_knn', type=int, default=16, help='KNN size for curvature estimation')
    parser.add_argument('--knn_smoothing_k', type=int, default=5, help='K for KNN smoothing loss')
    parser.add_argument('--knn_threshold_coef', type=float, default=1.10, help='Threshold coefficient for KNN smoothing')

    # System & Log
    parser.add_argument('-j', '--num_workers', type=int, default=4, metavar='N', help='Number of data loading workers')
    parser.add_argument('--use_normals', action='store_true', default=False, help='Use normals')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='Use CPU mode')

    parser.add_argument('--is_save_normal', action='store_true', default=False, help='Save normals')
    parser.add_argument('--is_debug', action='store_true', default=False, help='Enable debug mode')
    parser.add_argument('--is_low_memory', action='store_true', default=False, help='Enable low-memory mode')

    parser.add_argument('--is_real_offset', action='store_true', default=False, help='Use real offset projection')
    parser.add_argument('--is_pro_grad', action='store_true', default=False, help='Project gradients')

    # Jitter settings
    parser.add_argument('--is_pre_jitter_input', action='store_true', default=False, help='Apply jitter before optimization')
    parser.add_argument('--is_previous_jitter_input', action='store_true', default=False, help='Reuse previous jitter input')
    parser.add_argument('--calculate_project_jitter_noise_iter', type=int, default=50, help='Interval for recomputing projected jitter noise')
    parser.add_argument('--jitter_k', type=int, default=16, help='K for jitter estimation')
    parser.add_argument('--jitter_sigma', type=float, default=0.01, help='Sigma for jitter noise')
    parser.add_argument('--jitter_clip', type=float, default=0.05, help='Clip value for jitter noise')

    # DGCNN
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Number of nearest neighbors to use')
    parser.add_argument('--dropout', type=float, default=0.5, help='Initial dropout rate')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'], help='Scheduler to use: [cos, step]')

    # Mesh Opt
    parser.add_argument('--is_partial_var', dest='is_partial_var', action='store_true', default=False, help='Use partial variable optimization')
    parser.add_argument('--knn_range', type=int, default=3, help='KNN range for mesh optimization')
    parser.add_argument('--is_subsample_opt', dest='is_subsample_opt', action='store_true', default=False, help='Enable subsample optimization')
    parser.add_argument('--is_use_lr_scheduler', dest='is_use_lr_scheduler', action='store_true', default=False, help='Use learning rate scheduler')

    cfg = parser.parse_args()

    if cfg.strategy == 'Default':
        cfg.lr = 0.01
    elif cfg.strategy == 'GGS_CGC_NGU':
        cfg.lr = 0.15

    print(cfg, '\n')

    main(cfg)
