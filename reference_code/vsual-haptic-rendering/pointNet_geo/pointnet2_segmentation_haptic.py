import os.path as osp
import os 
import numpy as np
import time

import torch
import torch.nn as nn
import torch_geometric

from haptic.shared.haptic_dataset import PointNetHapticDataset as HapticDataset
from haptic.pointNet_geo.model import Net
from haptic.util import vv_to_args, get_force_weight, weighted_mse

import json
from chester import logger
from haptic.visual import plot_precision_recall_curve, do_plot, plot_traj
from torch.optim.lr_scheduler import ReduceLROnPlateau

def run(args, epoch, model, data_loader, device, optimizer, scheduler, criterion, 
    mode='train', save_visual_path=None):
    
    model, force_model = model
    optimizer, force_optimizer = optimizer
    criterion, force_criterion = criterion
    contact_scheduler, force_scheduler = scheduler
    if mode == 'train':
        model.train()
        force_model.train()
    elif mode == 'eval':
        model.eval()
        force_model.eval()

    total_loss = correct_nodes = total_nodes = 0
    total_force_loss = 0
    force_relative_error_pred = 0
    force_relative_error_gt = 0
    force_rmse_error_gt = 0

    all_contact_pred = []
    all_contact_label = []

    contact_forward_time = []
    force_forward_time = []

    with torch.set_grad_enabled(mode == 'train'):
        for i, data in enumerate(data_loader):
            data = data.to(device)
            if mode == 'train':
                if args.train_contact:
                    optimizer.zero_grad()
                if args.train_force:
                    force_optimizer.zero_grad()

            if args.use_gt_contact:
                out = data.contact_label
            else:
                beg = time.time()
                out = model(data)
                contact_forward_time.append(time.time() - beg)
            beg = time.time()
            force_out = force_model(data)
            force_forward_time.append(time.time() - beg)

            loss = torch.Tensor([0])
            if args.train_contact:
                loss = criterion(out, data.contact_label.view(-1, 1))

            force_loss = torch.Tensor([0])
            if args.train_force:
                weight = get_force_weight(data.contact_label.view(-1, 1), args.force_loss_mode, args.train_pos_label_weight)
                force_loss = force_criterion(force_out, data.force_label.view(-1, 1), weight.reshape(-1, 1))

            if mode == 'train':
                if args.train_contact:
                    loss.backward()
                    optimizer.step()
                if args.train_force:
                    force_loss.backward()
                    force_optimizer.step()

            total_loss += loss.item()
            total_force_loss += force_loss.item()

            out = out.view(-1, 1)
            data.contact_label = data.contact_label.view(-1, 1)
            correct_nodes += (out > 0).eq(data.contact_label).sum().item()
            total_nodes += data.x.shape[0]

            contact_label_numpy = data.contact_label.data.cpu().numpy()
            all_contact_label.append(contact_label_numpy)
            all_contact_pred.append((out.data.cpu().numpy()))

            out_numpy = out.data.cpu().numpy().flatten()
            contact_label_numpy = contact_label_numpy.flatten()
            force_out_numpy = force_out.data.cpu().numpy().flatten()[out_numpy > 0]
            force_target_numpy = data.force_label.data.cpu().numpy().flatten()[out_numpy > 0]
            if args.force_normalize_mean is not None:
                force_out_numpy = force_out_numpy * args.force_normalize_std + args.force_normalize_mean
                force_target_numpy = force_target_numpy * args.force_normalize_std + args.force_normalize_mean
            force_relative_error_pred += np.mean(np.abs(force_out_numpy - force_target_numpy) / (np.abs(force_target_numpy) + 1e-10))
            force_out_numpy = force_out.data.cpu().numpy().flatten()[contact_label_numpy == 1]
            force_target_numpy = data.force_label.data.cpu().numpy().flatten()[contact_label_numpy == 1]
            if args.force_normalize_mean is not None:
                force_out_numpy = force_out_numpy * args.force_normalize_std + args.force_normalize_mean
                force_target_numpy = force_target_numpy * args.force_normalize_std + args.force_normalize_mean
            force_relative_error_gt += np.mean(np.abs(force_out_numpy - force_target_numpy) / (np.abs(force_target_numpy) + 1e-10))                
            force_rmse_error_gt += np.sqrt(np.mean((force_out_numpy - force_target_numpy) ** 2))

            if ((epoch + 1) % args.plot_img_interval == 0 or epoch == args.epoch - 1):
                if i <= 10:
                    if args.force_normalize_mean is not None:
                        force_out = force_out * args.force_normalize_std + args.force_normalize_mean
                    do_plot(out, force_out, data, epoch, i, mode, save_visual_path)

        if mode == 'train':
            if args.train_contact:
                contact_scheduler.step(total_loss / len(data_loader))
            if args.train_force:
                force_scheduler.step(total_force_loss / len(data_loader))

    if ((epoch + 1) % args.plot_gif_interval == 0 or epoch == args.epoch - 1):
        print("plotting gif traj at epoch {}".format(epoch))
        plot_traj(args, model, force_model, data_loader, epoch, mode, save_visual_path, device)

    all_contact_label = np.concatenate(all_contact_label, axis=0).flatten()
    all_contact_pred = np.concatenate(all_contact_pred, axis=0).flatten()
    save_name = None
    if epoch % args.plot_img_interval == 0:
        save_name = osp.join(save_visual_path, 'precision-recall-{}-{}.png'.format(mode, epoch)) 
    # tp, fp, tn, fn, precision, recall, f1 = plot_precision_recall_curve(all_contact_pred, all_contact_label, save_name=save_name)
    tps, fps, tns, fns, precisions, recalls, f1s, _ = plot_precision_recall_curve(all_contact_pred, all_contact_label, save_name=save_name)
    best_f1_idx = np.argmax(f1s)
    best_f1 = f1s[best_f1_idx]
    best_tp, best_fp, best_tn, best_fn, best_precision, best_recall = \
        tps[best_f1_idx], fps[best_f1_idx], tns[best_f1_idx], fns[best_f1_idx], precisions[best_f1_idx], recalls[best_f1_idx]
    

    print("{} {} Epoch {} f1 {} contact Loss {} force loss {} force l1-relative error (pred) {} orce l1-relative error (gt) {} force rmse error gt {} {}".format(
        '=' * 20, 
        mode, epoch, best_f1, total_loss / (i + 1), total_force_loss / (i + 1), 
        force_relative_error_pred / (i + 1),
        force_relative_error_gt / (i + 1),
        force_rmse_error_gt / (i + 1),
        '=' * 20)
    )

    logger.record_tabular("{}/contact loss".format(mode), total_loss / (i + 1))
    logger.record_tabular("{}/force loss".format(mode), total_force_loss / (i + 1))
    logger.record_tabular("{}/force l1 relative error pred".format(mode), force_relative_error_pred / (i + 1))
    logger.record_tabular("{}/force l1 relative error gt".format(mode), force_relative_error_gt / (i + 1))
    logger.record_tabular("{}/force root of mse gt".format(mode), force_rmse_error_gt / (i + 1))
    logger.record_tabular("{}/accuracy".format(mode), correct_nodes / total_nodes)

    logger.record_tabular("{}/true positive".format(mode), best_tp)
    logger.record_tabular("{}/false positive".format(mode), best_fp)
    logger.record_tabular("{}/true negative".format(mode), best_tn)
    logger.record_tabular("{}/false negative".format(mode), best_fn)
    logger.record_tabular("{}/precision".format(mode), best_precision)
    logger.record_tabular("{}/recall".format(mode), best_recall)
    logger.record_tabular("{}/f1".format(mode), best_f1)
    logger.record_tabular("{}/best_threshold".format(mode), best_f1_idx * 0.05)

    logger.record_tabular("{}/contact_forward_time".format(mode), np.mean(contact_forward_time))
    logger.record_tabular("{}/force_forward_time".format(mode), np.mean(force_forward_time))
    logger.dump_tabular()

    return best_f1_idx, best_f1, force_rmse_error_gt

def run_task(vv, log_dir, exp_name):

    args = vv_to_args(vv)

    # configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Configure seed
    seed = vv['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device(args.cuda_idx)

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    checkpoints_dir = osp.join(logger.get_dir(), 'checkpoints')
    experiment_dir = logger.get_dir()
    save_visual_path = os.path.join(experiment_dir, "train_visual")
    if not osp.exists(save_visual_path):
        os.makedirs(save_visual_path, exist_ok=True)
    if not osp.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)

    model = Net(args, 1, feature_num=args.feature_num, num_layer=args.contact_num_layer, sa_radius=args.contact_sa_radius, sa_ratio=args.contact_sa_ratio, 
        residual=args.contact_residual).to(device)
    force_model = Net(
            args, 1, 
            feature_num=args.feature_num,
            num_layer=args.force_num_layer, 
            sa_radius=args.force_sa_radius, 
            sa_ratio=args.force_sa_ratio,
            residual=args.force_residual, 
            sa_mlp_list=args.force_sa_mlp_list,
            fp_mlp_list=args.force_fp_mlp_list,
            linear_mlp_list=args.force_linear_mlp_list,
            fp_k=args.force_fp_k,
         ).to(device)

    if args.load_contact_name is not None:
        checkpoint = torch.load(osp.join(vv['load_contact_name']))
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['contact_model'])
        print("loaded contact model from {}".format(vv['load_contact_name']))
    if args.load_force_name is not None:
        checkpoint = torch.load(osp.join(vv['load_force_name']))
        force_model.load_state_dict(checkpoint['force_model'])
        print("loaded force model from {}".format(vv['load_force_name']))
    else:
        print('No existing model, starting training from scratch...')
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    force_optimizer = torch.optim.Adam(force_model.parameters(), lr=args.lr)

    contact_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)
    force_scheduler = ReduceLROnPlateau(force_optimizer, 'min', factor=0.8, patience=3, verbose=True)

    root = './data/haptic-perspective/{}'.format(vv['data_dir'])
    train_dataset = HapticDataset(args, split='train', data_root=root, 
        add_gripper_vel=args.add_gripper_vel,
        force_normalize_mean=args.force_normalize_mean, force_normalize_std=args.force_normalize_std, 
        add_cloth_vel=args.add_cloth_vel, remove_normalized_coordinate=args.remove_normalized_coordinate
    )
    test_dataset = HapticDataset(args, split='valid', data_root=root, 
        add_gripper_vel=args.add_gripper_vel,
        force_normalize_mean=args.force_normalize_mean, force_normalize_std=args.force_normalize_std, 
        add_cloth_vel=args.add_cloth_vel, remove_normalized_coordinate=args.remove_normalized_coordinate
    )
    train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
    test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)

    train_pos_label_weight = args.train_pos_label_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_pos_label_weight])).to(device)
    force_criterion = weighted_mse()

    if not args.only_eval:
        best_eval_f1 = 0
        best_eval_force_rmse = np.inf
        for epoch in range(args.epoch):

            train_metrics = run(args, epoch, (model, force_model), train_loader, device, 
                (optimizer, force_optimizer), 
                (contact_scheduler, force_scheduler),
                (criterion, force_criterion), 
                mode='train',
                save_visual_path=save_visual_path)
            
            eval_metrics = run(args, epoch, (model, force_model), test_loader, device, 
                (None, None), 
                (None, None),
                (criterion, force_criterion), mode='eval',
                save_visual_path=save_visual_path)
            
            eval_f1_idx, eval_f1, eval_force_rmse = eval_metrics
            better_contact_model = False
            better_force_model = False
            if eval_f1 > best_eval_f1:
                best_eval_f1 = eval_f1
                better_contact_model = True
            if eval_force_rmse < best_eval_force_rmse:
                best_eval_force_rmse = eval_force_rmse
                better_force_model = True
            
            if better_contact_model or better_force_model or (epoch + 1) % args.save_interval == 0:
                save_dict = {
                    'contact_model': model.state_dict(),
                    'force_model': force_model.state_dict(),
                    'contact_optimizer': optimizer.state_dict(),
                    'force_optimizer': force_optimizer.state_dict(),
                    'eval_f1_idx': eval_f1_idx
                }

                if better_contact_model:
                    savepath = osp.join(checkpoints_dir, 'best_contact_model.pth')
                elif better_force_model:
                    savepath = osp.join(checkpoints_dir, 'best_force_model.pth')
                else:
                    savepath = osp.join(checkpoints_dir, 'model_{}.pth'.format(epoch))

                torch.save(save_dict, savepath)
    else:
        if args.split == 'eval':
            loader = test_loader
        else:
            loader = train_loader
        eval_metrics = run(args, 0, (model, force_model), loader, device, 
                (None, None), 
                (None, None),
                (criterion, force_criterion), mode='eval',
                save_visual_path=save_visual_path)

    # iou = test(test_loader)
    # print('Epoch: {:02d}, Test IoU: {:.4f}'.format(epoch, iou))
