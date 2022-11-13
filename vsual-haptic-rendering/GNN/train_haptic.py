"""
Author: Benny
Date: Nov 2019
"""
import os
import os.path as osp
import time
import numpy as np
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric
from torch_geometric.data import Data

from chester import logger
from haptic.shared.haptic_dataset import GNNHapticDataset as HapticDataset
from haptic.GNN.models_graph_res import GNNModel
from haptic.util import get_force_weight, weighted_mse, vv_to_args
from haptic.visual import do_plot, plot_traj, plot_precision_recall_curve

def run(args, epoch, dataloader, models, optimizers, schedulers, criterions, mode, device, save_visual_path=None):
    contact_optimizer, force_optimizer = optimizers
    contact_model, force_model = models
    contact_criterion, force_criterion = criterions
    contact_scheduler, force_scheduler = schedulers

    if mode == 'train':
        contact_model.train()
        force_model.train()
    elif mode == 'eval':
        contact_model.eval()
        force_model.eval()

    total_contact_loss = correct_nodes = total_nodes = 0
    total_force_loss = 0
    force_relative_error_pred = 0
    force_relative_error_gt = 0
    force_rmse_error_gt = 0

    all_contact_pred = []
    all_contact_label = []

    contact_forward_time = []
    force_forward_time = []

    with torch.set_grad_enabled(mode == 'train'):
        # time_beg = time.time()
        for i, data in enumerate(dataloader):
            # if i % 20 == 0:
            #     print("Training Epoch {} {}/{} time {}".format(epoch, i, len(dataloader), time.time() - time_beg), flush=True)
            #     time_beg = time.time()
            # if args.only_eval:
            #     if i > len(dataloader) / 2:
            #         break

            if mode == 'train':
                if args.train_contact:
                    contact_optimizer.zero_grad()
                if args.train_force:
                    force_optimizer.zero_grad()

            data = data.to(device)

            contact_target, force_target = data.contact_label, data.force_label
            
            beg = time.time()
            contact_pred = contact_model(data)
            contact_forward_time.append(time.time() - beg)
            beg = time.time()
            force_pred = force_model(data)
            force_forward_time.append(time.time() - beg)
            force_pred = force_pred.contiguous().view(-1, 1)

            # contact loss
            contact_target = contact_target.view(-1, 1)
            contact_loss = torch.Tensor([0])
            if args.train_contact:
                contact_loss = contact_criterion(contact_pred, contact_target)

            # force loss
            force_target = force_target.view(-1, 1)
            force_loss = torch.Tensor([0])
            if args.train_force:
                weight = get_force_weight(force_target, args.force_loss_mode, args.train_pos_label_weight)
                force_loss = force_criterion(force_pred, force_target, weight)

            if mode == 'train':
                if args.train_contact:
                    contact_loss.backward()
                    contact_optimizer.step()
                if args.train_force:
                    force_loss.backward()
                    force_optimizer.step()

            total_contact_loss += contact_loss.item()
            total_force_loss += force_loss.item()

            contact_pred = contact_pred.view(-1, 1)
            correct_nodes += (contact_pred > 0).eq(contact_target).sum().item()
            total_nodes += data['x'].shape[0]

            contact_label_numpy = contact_target.data.cpu().numpy()
            contact_pred_numpy = contact_pred.data.cpu().numpy()
            all_contact_label.append(contact_label_numpy)
            all_contact_pred.append(contact_pred_numpy)

            contact_pred_numpy = contact_pred_numpy.flatten()
            contact_label_numpy = contact_label_numpy.flatten()
            force_pred_numpy = force_pred.data.cpu().numpy().flatten()[contact_pred_numpy > 0]
            force_target_numpy = force_target.data.cpu().numpy().flatten()[contact_pred_numpy > 0]
            if args.force_normalize_mean is not None:
                force_pred_numpy = force_pred_numpy * args.force_normalize_std + args.force_normalize_mean
                force_target_numpy = force_target_numpy * args.force_normalize_std + args.force_normalize_mean
            force_relative_error_pred += np.mean(np.abs(force_pred_numpy - force_target_numpy) / (np.abs(force_target_numpy) + 1e-10))
            
            force_pred_numpy = force_pred.data.cpu().numpy().flatten()[contact_label_numpy == 1]
            force_target_numpy = force_target.data.cpu().numpy().flatten()[contact_label_numpy == 1]
            if args.force_normalize_mean is not None:
                force_pred_numpy = force_pred_numpy * args.force_normalize_std + args.force_normalize_mean
                force_target_numpy = force_target_numpy * args.force_normalize_std + args.force_normalize_mean
            force_relative_error_gt += np.mean(np.abs(force_pred_numpy - force_target_numpy) / (np.abs(force_target_numpy) + 1e-10))
            force_rmse_error_gt += np.sqrt(np.mean((force_pred_numpy - force_target_numpy) ** 2))

            if ((epoch + 1) % args.plot_img_interval == 0 or epoch == args.epoch - 1):
                if i <= 10:
                    do_plot(contact_pred, force_pred, data, epoch, i, mode, save_visual_path)

        # print("schedule learning rate!")

        if mode == 'train':
            if args.train_contact:
                contact_scheduler.step(total_contact_loss / len(dataloader))
            if args.train_force:
                force_scheduler.step(total_force_loss / len(dataloader))

    if ((epoch + 1) % args.plot_video_interval == 0 or epoch == args.epoch - 1):
        plot_traj(args, contact_model, force_model, dataloader, epoch, mode, save_visual_path, device, input_format='gnn')

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
    
    print("{} {} Epoch {} f1 {} contact Loss {} force loss {} force l1-relative error (pred) {} force l1-relative error (gt) {} force rmse error gt {} {}".format(
        '=' * 20, 
        mode, epoch, best_f1, total_contact_loss / (i + 1), total_force_loss / (i + 1), 
        force_relative_error_pred / (i + 1),
        force_relative_error_gt / (i + 1),
        force_rmse_error_gt / (i + 1),
        '=' * 20)
    )

    logger.record_tabular("{}/contact loss".format(mode), total_contact_loss / (i + 1))
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

    '''HYPER PARAMETER'''
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

    root = './data/haptic-perspective/{}'.format(vv['data_dir'])
    print("start loading training data ...", flush=True)
    TRAIN_DATASET = HapticDataset(args, split='train', data_root=root, traj_num=args.train_traj_num, 
        add_gripper_vel=args.add_gripper_vel,
        force_normalize_mean=args.force_normalize_mean, force_normalize_std=args.force_normalize_std, 
        add_cloth_vel=args.add_cloth_vel, remove_normalized_coordinate=args.remove_normalized_coordinate
    )
    print("start loading test data ...", flush=True)
    TEST_DATASET = HapticDataset(args, split='valid', data_root=root, traj_num=args.valid_traj_num,
        add_gripper_vel=args.add_gripper_vel,
        force_normalize_mean=args.force_normalize_mean, force_normalize_std=args.force_normalize_std, 
        add_cloth_vel=args.add_cloth_vel, remove_normalized_coordinate=args.remove_normalized_coordinate
    )
    if args.train_pos_label_weight == -1:
        train_pos_label_weight, _ = TRAIN_DATASET.get_dataset_statistics()
    else:
        train_pos_label_weight = args.train_pos_label_weight

    trainDataLoader = torch_geometric.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                                  pin_memory=True, drop_last=True)
    testDataLoader = torch_geometric.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                                 pin_memory=True, drop_last=True)

    print("The number of training data is: %d" % len(TRAIN_DATASET))
    print("The number of test data is: %d" % len(TEST_DATASET))

    contact_model = GNNModel(args, args.contact_proc_layer, args.global_size, 1)
    force_model = GNNModel(args, args.force_proc_layer, args.global_size, 1)
    contact_model.to(device)
    force_model.to(device)

    if args.contact_loss_balance:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_pos_label_weight])).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss()
    force_criterion = weighted_mse()

    if args.load_contact_name is not None:
        checkpoint = torch.load(osp.join(vv['load_contact_name']))
        contact_model.load_state_dict(checkpoint['contact_model'])
        print("loaded contact model from {}".format(vv['load_contact_name']))
    if args.load_force_name is not None:
        checkpoint = torch.load(osp.join(vv['load_force_name']))
        force_model.load_state_dict(checkpoint['force_model'])
        print("loaded force model from {}".format(vv['load_force_name']))
    else:
        print('No existing model, starting training from scratch...')

    optimizers = []
    for model in [contact_model, force_model]:
        optimizers.append(torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
        ))
    contact_optimizer, force_optimizer = optimizers
    
    # schedulers = []
    contact_scheduler = ReduceLROnPlateau(contact_optimizer, 'min', factor=0.8, patience=3, verbose=True)
    force_scheduler = ReduceLROnPlateau(force_optimizer, 'min', factor=0.8, patience=3, verbose=True)


    if not args.only_eval:
        best_eval_f1 = 0
        best_eval_force_rmse = np.inf
        for epoch in range(args.epoch):
            # def run(args, epoch, dataloader, models, optimizers, criterions, mode, device, save_visual_path=None):

            train_metrics = run(args, epoch, trainDataLoader, 
                (contact_model, force_model), 
                (contact_optimizer, force_optimizer), 
                (contact_scheduler, force_scheduler),
                (criterion, force_criterion),
                'train', device, save_visual_path)
            
            torch.cuda.empty_cache()
            
            eval_metrics = run(args, epoch, testDataLoader, 
                (contact_model, force_model),  
                (None, None), 
                (None, None),
                (criterion, force_criterion),
                'eval', device, save_visual_path)
            
            torch.cuda.empty_cache()
            
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
                    'contact_model': contact_model.state_dict(),
                    'force_model': force_model.state_dict(),
                    'contact_optimizer': contact_optimizer.state_dict(),
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
            dataLoader = testDataLoader
        else:
            dataLoader = trainDataLoader
        eval_metrics = run(args, 0, dataLoader, 
                (contact_model, force_model),  
                (None, None), 
                (None, None),
                (criterion, force_criterion),
                'eval', device, save_visual_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)
