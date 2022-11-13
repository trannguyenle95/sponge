"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from typing_extensions import Concatenate
from haptic.Pointnet_Pointnet2_pytorch.data_utils.HapticDataLoader import HapticDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import haptic.Pointnet_Pointnet2_pytorch.provider as provider
import numpy as np
import time
import torch.nn as nn
from haptic.visual import plot, plot_precision_recall_curve
import imageio
import os.path as osp
from chester import logger
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt
from haptic.Pointnet_Pointnet2_pytorch.utils import print_cuda
import time


def collate_wrapper(batch):
    # print("batch is: ", batch)
    # print("len of batch is: ", len(batch))
    # for x in batch:
    #     print(x)
    bs = len(batch)
    
    current_points = [b[0] for b in batch]
    dim_point = current_points[0].shape[0]
    current_points = np.vstack(current_points).reshape((bs, dim_point, -1))
    current_points = torch.FloatTensor(current_points)

    contact_labels = [b[1] for b in batch]
    contact_labels = np.vstack(contact_labels).reshape((bs, dim_point, -1))
    contact_labels = torch.FloatTensor(contact_labels)

    force_labels = [b[2] for b in batch]
    force_labels = np.vstack(force_labels).reshape((bs, dim_point, -1))
    force_labels = torch.FloatTensor(force_labels)

    ori_xyz = [b[3] for b in batch]
    ori_xyz = np.vstack(ori_xyz).reshape((bs, dim_point, -1))

    return current_points, (contact_labels, force_labels), ori_xyz

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

class weighted_mse(nn.Module):
    def __init__(self):
        super(weighted_mse, self).__init__()

    def forward(self, pred, target, weight):
        total_loss = torch.mean((weight * (pred - target)) ** 2)

        return total_loss

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_msg_haptic', help='model name [default: pointnet_sem_seg_msg]')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
    parser.add_argument('--plot_interval', type=int, default=10, help='Interval for dumping gifs [default: 10]')
    parser.add_argument('--epoch', default=200, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    # parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

    return parser.parse_args([])

def run_task(vv, log_dir, exp_name):
    args = parse_args()
    if not vv['train']:
        vv['epoch'] = 1
    args.__dict__.update(**vv)

    '''HYPER PARAMETER'''
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Configure torch
    seed = vv['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

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

    '''LOG'''
    root = './data/haptic-perspective/{}'.format(vv['data_dir'])
    NUM_CLASSES = 1
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = HapticDataset(split='train', data_root=root, num_point=args.npoint, traj_num=args.train_traj_num)
    print("start loading test data ...")
    TEST_DATASET = HapticDataset(split='valid', data_root=root, num_point=args.npoint, traj_num=args.valid_traj_num)
    if args.train_pos_label_weight == -1:
        train_num_point, train_pos_label_weight, train_data_idx = TRAIN_DATASET.get_dataset_statistics()
    else:
        train_pos_label_weight = args.train_pos_label_weight

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=vv['num_worker'],
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())),
                                                  collate_fn=collate_wrapper)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=True, num_workers=vv['num_worker'],
                                                 pin_memory=True, drop_last=True)
    # NOTE no weight is used for our task
    # weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    weights = None

    print("The number of training data is: %d" % len(TRAIN_DATASET))
    # print("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    # MODEL = importlib.import_module(args.model)
    import haptic.Pointnet_Pointnet2_pytorch.models.pointnet2_sem_seg_msg_haptic as MODEL
    shutil.copy('haptic/Pointnet_Pointnet2_pytorch/models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('haptic/Pointnet_Pointnet2_pytorch/models/pointnet2_utils.py', str(experiment_dir))

    if not vv['separate_model']:
        classifier = MODEL.get_shared_model(vv['use_batch_norm'], NUM_CLASSES).cuda()
    else:
        print("separately build model!", flush=True)
        contact_classifier = MODEL.get_model(vv['use_batch_norm'], 1, target='contact', 
            radius_list=vv['radius_list'], npoint_list=vv['npoint_list'], sample_point_1_list=vv['sample_point_1_list'],
            layer=vv['layer'], dropout=vv['dropout'], downsample=vv['downsample'], track_running_stats=vv['track_running_stats'],
            mlp1_size=vv['mlp1_size'], mlp2_size=vv['mlp2_size'], interpolation_mlp_size=vv['interpolation_mlp_size']
            ).cuda()
        force_classifier = MODEL.get_model(vv['use_batch_norm'], 1, target='force', 
            radius_list=vv['radius_list'], npoint_list=vv['npoint_list'], sample_point_1_list=vv['sample_point_1_list'],
            layer=vv['layer'], dropout=vv['dropout'], downsample=vv['downsample'], track_running_stats=vv['track_running_stats'],
            mlp1_size=vv['mlp1_size'], mlp2_size=vv['mlp2_size'], interpolation_mlp_size=vv['interpolation_mlp_size']
            ).cuda()
        normal_classifier = None
            # MODEL.get_model(vv['use_batch_norm'], 3, target='normal', 
            # radius_list=vv['radius_list'], npoint_list=vv['npoint_list'], sample_point_1_list=vv['sample_point_1_list'],
            # layer=vv['layer'], dropout=vv['dropout'], downsample=vv['downsample'], track_running_stats=vv['track_running_stats']).cuda()
        # all_classifiers = [contact_classifier, force_classifier, normal_classifier]
        all_classifiers = [contact_classifier, force_classifier]

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_pos_label_weight])).cuda()
    force_criterion = weighted_mse()
    normal_criterion = weighted_mse()

    if not vv['separate_model']:
        classifier.apply(inplace_relu)
    else:
        print("separately apply inplace_relu!", flush=True)
        # for model in all_classifiers:
        #     model.apply(inplace_relu)
        contact_classifier.apply(inplace_relu)
        force_classifier.apply(inplace_relu)
        # normal_classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        if not vv['separate_model']:
            checkpoint = torch.load(osp.join('best_model.pth'))
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            print('Use pretrain model')
        else:
            checkpoint = torch.load(osp.join(vv['load_dir'], 'checkpoints', 'model.pth'))
            contact_classifier.load_state_dict(checkpoint['contact_model_state_dict'])
            print("loaded model from {}".format(vv['load_dir']))
            start_epoch = 0
    except:
        print('No existing model, starting training from scratch...')
        start_epoch = 0
        if not vv['separate_model']:
            classifier = classifier.apply(weights_init)
        else:
            print("separately weight init!", flush=True)
            # for model in all_classifiers:
                # model = model.apply(weights_init)
            # contact_classifier, force_classifier, normal_classifier = all_classifiers
            contact_classifier = contact_classifier.apply(weights_init)
            force_classifier = force_classifier.apply(weights_init)
            # normal_classifier = normal_classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        if not vv['separate_model']:
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.decay_rate
            )
        else:
            print("separately build optimizers!", flush=True)
            optimizers = []
            for model in all_classifiers:
                optimizers.append(torch.optim.Adam(
                    model.parameters(),
                    lr=args.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=args.decay_rate
                ))
            # contact_optimizer, force_optimizer, normal_optimizer = optimizers
            contact_optimizer, force_optimizer = optimizers
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)
    
    if not vv['separate_model']:
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)
    else:
        print("separately build schedulers!", flush=True)
        schedulers = []
        for optim in optimizers:
            schedulers.append(ReduceLROnPlateau(optim, 'min', factor=0.8, patience=3, verbose=True))
        # contact_scheduler, force_scheduler, normal_scheduler = schedulers
        contact_scheduler, force_scheduler = schedulers

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    def print_momentum(model):
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                print(name, layer)
                print(layer.running_mean)
                print(layer.running_var, flush=True)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_acc = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        print('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        
        # NOTE: use a scheduler to schdule the learning rate
        if vv['manual_lr_adjust']:
            lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
            print('Learning rate:%f' % lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)

        if vv['adjust_momentum']:
            if not vv['separate_model']:
                classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
            else:
                # for model in all_classifiers:
                    # model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
                # contact_classifier, force_classifier, normal_classifier = all_classifiers
                contact_classifier = contact_classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
                force_classifier = force_classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
                # normal_classifier = normal_classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        # print_momentum(contact_classifier)

        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        contact_loss_sum = 0
        force_loss_sum = 0
        # normal_loss_sum = 0
        if not vv['separate_model']:
            classifier = classifier.train()
        else:
            # for model in all_classifiers:
            #     model = model.train()
            contact_classifier.train()
            force_classifier.train()
            # normal_classifier.train() 
            # force_classifier, normal_classifier = all_classifiers

        all_contact_pred = []
        all_batch_label = []
        force_relative_errors = []
        # print("right before dataloading, cuda usage")
        # print_cuda()

        if vv['train']:
            for i, (points, target, ori_xyz) in enumerate(trainDataLoader):
                # print("points.dtype: ", points.dtype)
                # print("target[0].dtype: ", target[0].dtype)
                # print("ori_xyz.dtype: ", ori_xyz.dtype)

                # print("=" * 100)
                # print("=" * 100)
                # print("after dataloading but before forward pass, cuda usage")
                # print_cuda()

                # if i % 5000 == 0:
                #     print("epoch {} i {}/{}".format(epoch, i, len(trainDataLoader)))
                print(i)
                batch_time_beg = time.time()

                if vv['debug']:
                    if i >= 100:
                        break
                
                if not vv['separate_model']:
                    optimizer.zero_grad()
                else:
                    for optimizer in optimizers:
                        optimizer.zero_grad()

                # contact_target, force_target, normal_target = target
                contact_target, force_target = target
                # points = points.data.numpy()
                # if vv['correct_z_rotation'] == 0: # was false
                #     points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                # elif vv['correct_z_rotation'] == 1: # was true
                #     points[:, :, :3] = provider.rotate_point_cloud_y(points[:, :, :3])
                # elif vv['correct_z_rotation'] == 2: # no augmentation
                #     pass
                # points = torch.Tensor(points)

                points = points.cuda()
                points = points.transpose(2, 1)
                
                contact_target = contact_target.cuda()
                force_target = force_target.cuda()
                # print("points shape: ", points.shape)
                # print("contact_target shape: ", contact_target.shape)
                # print("force_target shape: ", force_target.shape)
                # normal_target = normal_target.float().cuda()

                forward_time_beg = time.time()
                if not vv['separate_model']:
                    pred, _ = classifier(points)
                    contact_pred, normal_pred, force_pred = pred
                else:
                    # print("right before forward pass, cuda usage")
                    # print_cuda()
                    contact_pred, _ = contact_classifier(points)
                    force_pred, _ = force_classifier(points)
                    # print("right after forward pass, cuda usage")
                    # print_cuda()
                    # normal_pred, _ = normal_classifier(points)
                print("forward using time: ", time.time() - forward_time_beg)
                    
                compute_loss_time_beg = time.time()
                contact_pred = contact_pred.view(-1, NUM_CLASSES)
                force_pred = force_pred.view(-1, 1)
                # normal_pred = normal_pred.contiguous().view(-1, 3)

                # contact loss
                contact_target = contact_target.view(-1, 1)
                contact_loss = criterion(contact_pred, contact_target)

                # force loss
                contact_filter = contact_target.view(-1, 1)[:, 0].cpu().data.numpy()
                force_target = force_target.view(-1, 1)
                if vv['force_loss_mode'] == 'contact':
                    weight = torch.zeros_like(contact_target)
                    weight[contact_filter] = 1
                elif vv['force_loss_mode'] == 'all':
                    weight = torch.ones_like(contact_target)
                elif vv['force_loss_mode'] == 'balance':
                    weight = torch.ones_like(contact_target)
                    weight[contact_filter] = train_pos_label_weight
                force_loss = force_criterion(force_pred, force_target, weight)

                force_pred_numpy = force_pred.data.cpu().numpy().flatten()
                force_target_numpy = force_target.data.cpu().numpy().flatten()
                force_relative_errors.append(np.abs(force_pred_numpy - force_target_numpy) / np.abs(force_target_numpy)  + 1e-10)

                # normal loss
                # normal_target = normal_target.view(-1, 3)
                # if vv['normal_loss_mode'] == 'contact':
                #     weight = torch.zeros_like(contact_target)
                #     weight[contact_filter] = 1
                # elif vv['normal_loss_mode'] == 'all':
                #     weight = torch.ones_like(contact_target)
                # elif vv['normal_loss_mode'] == 'balance':
                #     weight = torch.ones_like(contact_target)
                #     weight[contact_filter] = train_pos_label_weight
                # normal_loss = normal_criterion(normal_pred, normal_target, weight)

                loss = contact_loss + force_loss * vv['force_loss_weight'] #+ normal_loss * vv['normal_loss_weight']
                print("compute loss time: ", time.time() - compute_loss_time_beg)

                optim_time_beg = time.time()
                loss.backward()
                if not vv['separate_model']:
                    optimizer.step()
                else:
                    for optimizer in optimizers:
                        optimizer.step()
                print("optim cost time: ", time.time() - optim_time_beg)


                stats_time_beg = time.time()
                pred_choice = (contact_pred.cpu().data.numpy().reshape(-1, 1) > 0).astype(np.float)
                pred_choice = pred_choice.flatten()
                contact_batch_label = contact_target[:, 0].cpu().data.numpy()
                correct = np.sum(pred_choice == contact_batch_label)
                total_correct += correct
                total_seen += points.shape[0] * points.shape[2]
                all_contact_pred.append(contact_pred.cpu().data.numpy().reshape(-1, 1))
                all_batch_label.append(contact_batch_label.reshape(-1, 1))

                loss_sum += loss.item()
                contact_loss_sum += contact_loss.item()
                force_loss_sum += force_loss.item()
                print("stats cost time: ", time.time() - stats_time_beg)

                print("batch total time: ", time.time() - batch_time_beg)
                # normal_loss_sum += normal_loss.item()

                # if (epoch % args.plot_interval == 0 or epoch == args.epoch - 1) and args.train:
                #     print("save plot at: ", save_visual_path, flush=True)
                #     plot(ori_xyz, pred_choice, contact_batch_label, 
                #             force_pred_numpy, force_target_numpy, 
                #             os.path.join(save_visual_path, "train-{}-{}.png".format(epoch, i)))

            # if (epoch % args.plot_interval == 0 or epoch == args.epoch - 1) and args.train:
            #     print("save gif at: ", save_visual_path, flush=True)
            #     images = []
            #     with imageio.get_writer(osp.join(save_visual_path, 'train-pred-visual-{}.gif'.format(epoch)), mode='I') as writer:
            #         for _ in range(len(trainDataLoader)):
            #             filename = os.path.join(save_visual_path, "train-{}-{}.png".format(epoch, _))
            #             image = imageio.imread(filename)
            #             writer.append_data(image)

            if vv['schedule_lr']:
                print("schedule learning rate!")
                if not vv['separate_model']:
                    scheduler.step(loss_sum / num_batches)
                else:
                    contact_scheduler.step(contact_loss_sum / num_batches)
                    force_scheduler.step(force_loss_sum / num_batches)
                    # normal_scheduler.step(normal_loss_sum / num_batches)

            all_contact_pred = np.concatenate(all_contact_pred, axis=0).flatten()
            all_batch_label = np.concatenate(all_batch_label, axis=0).flatten()
            if epoch % vv['plot_interval'] == 0:
                save_name = osp.join(save_visual_path, 'precision-recall-train-{}.png'.format(epoch)) 
            else:
                save_name = None
            tp, fp, tn, fn, precision, recall, f1 = \
                plot_precision_recall_curve(all_contact_pred, all_batch_label, save_name = save_name)

            force_relative_error = np.mean(np.hstack(force_relative_errors)) * 100

            print('Training mean loss: %f' % (loss_sum / num_batches))
            print('Training mean contact loss: %f' % (contact_loss_sum / float(num_batches)))
            print('Training mean force loss: %f' % (force_loss_sum / float(num_batches)))
            print('Training mean force relative error: %f' % force_relative_error)
            # print('Training mean normal loss: %f' % (normal_loss_sum / float(num_batches)))
            print('Training accuracy: %f' % (total_correct / float(total_seen)))

            print('Training all true positive: %f' % tp)
            print('Training all false positive: %f' % fp)
            print('Training all true negative: %f' % tn)
            print('Training all false negative: %f' % fn)
            print('Training all precision: %f' % precision)
            print('Training all recall: %f' % recall)
            print('Training all f1: %f' % f1)

            logger.record_tabular("Train/mean loss", loss_sum / num_batches)
            logger.record_tabular("Train/accuracy", total_correct / float(total_seen))
            logger.record_tabular("Train/mean contact loss", contact_loss_sum / num_batches)
            logger.record_tabular("Train/mean force loss", force_loss_sum / num_batches)
            logger.record_tabular("Train/mean force relative error", force_relative_error)
            # logger.record_tabular("Train/mean normal loss", normal_loss_sum / num_batches)

            logger.record_tabular("Train/true positive", tp)
            logger.record_tabular("Train/false positive", fp)
            logger.record_tabular("Train/true negative", tn)
            logger.record_tabular("Train/false negative", fn)
            logger.record_tabular("Train/precision", precision)
            logger.record_tabular("Train/recall", recall)
            logger.record_tabular("Train/f1", f1)

            if epoch % 5 == 0:
                logger.info('Save model...')
                savepath = osp.join(checkpoints_dir, 'model.pth')
                print('Saving at %s' % savepath)
                if not vv['separate_model']:
                    state = {
                        'epoch': epoch,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                else:
                    state = {
                        'epoch': epoch,
                        'contact_model_state_dict': contact_classifier.state_dict(),
                        'force_model_state_dict': force_classifier.state_dict(),
                        # 'normal_model_state_dict': normal_classifier.state_dict(),
                        'contact_optimizer_state_dict': contact_optimizer.state_dict(),
                        'force_optimizer_state_dict': force_optimizer.state_dict(),
                        # 'normal_optimizer_state_dict': normal_optimizer.state_dict(),
                    }
                torch.save(state, savepath)
                print('Saving model....')



        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            contact_loss_sum = 0
            force_loss_sum = 0
            # normal_loss_sum = 0
            force_relative_errors = []
            if not vv['separate_model']:
                classifier = classifier.eval()
            else:
                if vv['set_eval_for_batch_norm']:
                    contact_classifier.eval()
                    force_classifier.eval()
                    # normal_classifier.eval()
                else:
                    pass

            print('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            all_contact_pred = []
            all_batch_label = []
            for i, (points, target, ori_xyz) in enumerate(testDataLoader):
                if i >= 200: # TODO: solve this
                    break

                # points = points.data.numpy()
                # points = torch.Tensor(points)
                points = points.cuda() 
                points = points.transpose(2, 1)

                # contact_target, force_target, normal_target = target
                contact_target, force_target = target
                contact_target = contact_target.cuda()
                force_target = force_target.cuda()
                # normal_target = normal_target.float().cuda()

                if not vv['separate_model']:
                    pred, _ = classifier(points)
                    contact_pred, normal_pred, force_pred = pred
                else:
                    contact_pred, _ = contact_classifier(points)
                    force_pred, _ = force_classifier(points)
                    # normal_pred, _ = normal_classifier(points)

                contact_pred = contact_pred.view(-1, NUM_CLASSES)
                force_pred = force_pred.view(-1, 1)
                # normal_pred = normal_pred.contiguous().view(-1, 3)

                # contact loss
                contact_target = contact_target.view(-1, 1)
                contact_loss = criterion(contact_pred, contact_target)

                # force loss
                contact_filter = contact_target.cpu().data.numpy().flatten()
                force_target = force_target.view(-1, 1)
                if vv['force_loss_mode'] == 'contact':
                    weight = torch.zeros_like(contact_target)
                    weight[contact_filter] = 1
                elif vv['force_loss_mode'] == 'all':
                    weight = torch.ones_like(contact_target)
                elif vv['force_loss_mode'] == 'balance':
                    weight = torch.ones_like(contact_target)
                    weight[contact_filter] = train_pos_label_weight
                force_loss = force_criterion(force_pred, force_target, weight)

                force_pred_numpy = force_pred.cpu().data.numpy()
                force_target_numpy = force_target.cpu().data.numpy()
                force_relative_errors.append(np.abs(force_target_numpy.flatten() - force_pred_numpy.flatten()) \
                     / np.abs(force_target_numpy.flatten()) + 1e-10)

                # normal loss
                # normal_target = normal_target.view(-1, 3)
                # if vv['normal_loss_mode'] == 'contact':
                #     weight = torch.zeros_like(contact_target)
                #     weight[contact_filter] = 1
                # elif vv['normal_loss_mode'] == 'all':
                #     weight = torch.ones_like(contact_target)
                # elif vv['normal_loss_mode'] == 'balance':
                #     weight = torch.ones_like(contact_target)
                #     weight[contact_filter] = train_pos_label_weight
                # normal_loss = normal_criterion(normal_pred, normal_target, weight)

                loss = contact_loss + force_loss * vv['force_loss_weight']# + normal_loss * vv['normal_loss_weight']

                loss_sum += loss.item()
                contact_loss_sum += contact_loss.item()
                force_loss_sum += force_loss.item()
                # normal_loss_sum += normal_loss.item()

                contact_pred_val = ((contact_pred.cpu().data.numpy().reshape(-1, 1) > 0).astype(np.float)).flatten()
                batch_label = contact_target.cpu().data.numpy().flatten()
                all_contact_pred.append(contact_pred.cpu().data.numpy().reshape(-1, 1))
                all_batch_label.append(batch_label.reshape(-1, 1))
                correct = np.sum((contact_pred_val == batch_label))
                total_correct += correct
                total_seen += points.shape[0] * points.shape[2]

                if i <= 25:
                    if (epoch % args.plot_interval == 0 or epoch == args.epoch - 1) and args.train:
                        print("save plot at: ", save_visual_path, flush=True)
                        plot(ori_xyz, contact_pred_val, batch_label, 
                                force_pred_numpy, force_target_numpy, 
                                os.path.join(save_visual_path, "eval-{}-{}.png".format(epoch, i)))

            all_contact_pred = np.concatenate(all_contact_pred, axis=0).flatten()
            all_batch_label = np.concatenate(all_batch_label, axis=0).flatten()
            if epoch % vv['plot_interval'] == 0:
                save_name = osp.join(save_visual_path, 'precision-recall-eval-{}.png'.format(epoch)) 
            else:
                save_name = None

            tp, fp, tn, fn, precision, recall, f1 = \
                plot_precision_recall_curve(all_contact_pred, all_batch_label, 
                    save_name=save_name)

            force_relative_error = np.mean(np.hstack(force_relative_errors)) * 100

            # if (epoch % args.plot_interval == 0 or epoch == args.epoch - 1) and args.train:
            #     print("save gif at: ", save_visual_path, flush=True)
            #     with imageio.get_writer(osp.join(save_visual_path, 'eval-pred-visual-{}.gif'.format(epoch)), mode='I') as writer:
            #         for _ in range(len(testDataLoader)):
            #             filename = os.path.join(save_visual_path, "eval-{}-{}.png".format(epoch, _))
            #             image = imageio.imread(filename)
            #             writer.append_data(image)

            eval_acc = (total_correct / float(total_seen))
            print('eval mean loss: %f' % (loss_sum / float(num_batches)))
            print('eval mean contact loss: %f' % (contact_loss_sum / float(num_batches)))
            print('eval mean force loss: %f' % (force_loss_sum / float(num_batches)))
            print('eval mean force relative error: %f' % force_relative_error)
            # print('eval mean normal loss: %f' % (normal_loss_sum / float(num_batches)))
            print('eval all accuracy: %f' % eval_acc)
            
            print('eval all true positive: %f' % tp)
            print('eval all false positive: %f' % fp)
            print('eval all true negative: %f' % tn)
            print('eval all false negative: %f' % fn)
            print('eval all precision: %f' % precision)
            print('eval all recall: %f' % recall)
            print('eval all f1: %f' % f1)

            logger.record_tabular("Eval/mean loss", loss_sum / float(num_batches))
            logger.record_tabular("Eval/mean contact loss", contact_loss_sum / float(num_batches))
            logger.record_tabular("Eval/mean force loss", force_loss_sum / float(num_batches))
            logger.record_tabular("Eval/mean force relative error", force_relative_error)
            # logger.record_tabular("Eval/mean normal loss", normal_loss_sum / float(num_batches))
            logger.record_tabular("Eval/accuracy", eval_acc)

            logger.record_tabular("Eval/true positive", tp)
            logger.record_tabular("Eval/false positive", fp)
            logger.record_tabular("Eval/true negative", tn)
            logger.record_tabular("Eval/false negative", fn)
            logger.record_tabular("Eval/precision", precision)
            logger.record_tabular("Eval/recall", recall)
            logger.record_tabular("Eval/f1", f1)

            logger.dump_tabular()

            if eval_acc >= best_acc:
                best_acc = eval_acc
                logger.info('Save model...')
                savepath = osp.join(checkpoints_dir, 'best_model.pth')
                print('Saving at %s' % savepath)
                if not vv['separate_model']:
                    state = {
                        'epoch': epoch,
                        'class_acc': eval_acc,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                else:
                    state = {
                        'epoch': epoch,
                        'class_acc': eval_acc,
                        'contact_model_state_dict': contact_classifier.state_dict(),
                        'force_model_state_dict': force_classifier.state_dict(),
                        # 'normal_model_state_dict': normal_classifier.state_dict(),
                        'contact_optimizer_state_dict': contact_optimizer.state_dict(),
                        'force_optimizer_state_dict': force_optimizer.state_dict(),
                        # 'normal_optimizer_state_dict': normal_optimizer.state_dict(),
                    }
                torch.save(state, savepath)
                print('Saving model....')
            
            print('Best acc: %f' % best_acc)

        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
