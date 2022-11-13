"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import os.path as osp
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import matplotlib.cm as cm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_msg', help='model name [default: pointnet_sem_seg_msg]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')

    return parser.parse_args()
    
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = '../data/haptic-perspective/2021-10-12'
    NUM_CLASSES = 1

    print("start loading training data ...")
    TEST_DATASET = HapticDataset(split='train', data_root=root, num_point=1, block_size=1.0, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=True, num_workers=5,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))

    log_string("The number of training data is: %d" % len(TEST_DATASET))
    # log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.train()
    criterion = MODEL.get_loss().cuda()

    '''test on dataset'''
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    num_batches = len(testDataLoader)

    with torch.no_grad():
        for i, (points, target, original_xyz) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

            points = points.data.numpy()
            original_xyz = original_xyz.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.float().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)
            loss = criterion(seg_pred, target, trans_feat, None)

            pred_choice = (seg_pred.cpu().data.numpy().reshape(-1, 1) > 0).astype(np.float)
            pred_choice = pred_choice.flatten()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += points.shape[0] * points.shape[2]
            loss_sum += loss.item()
            plot(original_xyz, pred_choice, batch_label, os.path.join(visual_dir, '{}.png'.format(i)))

        log_string('Eval mean loss: %f' % (loss_sum / num_batches))
        log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

    images = []
    for _ in range(len(testDataLoader)):
        filename = osp.join(visual_dir, "{}.png".format(_))
        images.append(imageio.imread(filename))

    imageio.mimsave(osp.join(visual_dir, 'pred-visual.gif'), images)

if __name__ == '__main__':
    pass
    # args = parse_args()
    # main(args)
