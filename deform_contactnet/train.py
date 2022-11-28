"""
Author: Tran Nguyen Le
Date: Nov 2022
"""

import os
import sys
import torch
import numpy as np
import datetime
import logging
import provider
import importlib
import shutil
import argparse
import socket
from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from torch.nn.utils.rnn import pad_sequence 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
                    
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=1, type=int,  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--trainval_split', default=0.7, type=float, help='Split percent train&val')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='use wandb')

    return parser.parse_args()

def custom_collate(data): #(2)
    point = [torch.tensor(d[0]) for d in data] #(3)
    label = [torch.tensor(d[1]) for d in data]
    point = pad_sequence(point, batch_first=True, padding_value=0) #(4)
    label = pad_sequence(label, batch_first=True, padding_value=0) #(4)
    return point,label

def f1_confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()
    
    epsilon = 1e-7
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)

    return f1,true_positives, false_positives, true_negatives, false_negatives


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=1):
    classifier = model.eval()
    f1_score_val = 0
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        point_vis = points[0,:,0:3] #For vis
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        predictions = (pred > 0.5).float()
        f1_score_per_batch,_,_,_, _ = f1_confusion(predictions,target.float())
        f1_score_val += f1_score_per_batch

        #    Ground_truth vis
        pred_vis = predictions[0,:].data.cpu().numpy().reshape((predictions.shape[1],1))
        prediction_vis = np.hstack((point_vis,pred_vis))

        #    Ground_truth vis
        target_vis = target[0,:].data.cpu().numpy().reshape((target.shape[1],1))
        ground_truth_vis = np.hstack((point_vis,target_vis))
        # --- 
    val_acc = f1_score_val/len(loader)
    if args.use_wandb:
        wandb.log({
                "Ground_truth": wandb.Object3D(
                    {
                        "type": "lidar/beta",
                        "points": ground_truth_vis,
                    }
                )})
        wandb.log({
                "Prediction": wandb.Object3D(
                    {
                        "type": "lidar/beta",
                        "points": prediction_vis,
                    }
                )})

    return val_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'dataset/'

    trainval_dataset = ModelNetDataLoader(root=data_path, split='train')
    train_dataset_percentage = args.trainval_split
    train_dataset, validation_dataset = torch.utils.data.random_split(trainval_dataset,[int(train_dataset_percentage * len(trainval_dataset)), len(trainval_dataset) - int(train_dataset_percentage * len(trainval_dataset))], generator=torch.Generator().manual_seed(42))
    # test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=custom_collate,num_workers=10, drop_last=True)
    valDataLoader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=custom_collate, num_workers=10, drop_last=True)

    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet.py', str(exp_dir))
    shutil.copy('./train.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_val_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    # Magic
    if args.use_wandb:
        wandb.watch(classifier, log_freq=100)
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier = classifier.train()
        loss_per_epoch = 0
        f1_score_train = 0
        scheduler.step()
        
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1) #For network

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.float(), trans_feat)
            loss.backward()
            optimizer.step()
            global_step += 1
            predictions = (pred > 0.5).float()
            f1_score_per_batch,_,_,_, _ = f1_confusion(predictions,target.float())
            f1_score_train += f1_score_per_batch
            loss_per_epoch += loss.item()
            # --- 
            if args.use_wandb:
                wandb.log({"loss_per_batch": loss.item()})
                wandb.log({"f1_score_per_batch": f1_score_per_batch})
        f1_score_train /= len(trainDataLoader)
        loss_per_epoch /= len(trainDataLoader)
        if args.use_wandb:
            wandb.log({"Training/Train_f1_score_per_epoch": f1_score_train, "epoch": epoch})
            wandb.log({"Training/Train_loss_per_epoch": loss_per_epoch, "epoch": epoch})

        log_string('F1_score: %f' % f1_score_train)  

        with torch.no_grad():
            val_acc = test(classifier.eval(), valDataLoader, num_class=num_class)
            if (val_acc >= best_val_acc):
                best_val_acc = val_acc
                best_epoch = epoch + 1
            if args.use_wandb:
                wandb.log({"Validation/Valid_f1_score_per_epoch": val_acc,"epoch": epoch})

            if (val_acc >= best_val_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'val_acc': val_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    if args.use_wandb:
        import wandb
        import uuid
        import wandb
        config = dict (
        learning_rate = args.learning_rate,
        batch_size = args.batch_size,
        decay_rate = args.decay_rate,
        optimizer = str(args.optimizer),
        )
        run_id = str(uuid.uuid4())[:8]
        computer_name = str(socket.gethostname())
        wandb.init(project="robo-sponge",
                            name=f'Deform-ContactNet-{computer_name}-{str(args.model)}-{run_id}',
                            group=f'Deform-ContactNet',
                            config=config)
    main(args)
