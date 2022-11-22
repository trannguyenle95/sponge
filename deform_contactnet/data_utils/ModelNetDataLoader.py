'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, split='train',num_point=1000,use_normals=True,use_uniform_sample=True):
        self.root = root
        self.npoints = num_point
        self.uniform = use_uniform_sample
        self.use_normals = use_normals
        total_num_data = 0
        all_files = [os.path.join(self.root, o) for o in os.listdir(self.root) if o.endswith(".pickle")]
        all_data = []
        for file in all_files:
            data = np.load(file,allow_pickle=True)
            all_data.extend(data)
        self.data = data
        print(len(all_data))
        self.data_idxs = [i for i in range(len(all_data))]
    def get_dataset_statistics(self):
        # return self.num_point, self.pos_label_weight, self.data_idxs
        return None, 20, self.data_idxs

    def __len__(self):
        return len(self.data_idxs)

    def _get_item(self, index):
        # ----
        # data_idx = self.all_data_file[index]
        # print(data_idx)
        # data = np.load(self.all_data_file[data_idx])  
        data = self.data[index]
        self.npoints = data.shape[0]
        if not self.use_normals:
            point_set = np.hstack((data[:, 0:3],data[:,6:8]))
        else:
            point_set = data[:, 0:8].astype(np.float32) # xyz + normals+ feature-vec  
        contact_labels = data[:, 8].astype(np.int32) 
        # -----
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints, :]
        contact_labels = contact_labels[0:self.npoints]
        # Normalize pc
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])       

        return point_set, contact_labels

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader(root='../dataset/', split='train')
    train_dataset, validation_dataset = torch.utils.data.random_split(data,[int(0.8 * len(data)), len(data) - int(0.8 * len(data))], generator=torch.Generator().manual_seed(42))
    DataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)

    # Previously rlgpu used Pytorch 1.7.0 
