from random import sample
import numpy as np
from copy import deepcopy
from isaacgym import gymtorch
from isaacgym import gymapi
import open3d
from scipy.signal import butter, filtfilt
import sys

def sample_points_from_pcd(file_path, number_of_points):
    sampled_points = []
    # Load pcd file
    pcd = open3d.io.read_point_cloud(file_path)
    pcd = np.asarray(pcd.points)
    randomRows_idxs = sample(range(pcd.shape[0]), number_of_points)
    for i in randomRows_idxs:
        sampled_points.append(pcd[i])
    return sampled_points

def visualize_pc_open3d(pcds):
    open3d.visualization.draw_geometries([pcds]) 
