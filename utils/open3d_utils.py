from random import sample
import numpy as np
from copy import deepcopy
from isaacgym import gymtorch
from isaacgym import gymapi
import open3d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
import sys

def sample_points_and_normals_from_pcd(file_path, number_of_points):
    epsilon = 0.05
    sampled_points_real = [] #Real contact point on target object surface
    sampled_points_approach = [] #Approach point, few cm away from the real contact point along the approaching vector
    sampled_points_normals= []
    sampled_points_normals_euler = []
    b = np.array([0,1,0]).reshape((1,3))
    
    # Load pcd file
    pcd = open3d.io.read_point_cloud(file_path)
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    points = np.asarray(pcd.points)
    points_normals = np.asarray(pcd.normals)
    randomRows_idxs = sample(range(points.shape[0]), number_of_points)
    for i in randomRows_idxs:
        sampled_points_real.append(points[i])
        if points_normals[i][1] < 0:
           points_normals[i] *= -1
        approach_pts = points[i] + epsilon*points_normals[i]
        print("pts: ",points[i], " -- normal: ",points_normals[i], "-- approach: ",approach_pts)
        sampled_points_approach.append(approach_pts)
        rotation = R.align_vectors(points_normals[i].reshape((1,3)),b)
        sampled_points_normals.append(points_normals[i])
        euler_xyz = rotation[0].as_euler('XYZ', degrees=True)
        print("euler_xyz: ",euler_xyz)
        sampled_points_normals_euler.append(euler_xyz)
    return sampled_points_real, sampled_points_approach , sampled_points_normals, sampled_points_normals_euler

def visualize_pc_open3d(pcds):
    open3d.visualization.draw_geometries([pcds]) 
