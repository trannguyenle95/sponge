import numpy as np
import os
import shutil

import h5py
import argparse
import sys
sys.path.insert(1, '../utils')  # caution: path[0] is reserved for script path (or '' in REPL)
import open3d_utils

# Create command line flag options
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--object', required=True, help="Name of object")

args = parser.parse_args()
object_name = args.object

OBJ_RESULTS_DIR = "../results/"+ object_name +"_all"

target_object_pcd_file_name = "target_object_pc/"+str(object_name)+".pcd"
target_object_pcd_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),target_object_pcd_file_name)

#  Voxelize the target objects
if True:
    voxel_centers = open3d_utils.voxelize(target_object_pcd_file)
print(voxel_centers.shape)
one_hot_vector = np.ones((voxel_centers.shape[0],2))
one_hot_vector_target_object = one_hot_vector * [1,0]
one_hot_vector_sponge = one_hot_vector * [0,1]

target_object_pc_input = np.hstack((voxel_centers,one_hot_vector_target_object))
print(target_object_pc_input)

# Read data from mutlple h5 files of the object and stack them
all_files = [os.path.join(OBJ_RESULTS_DIR, o) for o in os.listdir(OBJ_RESULTS_DIR) if o.endswith(".h5") and "iter" in o]
print(all_files)