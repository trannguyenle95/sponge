import pickle
import numpy as np
import os
import shutil

import h5py
import argparse
import sys
sys.path.insert(1, '../utils')  # caution: path[0] is reserved for script path (or '' in REPL)
import open3d
import open3d_utils
from scipy.spatial.distance import cdist
# Create command line flag options
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--object', required=True, help="Name of object")

args = parser.parse_args()
object_name = args.object

OBJ_RESULTS_DIR = "../results/"+ object_name +"_all"
PICKLE_DATA_DIR = "../dataset/"

target_object_pcd_file_name = "target_object_pc/"+str(object_name)+".pcd"
target_object_pcd_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),target_object_pcd_file_name)

# #  Voxelize the target objects
# if True:
#     voxel_centers = open3d_utils.voxelize(target_object_pcd_file)
# print(voxel_centers.shape)
# one_hot_vector = np.ones((voxel_centers.shape[0],2))
# one_hot_vector_target_object = one_hot_vector * [1,0]
# one_hot_vector_sponge = one_hot_vector * [0,1]

if args.mode == "write":
    # target_object_pc_input = np.hstack((voxel_centers,one_hot_vector_target_object))
    # print(target_object_pc_input)
    target_object_pc = open3d.io.read_point_cloud(target_object_pcd_file)
    target_object_pc_pts = np.asarray(target_object_pc.points)
    # Read data from mutlple h5 files of the object and stack them
    all_files = [os.path.join(OBJ_RESULTS_DIR, o) for o in os.listdir(OBJ_RESULTS_DIR) if o.endswith(".h5") and "iter" in o]
    print(all_files)
    sponge_position_at_force_contacted_points_all = []
    press_locations_all = []
    gripper_ori_all = []
    for file in all_files:
        print(file)
        file = h5py.File(file,'r')
        action_success = np.array(file['action_success']).astype(bool) #Only take into account success actions.
        contact_indexes = np.array(file['contact_indexes']).astype(bool)[action_success,:] #Filter with success actions
        gripper_ori = np.array(file['gripper_ori'])[action_success,:,:] #Filter with success actions
        press_locations = np.array(file['press_locations'])[action_success,:]
        normal_forces_on_nodes = np.array(file['normal_forces_on_nodes'])[action_success,:,:]
        sponge_position_at_force= np.array(file['sponge_position_at_force'])[action_success,:,:]
        print(file.keys()) 
        for i in range(int(sponge_position_at_force.shape[0])):
            sponge_position_at_force_contacted_points_all.append(sponge_position_at_force[i, contact_indexes[i],:])
            press_locations_all.append(press_locations[i])
            gripper_ori_all.append(float(gripper_ori[i][:,1]))
    print(sponge_position_at_force_contacted_points_all[0].shape)
    print(target_object_pc_pts.shape)
    # Construct ground-truth label per contact.
    contact_label_all = []
    for i in range(len(sponge_position_at_force_contacted_points_all)):
        contact_label = np.zeros((target_object_pc_pts.shape[0],1))
        for j in range(target_object_pc_pts.shape[0]):
            dist = cdist(target_object_pc_pts[j].reshape((1,3)),sponge_position_at_force_contacted_points_all[i])
            if np.any(dist < 0.005):
                contact_label[j] = 1.0       
        
        contact_label_all.append(contact_label)
    print(len(contact_label_all),contact_label_all[0].shape, target_object_pc_pts.shape)
    target_object_pc_pts_with_contact_label_all = []
    for i in range(len(sponge_position_at_force_contacted_points_all)):
        print(sum(contact_label_all[i]))
        target_object_pc_pts_with_contact_label = np.append(target_object_pc_pts, contact_label_all[i], axis=1)
        target_object_pc_pts_with_contact_label_all.append(target_object_pc_pts_with_contact_label)

    with open(os.path.join(PICKLE_DATA_DIR,object_name+'.pickle'), 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump((target_object_pc_pts_with_contact_label_all, gripper_ori_all), f, pickle.HIGHEST_PROTOCOL)
else:
    with open(os.path.join(PICKLE_DATA_DIR,object_name+'.pickle'), 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        value1 ,value2= pickle.load(f)
        print(value1)
        print(value2)
