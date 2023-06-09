import pickle
import numpy as np
import os
import shutil
import math
import h5py
import argparse
import sys
sys.path.insert(1, '../utils')  # caution: path[0] is reserved for script path (or '' in REPL)
import open3d
import open3d_utils
from scipy.spatial.distance import cdist
# Create command line flag options

def main():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--object', required=True, help="Name of object")
    parser.add_argument('--mode', default="write", type=str, help="Density")

    args = parser.parse_args()
    object_name = args.object

    OBJ_RESULTS_DIR = "../results/"+ object_name +"_all"
    PICKLE_DATA_DIR = "../deform_contactnet/dataset/"

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
        target_object_pc.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_object_pc.orient_normals_towards_camera_location(camera_location=np.array([0.0,1.0,0.0]))

        target_object_pc_pts = np.asarray(target_object_pc.points)
        target_object_pc_pts_normals = np.asarray(target_object_pc.normals)
        # Read data from mutlple h5 files of the object and stack them
        all_files = [os.path.join(OBJ_RESULTS_DIR, o) for o in os.listdir(OBJ_RESULTS_DIR) if o.endswith(".h5") and "iter" in o]
        # print(all_files)

        sponge_position_at_force_contacted_points_all = []
        press_locations_all = []
        gripper_ori_all = []
        for file in all_files:
            file = h5py.File(file,'r')
            action_success = np.array(file['action_success']).astype(bool) #Only take into account success actions.
            contact_indexes = np.array(file['contact_indexes']).astype(bool)[action_success,:] #Filter with success actions
            gripper_ori = np.array(file['gripper_ori'])[action_success,:,:] #Filter with success actions
            press_locations = np.array(file['press_locations'])[action_success,:]
            normal_forces_on_nodes = np.array(file['normal_forces_on_nodes'])[action_success,:,:]
            sponge_position_at_force= np.array(file['sponge_position_at_force'])[action_success,:,:]
            # print(file.keys()) 
            # --- Append all the contact_location into press_locations_all n*(1,3) and corresponded gripper orientation sin/cos component n*(1,2).
            for i in range(int(sponge_position_at_force.shape[0])):
                sponge_position_at_force_contacted_points_all.append(sponge_position_at_force[i, contact_indexes[i],:])
                press_locations_all.append(press_locations[i])
                # Convert gripper_ori into sin(theta) & cos(theta) components
                sin_component = np.sin(float(gripper_ori[i][:,1])*np.pi/180)
                cos_component = np.cos(float(gripper_ori[i][:,1])*np.pi/180)
                ori = np.array([sin_component,cos_component])
                gripper_ori_all.append(ori)
            # ---  *** ---
        # print(sponge_position_at_force_contacted_points_all[0].shape)
        # print(target_object_pc_pts.shape)
        # print((press_locations_all[0] == target_object_pc_pts).all(1).any()) #Check if sampled points is on target pc
                            
        # Construct ground-truth label n*(n_pts,1) per contact (contact:1 , non-contact:0).
        contact_label_all = []
        for i in range(len(sponge_position_at_force_contacted_points_all)):
            contact_label = np.zeros((target_object_pc_pts.shape[0],1))
            for j in range(target_object_pc_pts.shape[0]):
                dist = cdist(target_object_pc_pts[j].reshape((1,3)),sponge_position_at_force_contacted_points_all[i])
                if np.any(dist < 0.02):
                    contact_label[j] = 1.0       
            
            contact_label_all.append(contact_label)
        # ==========
        # print(contact_label_all[0].shape)
        # pts_color = np.zeros((target_object_pc_pts.shape[0],3)) 
        # test = (contact_label_all[0]*255).reshape((target_object_pc_pts.shape[0],))
        # print(test.shape)
        # pts_color[:,0] = test
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(np.array(target_object_pc_pts))
        # pcd.colors = open3d.utility.Vector3dVector(np.array(pts_color))
        # open3d.visualization.draw_geometries([pcd]) 
        # ============
        # Construct feature vector 
        feature_vec_all = []
        for i in range(len(press_locations_all)):
            feature_vec = np.zeros((target_object_pc_pts.shape[0],2))
            ind = get_index_array(press_locations_all[i][:,0],target_object_pc_pts[:,0]) #Get index of the press location in the target object pc and set the feature vector to be the orientation
            feature_vec[ind,:] = gripper_ori_all[i]
            # print(ind)
            feature_vec_all.append(feature_vec)

        # print(len(contact_label_all),contact_label_all[0].shape, target_object_pc_pts.shape)
        target_object_pc_pts_with_contact_label_all = []
        target_object_pc_pts_plus_normals = np.append(target_object_pc_pts,target_object_pc_pts_normals , axis=1) #append contact loc with normals
        for i in range(len(sponge_position_at_force_contacted_points_all)):
            # print(sum(contact_label_all[i]))
            target_object_pc_pts_plus_normals_feature = np.append(target_object_pc_pts_plus_normals, feature_vec_all[i], axis=1) #append contact loc and normals with feature vector = input
            target_object_pc_pts_with_contact_label = np.append(target_object_pc_pts_plus_normals_feature, contact_label_all[i], axis=1) #append input with the label
            target_object_pc_pts_with_contact_label_all.append(target_object_pc_pts_with_contact_label) # All in all: n*(n_pts,9) - 3: pts location, 3: pts normals, 2: feature vectors, 1: label

        with open(os.path.join(PICKLE_DATA_DIR,object_name+'.pickle'), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump((target_object_pc_pts_with_contact_label_all), f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(PICKLE_DATA_DIR,object_name+'.pickle'), 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data = pickle.load(f)
            print(len(data),data[0].shape)
            print(data[0])
            np.savetxt('test.csv', data[0], delimiter=',')  

def get_index_array(element, arr , rel_tol=1e-6):
    # target_object_pc_pts[:,0]
    for index, item in enumerate(arr):
        if math.isclose(item, element, rel_tol=rel_tol):
            res = index
    return res

if __name__ == "__main__":
    main()
