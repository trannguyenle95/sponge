import h5py
import numpy as np
import argparse
import os
import open3d
import sys
sys.path.insert(1, '../utils')  # caution: path[0] is reserved for script path (or '' in REPL)
import open3d_utils
from descartes import PolygonPatch
import alphashape
import matplotlib.pyplot as plt
import matplotlib
RESULTS_DIR = "../results/"
RESULTS_STORAGE_TAG = "_all"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', required=True, help="Name of object")
    parser.add_argument('--density', default=100, type=float, help="Density")
    parser.add_argument('--youngs', required=True, type=float, help="Elastic modulus of the sponge [Pa]")

    args = parser.parse_args()
    object_name = args.object
    # env_id = 9
    ### Result file
    big_folder_name = object_name + RESULTS_STORAGE_TAG
    small_folder_name = object_name + "_" + str(int(args.youngs))
    object_file_name = object_name +  "_iter"+"_"+"2"+".h5"
    h5_file_path = os.path.join(RESULTS_DIR, big_folder_name, small_folder_name, object_file_name)
    ### Load target object point cloud 
    target_object_pcd_file_name = "target_object_pc/"+str(object_name)+".pcd"
    target_object_pcd_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),target_object_pcd_file_name)
    target_object_pc,_ = read_point_clouds_from_open3d(target_object_pcd_file)
    ### Read data file
    action_success,pressed_forces, sponge_position_at_force, contact_indexes,normal_forces_on_nodes = read_h5_data(h5_file_path)
    ###
    fig, axs = plt.subplots(2,5,constrained_layout=True, figsize=(20, 10))
    for env_id in range(action_success.shape[0]):
        sponge_pc = open3d.geometry.PointCloud()
        sponge_pc.points = open3d.utility.Vector3dVector(sponge_position_at_force[env_id])
        # open3d_utils.visualize_pc_open3d(sponge_pc+target_object_pc) 

        mask = contact_indexes[env_id].astype(bool)
        contact_points_loc = sponge_position_at_force[env_id][mask, :]
        normal_forces_on_nodes_filtered = np.sum(normal_forces_on_nodes[env_id][mask,:],axis=1)
        # normal_forces_on_nodes_filtered *= 10 #for plot

        contact_points_loc_2D = np.delete(contact_points_loc, 1, 1)
        alpha_shape = alphashape.alphashape(contact_points_loc_2D.tolist(), 0.)
        cmap = matplotlib.cm.cool
        norm = matplotlib.colors.Normalize(vmin=0, vmax=3)
        if env_id <= 4:
            axs[0,env_id].scatter(*zip(*contact_points_loc_2D.tolist()),c=normal_forces_on_nodes_filtered, vmin=0, vmax=2.8, cmap='Greens')
            # axs[0,env_id].scatter(*zip(*contact_points_loc_2D.tolist()),c=normal_forces_on_nodes_filtered, cmap=cmap, norm=norm)
            axs[0,env_id].add_patch(PolygonPatch(alpha_shape, alpha=0.2))
            axs[0,env_id].set_title('F = ' + str(pressed_forces[env_id])+ " - A = "+str(format(float(alpha_shape.area),".5f")))

        else:
            axs[1,env_id-5].scatter(*zip(*contact_points_loc_2D.tolist()),c=normal_forces_on_nodes_filtered, vmin=0, vmax=2.8, cmap='Greens')
            # axs[1,env_id-5].scatter(*zip(*contact_points_loc_2D.tolist()),c=normal_forces_on_nodes_filtered,cmap=cmap, norm=norm)
            axs[1,env_id-5].add_patch(PolygonPatch(alpha_shape, alpha=0.2))
            axs[1,env_id-5].set_title('F = ' + str(pressed_forces[env_id]) + " - A = "+str(format(float(alpha_shape.area),".5f")))
        print(env_id, alpha_shape.bounds)
    plt.show()

    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.asarray(a))
    # open3d_utils.visualize_pc_open3d(pcd+target_object_pc) 
def read_h5_data(h5_file):
    if os.path.exists(h5_file):
        h5_file = h5py.File(h5_file,'r')
        print(h5_file.keys())
        pressed_forces = np.array(h5_file['pressed_forces'])
        normal_forces_on_nodes = np.array(h5_file['normal_forces_on_nodes'])
        sponge_position_at_force = np.array(h5_file['sponge_position_at_force'])
        contact_indexes = np.array(h5_file['contact_indexes'])
        action_success = np.array(h5_file['action_success'])
    return action_success, pressed_forces, sponge_position_at_force, contact_indexes,normal_forces_on_nodes

def read_point_clouds_from_open3d(target_object_pc_file):
    target_object_pc = open3d.io.read_point_cloud(target_object_pc_file)
    target_object_points = np.asarray(target_object_pc.points)
    return target_object_pc,target_object_points
if __name__ == "__main__":
    main()
