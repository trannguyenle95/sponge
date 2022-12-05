import numpy as np
import argparse
import os
import open3d
import sys
sys.path.insert(1, '../utils')  # caution: path[0] is reserved for script path (or '' in REPL)
import random 
import importlib
import torch
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR,'deform_contactnet','models'))
import numpy_indexed as npi
from tqdm import tqdm


MODEL_DIR = '../deform_contactnet/log/classification/deform_contactnet_pointnet'
RESULTS_STORAGE_TAG = "_all"

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def predict_contact(input, use_cpu):
    input[:, 0:3] = pc_normalize(input[:, 0:3]) # normalize input pc very important!      
    input = torch.reshape(torch.from_numpy(input), (1, input.shape[0], input.shape[1]))

    model_name = os.listdir(MODEL_DIR + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)
    predictor = model.get_model(num_class=1, normal_channel=True)
    checkpoint = torch.load(str(MODEL_DIR) + '/checkpoints/best_model.pth')
    predictor.load_state_dict(checkpoint['model_state_dict'])
    predictor.eval()
    if not use_cpu:
        predictor.cuda()
        input = input.cuda()
    input = input.transpose(2, 1)
    pred, _ = predictor(input)
    predictions = (pred > 0.5).float()
    predictions = torch.squeeze(predictions)
    return predictions.cpu().detach().numpy()


def pick_a_point(list_of_pts, list_of_normals):
    """ Returns waypoint_pos (not normalized), waypoint_ori (in degree), formatted_input to network
    """
    feature_vec = np.zeros((list_of_pts.shape[0],2))
    input = np.concatenate((list_of_pts, list_of_normals,feature_vec), axis=1)

    randomRows_idx = random.sample(range(list_of_pts.shape[0]), 1)
    chosen_gipper_ori_degree = random.randint(-90, 90)
    sin_component = np.sin(float(chosen_gipper_ori_degree)*np.pi/180)
    cos_component = np.cos(float(chosen_gipper_ori_degree)*np.pi/180)
    chosen_gipper_ori = np.array([sin_component,cos_component])
    input[randomRows_idx,-2:] = chosen_gipper_ori
    return list_of_pts[randomRows_idx,:], chosen_gipper_ori_degree, input.astype(np.float32)

def sample_waypoints(num_waypoints, list_of_pts, list_of_normals, vis):
    waypoints = []
    waypoints_ori = []
    # waypoints_idxs = []
    i = 0
    remain_list_of_pts = list_of_pts
    # for i in range(num_waypoints):
    while remain_list_of_pts.shape[0] > 10:
        if i == 0:
            init_waypoint, init_waypoint_ori, init_waypoint_input = pick_a_point(list_of_pts,list_of_normals)
            waypoints.append(init_waypoint)
            waypoints_ori.append(init_waypoint_ori)
            with torch.no_grad():
                contact_prediction = predict_contact(init_waypoint_input,use_cpu=False)
            prev_contact_ind = np.asarray(np.where(contact_prediction))
            remain_list_of_pts = list_of_pts[~contact_prediction.astype(bool),:]
            remain_list_of_normals = list_of_normals[~contact_prediction.astype(bool),:]
            i += 1
        else:
            while True:
                waypoint, waypoint_ori, waypoint_input = pick_a_point(remain_list_of_pts,remain_list_of_normals)
                with torch.no_grad():
                    contact_prediction = predict_contact(waypoint_input,use_cpu=False)
                curr_contact_ind = np.asarray(np.where(contact_prediction))
                # If overlap too much with prev contact patch, pick again
                if np.sum(prev_contact_ind == curr_contact_ind) < 20:
                    # print("No overlap")
                    # ==========
                    if vis:
                        pts_color = np.zeros((remain_list_of_pts.shape[0],3)) 
                        test = (contact_prediction*255).reshape((remain_list_of_pts.shape[0],))
                        pts_color[:,0] = test
                        pcd = open3d.geometry.PointCloud()
                        pcd.points = open3d.utility.Vector3dVector(np.array(remain_list_of_pts))
                        pcd.colors = open3d.utility.Vector3dVector(np.array(pts_color))
                        open3d.visualization.draw_geometries([pcd]) 
                    # ============
                    waypoints.append(waypoint)
                    waypoints_ori.append(waypoint_ori)
                    remain_list_of_pts = remain_list_of_pts[~contact_prediction.astype(bool),:]
                    remain_list_of_normals = remain_list_of_normals[~contact_prediction.astype(bool),:]
                    prev_contact_ind = curr_contact_ind
                    break
    return waypoints, waypoints_ori

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', required=True, help="Name of object")
    parser.add_argument('--use_vis', action='store_true', default=False, help='Use visualization')

    args = parser.parse_args()
    object_name = args.object

    ### Load target object point cloud 
    target_object_pcd_file_name = "target_object_pc/"+str(object_name)+".pcd"
    target_object_pcd_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),target_object_pcd_file_name)
    target_object_pcd = open3d.io.read_point_cloud(target_object_pcd_file)

    target_object_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_object_pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0,1.0,0.0]))
    target_object_points = np.asarray(target_object_pcd.points)
    target_object_normals = np.asarray(target_object_pcd.normals)

    # Sample n sequence and seek to the one with least waypoints 
    n_sequences = 50
    sequences_waypoints = []
    sequences_waypoints_ori = []
    for i in tqdm(range(n_sequences)):
        waypoints, waypoints_ori = sample_waypoints(10,target_object_points,target_object_normals,vis=False)
        waypoints = np.vstack(waypoints)
        waypoints_ori = np.asarray(waypoints_ori)
        sequences_waypoints.append(waypoints)
        sequences_waypoints_ori.append(waypoints_ori)
        
    shortest_sequence_waypoints = min(sequences_waypoints,key=len)
    shortest_sequence_ori = sequences_waypoints_ori[sequences_waypoints.index(shortest_sequence_waypoints)]
    # print(len(sequences_waypoints), sequences_waypoints[0].shape, len(sequences_waypoints_ori), sequences_waypoints_ori[0].shape)
    for i in sequences_waypoints:
        print(i.shape)
    print(sequences_waypoints.index(shortest_sequence_waypoints),shortest_sequence_waypoints.shape, shortest_sequence_waypoints, shortest_sequence_ori)

    waypoints_idxs = npi.indices(target_object_points,shortest_sequence_waypoints) #Get index of waypoints respect to point cloud

    print(waypoints_idxs)
    for i in waypoints_idxs:
        print(target_object_points[i,:])
    # ==========
    if args.use_vis:
        pts_color = np.zeros((target_object_points.shape[0],3)) 
        for i in waypoints_idxs:
            pts_color[i,1] = 255
        # pts_color[:,0] = label_colors
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(target_object_points))
        pcd.colors = open3d.utility.Vector3dVector(np.array(pts_color))
        open3d.visualization.draw_geometries([pcd]) 
    # ============
if __name__ == "__main__":
    main()

