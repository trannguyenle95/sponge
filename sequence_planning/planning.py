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
import warnings
import tsp_2opt
warnings.filterwarnings("ignore", category=DeprecationWarning) 
MODEL_DIR = '../deform_contactnet/log/classification/deform_contactnet_pointnet'

def write_results(object_name, waypoints, waypoints_ori):
    import pickle
    import re
    RESULTS_DIR = "planning_results/"
    regex = re.compile(r'\d+')
    num_seq = 0
    file_idxes = []
    os.makedirs(os.path.join(RESULTS_DIR, object_name), exist_ok=True)
    if os.listdir(os.path.join(RESULTS_DIR, object_name)):
        for file in os.listdir(os.path.join(RESULTS_DIR, object_name)):
            file_idx = int(regex.search(file).group(0)) #extract number from file name
            file_idxes.append(file_idx)
        num_seq = max(file_idxes)+1
    else:
        num_seq = 0
    object_file_name = object_name +  "_seq"+"_"+str(num_seq)+".pickle"
    object_file_name = os.path.join(RESULTS_DIR, object_name, object_file_name)

    with open(object_file_name, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump((waypoints,waypoints_ori), f, pickle.HIGHEST_PROTOCOL)
    print("Done writing results to ", object_file_name)

def pc_normalize(pc):
    """
    This function normalizes the points in point cloud. 
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def predict_contact(input, use_cpu):
    """
    This function predicts the contact patch on target objects given the formatted input (point_xyz, point_normal, feature_vector). 
    Inputs:
        input (n_pts,8)
        use_cpu (bool)
    Outputs:
        predictions (n_pts,8): include only 0 (non-contact) and 1 (in-contact)
    """
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
    """ This function randomly pick a point in the point cloud.
    Inputs:
        list_of_pts (n_pts, 3): list of xyz coordinates of point cloud. 
        list_of_normals (n_pts, 3): list of normals of point cloud.
    Outputs:
        waypoint_pos (1,3): xyz coordinate of chosen point.
        waypoint_ori (int): gripper orientation in degree
        formatted_input (n_pts,8): input for deform_contactnet
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

def sample_waypoints(list_of_pts, list_of_normals, vis_seq, vis_seq_count):
    """ This function samples a list of waypoints to maximize the coverage area of the target object.

    1. Read the point cloud, randomly pick a point as the initial waypoint and predict the contact patch with the target object.
    2. Randomly pick another point that is not inside the prev contact patch and predict the new contact patch. 
    3. If the intersection between new contact patch and the prev one, BIGGER than a threshold -> Abort this point and pick another point. 
    4. If SMALLER than a threshold -> Save the chosen point (pos+ori) to the list of waypoints.
    5. Loop 2-4 until we cover all the points in the point cloud.
    Inputs:
        list_of_pts (n_pts, 3): list of xyz coordinates of point cloud. 
        list_of_normals (n_pts, 3): list of normals of point cloud.
        vis_seq (bool): Visulization of the process.
        vis_seq_count (int): only visualize the first seq
    Outputs:
        waypoints (n_waypoints,3)
        waypoints_ori (n_waypoints,1)
    """

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
                    if vis_seq and vis_seq_count == 0:
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

def tsp(waypoints,waypoints_ori):
    distances = tsp_2opt.precalculate_distances(waypoints)
    print(distances)
    num_points = waypoints.shape[0]    
    best_route = tsp_2opt.random_route(num_points)
    path_improved = True
    while (path_improved):
        path_improved = False
        for i in range(0, num_points - 1):
            for j in range(i + 1, num_points):

                original_distance = tsp_2opt.get_distance(
                    best_route[i], best_route[(i + 1) % num_points]) + tsp_2opt.get_distance(best_route[j], best_route[(j + 1) % num_points])
                swapped_distance = tsp_2opt.get_distance(
                    best_route[i], best_route[j]) + tsp_2opt.get_distance(best_route[(i + 1) % num_points], best_route[(j + 1) % num_points])

                if (swapped_distance < original_distance):
                    best_route = tsp_2opt.swap_2opt(
                        best_route, i, j)

                    path_improved = True
                    break
            if path_improved:
                break
    return best_route
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', required=True, help="Name of object")
    parser.add_argument('--use_vis', action='store_true', default=False, help='Use visualization')
    parser.add_argument('--write', action='store_true', default=False, help='Use visualization')

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
    n_sequences = 1
    sequences_waypoints = []
    sequences_waypoints_ori = []
    for i in tqdm(range(n_sequences)):
        waypoints, waypoints_ori = sample_waypoints(target_object_points,target_object_normals,vis_seq=False,vis_seq_count=i) 
        waypoints = np.vstack(waypoints)
        waypoints_ori = np.asarray(waypoints_ori)
        sequences_waypoints.append(waypoints)
        sequences_waypoints_ori.append(waypoints_ori)
        
    shortest_sequence_waypoints = min(sequences_waypoints,key=len)
    shortest_sequence_ori = sequences_waypoints_ori[sequences_waypoints.index(shortest_sequence_waypoints)]
    if args.write:
        write_results(object_name,shortest_sequence_waypoints,shortest_sequence_ori)
    for i in sequences_waypoints:
        print(i.shape)
    print(sequences_waypoints.index(shortest_sequence_waypoints),shortest_sequence_waypoints.shape, shortest_sequence_waypoints, shortest_sequence_ori)

    waypoints_idxs = npi.indices(target_object_points,shortest_sequence_waypoints) #Get index of waypoints respect to point cloud
    best_route = tsp(shortest_sequence_waypoints, shortest_sequence_ori)
    print(best_route)
    for i in waypoints_idxs:
        print(target_object_points[i,:])
    # ========== Visualization of waypoints ==========
    if args.use_vis:
        pts_color = np.zeros((target_object_points.shape[0],3)) 
        pts_color[waypoints_idxs[best_route[0]],1] = 255 #First point in best route coresponded to original pc
        for i in waypoints_idxs:
            pts_color[i,0] = 255

        lines = []
        for i in range(len(best_route)-1):
            line = np.array([best_route[i],best_route[i+1]])
            lines.append(line)
        lines = np.asarray(lines)
        print(lines)
        colors = [[1, 0, 0] for i in range(len(lines))]

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(target_object_points))
        pcd.colors = open3d.utility.Vector3dVector(np.array(pts_color))
        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(np.array(shortest_sequence_waypoints))
        line_set.lines = open3d.utility.Vector2iVector(lines)
        line_set.colors = open3d.utility.Vector3dVector(colors)
        
        open3d.visualization.draw_geometries([line_set,pcd]) 
    # =================================================

if __name__ == "__main__":
    main()

