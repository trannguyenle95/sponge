from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np
from copy import deepcopy
from isaacgym import gymtorch
from isaacgym import gymapi
import open3d
from scipy.signal import butter, filtfilt
import sys

def setup_cam(gym, env, cam_width, cam_height, cam_pos, cam_target):
    cam_props = gymapi.CameraProperties()
    cam_props.width = cam_width
    cam_props.height = cam_height    
    cam_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)
    return cam_handle, cam_props

def get_partial_point_cloud(gym, sim, env, cam_handle, cam_prop, min_z = 0.005, visualization=False):
    
    cam_width = cam_prop.width
    cam_height = cam_prop.height
    # Render all of the image sensors only when we need their output here
    # rather than every frame.
    gym.render_all_camera_sensors(sim)

    points = []
    # print("Converting Depth images to point clouds. Have patience...")

    depth_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)


    # Get the camera view matrix and invert it to transform points from camera to world
    # space
    
    vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, cam_handle)))

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    proj = gym.get_camera_proj_matrix(sim, env, cam_handle)
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]
    # Ignore any points which originate from ground plane or empty space
    depth_buffer[seg_buffer == 0] = -10001

    centerU = cam_width/2
    centerV = cam_height/2
    for i in range(cam_width):
        for j in range(cam_height):
            if depth_buffer[j, i] < -10000:
                continue
            if seg_buffer[j, i] > 0:
                u = -(i-centerU)/(cam_width)  # image-space coordinate
                v = (j-centerV)/(cam_height)  # image-space coordinate
                d = depth_buffer[j, i]  # depth buffer value
                X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                p2 = X2*vinv  # Inverse camera view to get world coordinates
                points.append([p2[0, 0], p2[0, 1], p2[0, 2]])
                # color.append(c)
    
    if visualization:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))
        open3d.visualization.draw_geometries([pcd]) 

    return np.array(points).astype('float32')  

def visualize_pc_open3d(pcds):
    open3d.visualization.draw_geometries([pcds]) 

# ========================================= FORCE ====================================
def get_force_based_torque(F_des, F_curr,moving_average,torque_des):
        """Torque-based control with target gripper force F_des."""
        total_F_curr = moving_average[
            -1]  # Use the LP and averaged value instead of raw readings
        if np.sum(F_curr) == 0.0:
            total_F_curr = 0

        total_F_err = np.sum(F_des) - total_F_curr
        # Compute error values for state transitions
        F_curr_mag = (np.abs(F_curr))

        # Kp = self.cfg['force_control']['Kp']
        # min_torque = self.cfg['force_control']['min_torque']
        Kp =  1.5
        min_torque = -5
        torque_des[0] -= min(total_F_err * Kp, 3 * Kp)
        torque_des[0] = min(min_torque, torque_des[0])
        torque_des[1] -= min(total_F_err * Kp, 3 * Kp)
        torque_des[1] = min(min_torque, torque_des[1])
        torque_des[2] -= min(total_F_err * Kp, 3 * Kp)
        torque_des[2] = min(min_torque, torque_des[2])
        return torque_des, F_curr_mag, total_F_err

def butter_lowpass_filter(data):
    """Low-pass filter the dynamics data."""
    fs = 20
    cutoff = 0.05
    nyq = 0.5 * fs
    order = 1
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
