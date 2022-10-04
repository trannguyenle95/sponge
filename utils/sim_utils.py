from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np
from copy import deepcopy
from isaacgym import gymtorch
from isaacgym import gymapi
import open3d
from scipy.signal import butter, filtfilt
import sys
import os
import copy
import fileinput
from shutil import copyfile

# ==================== SIM UTILS ====================================

def set_sponge_matl_props(base_dir, elast_mod, poiss_ratio,density):
    """Set the Sponge material properties by copying and modifying a URDF template."""
    # TODO: There may be now be a built-in function for setting material properties in the Gym Python API.

    template_path = os.path.join(base_dir, 'sponge_template.urdf')
    file_path = os.path.join(base_dir, 'soft_body.urdf')
    copyfile(template_path, file_path)
    with fileinput.FileInput(file_path, inplace=True) as file_obj:
        for line in file_obj:
            if 'youngs' in line:
                print(line.replace('youngs value=""', f'youngs value="{elast_mod}"'), end='')
            elif 'poissons' in line:
                print(line.replace('poissons value=""', f'poissons value="{poiss_ratio}"'), end='')
            elif 'density' in line:
                print(line.replace('density value=""', f'density value="{density}"'), end='')
            else:
                print(line, end='')
# ==================== VISUAL PROCESSING UTILS ====================================
def setup_cam(gym, envs, props):
    cam_prop = gymapi.CameraProperties()
    cam_prop.width = 300
    cam_prop.height = 300 
    cam_position = gymapi.Vec3(0.0, 0.3, 0.001) #x ngang, y cao, z nhin xa
    cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
    cam_handles = []
    cam_props = []
    for i in range(props['num_envs']):
        cam_handle = gym.create_camera_sensor(envs[i], cam_prop)
        gym.set_camera_location(cam_handle, envs[i], cam_position, cam_target)
        cam_props.append(cam_prop)
        cam_handles.append(cam_handle)
    return cam_handles,cam_props

def get_partial_point_cloud(gym, sim, env, cam_handles, cam_props, min_z = 0.005, visualization=False):
        cam_width = cam_props.width
        cam_height = cam_props.height
        # Render all of the image sensors only when we need their output here
        # rather than every frame.
        gym.render_all_camera_sensors(sim)

        points = []
        # print("Converting Depth images to point clouds. Have patience...")

        depth_buffer = gym.get_camera_image(sim, env, cam_handles, gymapi.IMAGE_DEPTH)
        seg_buffer = gym.get_camera_image(sim, env, cam_handles, gymapi.IMAGE_SEGMENTATION)


        # Get the camera view matrix and invert it to transform points from camera to world
        # space
        
        vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, cam_handles)))

        # Get the camera projection matrix and get the necessary scaling
        # coefficients for deprojection
        proj = gym.get_camera_proj_matrix(sim, env, cam_handles)
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



# ==================== FORCE CONTROL RELATED THINGS====================================
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
        Kp =  0.25
        min_torque = -0.005
        # torque_des[0] -= min(total_F_err * Kp, 3 * Kp)
        # torque_des[0] = min(min_torque, torque_des[0])
        torque_des[1] -= min(total_F_err * Kp, 3 * Kp)
        torque_des[1] = min(min_torque, torque_des[1])
        # torque_des[2] -= min(total_F_err * Kp, 3 * Kp)
        # torque_des[2] = min(min_torque, torque_des[2])
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
