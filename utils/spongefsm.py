import argparse
import copy
import fileinput
import h5py
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from shutil import copyfile
import timeit
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
import open3d
import torch
from utils import sim_utils
from utils import data_utils
from utils import open3d_utils

import numpy as np
from os.path import exists

class SpongeFsm:
    """FSM for control of sponge the pushing test."""
    def __init__(self, gym, sim, envs, env_id, cam_handles,
                 cam_props, sponge_actor, viewer, axes,state,target_object_name,gripper_ori,press_loc,press_force,
                 show_contacts=False):
        """Initialize attributes of grasp evaluation FSM.

        Args:
            gym (gymapi.Gym): Gym object.
            sim (gymapi.Sim): Simulation object.
            envs (list of gymapi.Env): List of all environments.
            env_id (int): Index of environment from env_handles.
            cam_handles : Handle of cameras.
            cam_props: Properties of cameras.
            sponge_actor: Sponge actor.
            viewer (gymapi.Viewer): Graphical display object.
            state: current state of the system {'init', 'approach', 'press', 'done'}.
            target_object_name (str): Name of object to be pressed.
            gripper_ori: Orientation of the sponge in euler angle (y axis up).
            press_loc: press location
            show_contacts: contacts visualization.
        """
        # Simulation handles
        self.started = False
        self.gym = gym
        self.sim = sim
        self.envs = envs
        self.env_id = env_id
        self.env = self.envs[self.env_id]
        self.viewer = viewer
        # Camera sensor handles
        self.cam_handles = cam_handles
        self.cam_props = cam_props
        # Sponge actor
        self.sponge_actor = sponge_actor[0]

        # Sim sponge arguments
        self.num_envs = len(self.envs)
        self.moving_average = []
        self.F_history = []
        self.filtered_forces =[]
        self.F_max_window_size = 300
        self.torque_des = [0, -0.1, -0.1]           
        self.F_des = np.asarray(press_force, dtype=np.float32)
        self.state = state
        self.target_object_name = target_object_name
        self.pressed_forces = 0
        self.gripper_ori = gripper_ori
        self.press_locations_buffer= press_loc
        self.action_success = False
        self.loop_start = None
        self.contacts = np.array([])
        self.show_contacts = show_contacts
    def run_state_machine(self):
        """Run state machine for running pressing tests."""
        results = []
        allpcs = []
        ctr_pos = 0.0
        target_object_pc_file = "/home/trannguyenle/RemoteWorkingStation/ros_workspaces/IsaacGym/isaacgym/python/robot_sponge/target_object_pc/"+str(self.target_object_name)+".pcd"
        if self.state == "init":
            self.started = True
            self.state = "capture_target_pc"
        # Get particle state tensor and convert to PyTorch tensor
        self.particle_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_particle_state_tensor(self.sim))
        self.gym.refresh_particle_state_tensor(self.sim)
        sponge_state_init = copy.deepcopy(self.particle_state_tensor)
        sponge_state_init_array = self.particle_state_tensor.numpy()[:, :3].astype('float32')  
        self.state_tensor_length = int(sponge_state_init_array.shape[0]/self.num_envs)

        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(np.array(sponge_state_init_array))
        # open3d.io.write_point_cloud("/home/trannguyenle/test.pcd", pcd)
        if self.state == "failed":
            self.action_success = False
            self.sponge_position_at_force = np.zeros((self.state_tensor_length,3))
            self.normal_forces_on_nodes = np.zeros((self.state_tensor_length,3))
            self.contact_indexes = np.zeros((self.state_tensor_length))
            self.press_locations = np.zeros((1,3))
            self.pressed_forces = 0
            self.state = "done"

        if self.state == "capture_target_pc":
            if not os.path.exists(target_object_pc_file):
                object_pc = sim_utils.get_partial_point_cloud(self.gym, self.sim, self.env, self.cam_handles, self.cam_props, visualization=False)
                target_pcd = open3d.geometry.PointCloud()
                target_pcd.points = open3d.utility.Vector3dVector(np.array(object_pc))
                open3d.io.write_point_cloud("/home/trannguyenle/RemoteWorkingStation/ros_workspaces/IsaacGym/isaacgym/python/robot_sponge/target_object_pc/"+str(self.target_object_name)+".pcd", target_pcd)
            else:
                print("Target object point cloud existed.",end="\r")
            self.state = "approach"
            self.approach_start = timeit.default_timer()


        elif self.state == "approach":
            targetobject_dof_state = self.gym.get_actor_dof_states(self.env, self.sponge_actor, gymapi.STATE_ALL)
            targetobject_dof_pos = targetobject_dof_state['pos']
            F_curr_all_env = data_utils.extract_net_forces(self.gym,self.sim)  
            F_curr = F_curr_all_env[self.env_id]
            if np.abs(F_curr[1]) < 1 and (timeit.default_timer() - self.approach_start) < 45:
                if self.target_object_name in ["square_pot"]:
                    vel_des = -0.06
                else:
                    vel_des = -0.03
                # if targetobject_dof_pos < -0.34:
                #     vel_des = -0.02  

                dof_props = self.gym.get_actor_dof_properties(self.env, self.sponge_actor)
                dof_props['driveMode'][0] = gymapi.DOF_MODE_VEL

                self.gym.set_actor_dof_properties(self.env,self.sponge_actor,dof_props)

                self.gym.set_actor_dof_velocity_targets(self.env,self.sponge_actor,vel_des)
            elif np.abs(F_curr[1]) > 1:
                self.loop_start = timeit.default_timer()
                self.state = "press"
                self.gym.draw_env_rigid_contacts(self.viewer, self.env, gymapi.Vec3(1.0, 0.5, 0.0), 0.05, True)
                self.gym.draw_env_soft_contacts(self.viewer, self.env, gymapi.Vec3(0.6, 0.0, 0.6), 0.05, False, True)
            elif (timeit.default_timer() - self.approach_start) > 45:
                self.state = "failed"

        elif self.state == "press":
            # =================================================
            # Process finger grasp forces with LP filter and moving average
            if (timeit.default_timer() - self.loop_start) < 30:
                self.F_curr = data_utils.extract_net_forces(self.gym,self.sim)  
                self.F_history.append(np.sum(self.F_curr[1:]))
                window = self.F_history[-self.F_max_window_size:]
                filtered_force, avg_of_filter = 0.0, 0.0
                if len(window) > 10:
                    filtered_force = sim_utils.butter_lowpass_filter(window)[-1]
                self.filtered_forces.append(filtered_force)
                if len(self.F_history) > 0:
                    avg_of_filter = np.mean(self.filtered_forces[-30:])
                self.moving_average.append(avg_of_filter)

                self.torque_des_force, _, _ = sim_utils.get_force_based_torque(self.F_des, self.F_curr,self.moving_average,self.torque_des)
                # Change mode of the fingers to torque control
                dof_props = self.gym.get_actor_dof_properties(self.env, self.sponge_actor)
                dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
                self.gym.set_actor_dof_properties(self.env,self.sponge_actor,dof_props)
                self.gym.apply_dof_effort(self.env,
                                        self.sponge_actor,
                                        self.torque_des_force[1])
                # print("Error evn ",int(self.env_id), " ---- ",np.abs(np.abs(self.torque_des_force[1])-np.abs(self.F_des[1])))
                # if  np.abs(np.abs(self.torque_des_force[1])-np.abs(self.F_des[1])) < np.abs(0.05 * self.F_des[1]):
                if self.show_contacts:
                    self.gym.draw_env_rigid_contacts(self.viewer, self.env, gymapi.Vec3(1.0, 0.5, 0.0), 0.05, True)
                    self.gym.draw_env_soft_contacts(self.viewer, self.env, gymapi.Vec3(0.6, 0.0, 0.6), 0.05, False, True)

                print("Error evn ",int(self.env_id), " ---- ",np.abs(np.abs(self.F_curr[self.env_id][1])-np.abs(self.F_des[1]))," **** ",end=" ")
                if  np.abs(np.abs(self.F_curr[self.env_id][1])-np.abs(self.F_des[1])) < np.abs(0.1 * self.F_des[1]):
                    self.pressed_forces = data_utils.extract_net_forces(self.gym,self.sim)[self.env_id][1] 
                    self.state = "capture_final_state"
            else: 
                self.state = "failed"

        elif self.state == "capture_final_state":
            deformed_state = gymtorch.wrap_tensor(self.gym.acquire_particle_state_tensor(self.sim))
            self.gym.refresh_particle_state_tensor(self.sim)
            self.deformed_state_all_envs = data_utils.extract_nodal_coords(gym=self.gym, 
                                                   sim=self.sim, 
                                                   particle_states=deformed_state) 
            self.sponge_position_at_force = self.deformed_state_all_envs[self.env_id]
            self.contacts = self.gym.get_soft_contacts(self.sim)    
            normal_forces_on_nodes, contact_indexes = data_utils.extract_nodal_force(self.gym,self.sim,deformed_state)
            self.normal_forces_on_nodes = normal_forces_on_nodes[self.env_id]
            self.contact_indexes = contact_indexes[self.env_id]
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(np.array(self.deformed_state_all_envs[self.env_id]))
            target_pcd = open3d.io.read_point_cloud("/home/trannguyenle/RemoteWorkingStation/ros_workspaces/IsaacGym/isaacgym/python/robot_sponge/target_object_pc/"+str(self.target_object_name)+".pcd")
            open3d_utils.visualize_pc_open3d(target_pcd+pcd)
            self.press_locations = self.press_locations_buffer
            self.action_success = True
            self.state = "done"                
        return

   

