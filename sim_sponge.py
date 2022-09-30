# Copyright (c) 2021 NVIDIA Corporation

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

"""
Simulate indentations of the BioTac sensor with 8 different indenters for a sample trajectory.

Extracts net force vectors, nodal coordinates, and optionally, element-wise stresses for the BioTac.
"""
from ast import arg
import re
import argparse
import copy
import fileinput
import h5py
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from shutil import copyfile
import time
import random
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
import open3d
import torch
from utils import sim_utils
from utils import data_utils
from utils import spongefsm
from utils import open3d_utils
import numpy as np
from distutils.util import strtobool

np.set_printoptions(threshold=np.inf)
# Set target indenter pose
RESULTS_DIR = "results/"
RESULTS_STORAGE_TAG = "_all"
def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'boolean', strtobool) #to help deal with problem bool("False")=True
    parser.add_argument('--object', required=True, help="Name of object")
    parser.add_argument('--density', default=100, type=float, help="Density")
    parser.add_argument('--youngs', default=1000, type=float, help="Elastic modulus of BioTac [Pa]")
    parser.add_argument('--poiss_ratio', default=0.45, type=float, help="Poisson's ratio of BioTac")
    parser.add_argument('--extract_stress', default=False, type='boolean', help='Extract stress at each indentation step (will reduce simulation speed)')
    parser.add_argument('--num_envs', default=1, type=int, help='Number of envs')
    parser.add_argument('--random_force', default=True, type='boolean', help='Random the rotation of the gripper')
    parser.add_argument('--random_rotation', default=True, type='boolean', help='Random the rotation of the gripper')
    parser.add_argument('--run_headless', default=False, type='boolean', help='Run the simulator headless mode, i.e., no graphical interface')
    parser.add_argument('--write_results', default=True, type='boolean', help='Export results to H5')

    args = parser.parse_args()
    target_name = [args.object]

    gym = gymapi.acquire_gym()
    state = "capture_target_pc"
    # Define sim parameters and create Sim object
    sim = create_sim(gym=gym)
    # Define and load assets
    asset_options = set_asset_options()
    biotac_urdf_dir = os.path.join('urdf', 'biotac', 'shell_core')
    sim_utils.set_biotac_matl_props(base_dir=biotac_urdf_dir,
                          elast_mod=args.youngs,
                          poiss_ratio=args.poiss_ratio,
                          density=args.density)
    asset_handles_biotac = load_assets(gym=gym,
                                       sim=sim,
                                       base_dir=biotac_urdf_dir,
                                       objects=['soft_body'],
                                       options=asset_options)

    target_list  = target_name * args.num_envs
    asset_options.thickness = 0.006 #0.005 
    asset_handles_indenters = load_assets(gym=gym,
                                          sim=sim,
                                          base_dir=os.path.join('urdf', 'indenters'),
                                          objects=target_list,
                                          options=asset_options)
    # Define and create scene
    scene_props = set_scene_props(num_envs=len(target_list))
    env_handles, actor_handles_biotacs, actor_handles_indenters, gripper_rotation_with_random_z,press_loc  = create_scene(gym=gym, 
                                                                               sim=sim, 
                                                                               object_name=target_name[0],
                                                                               props=scene_props,
                                                                               assets_biotac=asset_handles_biotac,
                                                                               assets_indenters=asset_handles_indenters,
                                                                               random_rotation = args.random_rotation)  
    if not args.run_headless:
        viewer, axes_geom = create_viewer(gym=gym, 
                                        sim=sim)
    elif args.run_headless:
        viewer = None
        axes_geom = None
    # Setup cameras
    cam_handles,cam_props = sim_utils.setup_cam(gym, env_handles, scene_props)
    # Define controller for indenters
    # set_ctrl_props(gym=gym,
    #                envs=env_handles,
    #                indenters=actor_handles_biotacs)    
    # Run simulation loop
    state = 'init'
    sponge_fsms = []
    print("Running for object: ", "-- Youngs: ", args.youngs, "--random_force: ",args.random_force, "--random rotation: ", args.random_rotation)
    for i in range(len(env_handles)):
        if args.random_force:       
            sponge_fsm = spongefsm.SpongeFsm(gym=gym, 
                                sim=sim, 
                                envs=env_handles, 
                                env_id = i,
                                cam_handles=cam_handles[i],
                                cam_props = cam_props[i],
                                sponge_actor=actor_handles_biotacs,
                                viewer=viewer,
                                axes=axes_geom,
                                state=state,
                                target_object_name=target_name[0],
                                gripper_ori=gripper_rotation_with_random_z[i],
                                press_loc=press_loc[i],
                                press_force=np.array([0.0,random.randint(1, 25),0.0]),
                                show_contacts=True)
        elif not args.random_force:
            sponge_fsm = spongefsm.SpongeFsm(gym=gym, 
                                sim=sim, 
                                envs=env_handles, 
                                env_id = i,
                                cam_handles=cam_handles[i],
                                cam_props = cam_props[i],
                                sponge_actor=actor_handles_biotacs,
                                viewer=viewer,
                                axes=axes_geom,
                                state=state,
                                target_object_name=target_name[0],
                                gripper_ori=gripper_rotation_with_random_z[i],
                                press_loc=press_loc[i],
                                press_force=np.array([0.0,5.0,0.0]),
                                show_contacts=True)
        sponge_fsms.append(sponge_fsm)
    all_done = False
    result = []
    while not all_done:
        if not args.run_headless:
            pass

        all_done = all(sponge_fsms[i].state == 'done'
                       for i in range(len(env_handles)))
        
        gym.refresh_particle_state_tensor(sim)
        for i in range(len(env_handles)):
            if sponge_fsms[i].state != "done":
                sponge_fsms[i].run_state_machine()
            print("State env ", str(i), "---- state: ",sponge_fsms[i].state, "----- force: ", sponge_fsms[i].F_des[1])

        # Run simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        # Visualize motion and deformation
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        gym.clear_lines(viewer)

        gym.step_graphics(sim)
        # render the camera sensors

        if not args.run_headless:
            gym.draw_viewer(viewer, sim, True)
    # Clean up
    if not args.run_headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Finished the simulation")

    # Store data     
    if args.write_results:
        regex = re.compile(r'\d+')
        num_iter = 0
        object_name = target_name[0]
        big_folder_name = object_name + RESULTS_STORAGE_TAG
        small_folder_name = object_name + "_" + str(args.youngs)
        file_idxes = []
        os.makedirs(os.path.join(RESULTS_DIR, big_folder_name, small_folder_name), exist_ok=True)
        if os.listdir(os.path.join(RESULTS_DIR, big_folder_name, small_folder_name)):
            for file in os.listdir(os.path.join(RESULTS_DIR, big_folder_name, small_folder_name)):
                file_idx = int(regex.search(file).group(0)) #extract number from file name
                file_idxes.append(file_idx)
            num_iter = max(file_idxes)+1
        else:
            num_iter = 0
        object_file_name = object_name +  "_iter"+"_"+str(num_iter)+".h5"
        h5_file_path = os.path.join(RESULTS_DIR, big_folder_name, small_folder_name, object_file_name)
        data_utils.write_metrics_to_h5(num_envs=len(env_handles),h5_file_path=h5_file_path,sponge_fsms=sponge_fsms)
    

def create_scene(gym, sim, object_name, props, assets_biotac, assets_indenters, biotac_offset=0.2, indenter_offset=0.05,random_rotation=False):
    """Create a scene (i.e., ground plane, environments, BioTac actors, and indenter actors)."""
    z_angle = 0
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    env_handles = []
    actor_handles_biotacs = []
    actor_handles_indenters = []
    target_object_pcd_file = "/home/trannguyenle/RemoteWorkingStation/ros_workspaces/IsaacGym/isaacgym/python/robot_sponge/target_object_pc/"+object_name+".pcd"
    if os.path.exists(target_object_pcd_file):
        sampled_points_real, sampled_points_approach , sampled_points_normals, sampled_points_normals_euler = open3d_utils.sample_points_and_normals_from_pcd(target_object_pcd_file,props['num_envs'])
    else:
        print("Point cloud of this object does not exist")
        sampled_points_real = np.tile(np.array([0.0,0.0,0.0]),(props['num_envs'],3))
        sampled_points_approach = np.tile(np.array([0.0,0.0,0.0]),(props['num_envs'],3))
        sampled_points_normals_euler =  np.tile(np.array([0.0,0.0,0.0]),(props['num_envs'],3))
    gripper_rotation_with_random_z = []
    for i in range(props['num_envs']):
        env_handle = gym.create_env(sim, props['lower'], props['upper'], props['per_row'])
        env_handles.append(env_handle)

        pose = gymapi.Transform()
        collision_group = i
        collision_filter = 0
        # pose.p = gymapi.Vec3(sampled_points[i][0], biotac_offset, sampled_points[i][2])
        pose.p = gymapi.Vec3(sampled_points_approach[i][0], sampled_points_approach[i][1], sampled_points_approach[i][2])

        if random_rotation:
            z_angle = random.randint(-90, 90)
        elif not random_rotation:
            z_angle = 0    
        gripper_rotation_per_env = np.array([sampled_points_normals_euler[i][0],z_angle,sampled_points_normals_euler[i][2]])
        r = R.from_euler('XYZ',gripper_rotation_per_env, degrees=True)
        gripper_rotation_with_random_z.append(gripper_rotation_per_env)
        quat = r.as_quat()
        pose.r = gymapi.Quat(*quat)
        actor_handle_biotac = gym.create_actor(env_handle, assets_biotac[0], pose, f"biotac_{i}", collision_group, collision_filter)
        actor_handles_biotacs.append(actor_handle_biotac)
        pose.p = gymapi.Vec3(0.0, indenter_offset, 0.0)
        r = R.from_euler('XYZ', [0, 0, 0], degrees=True)
        quat = r.as_quat()
        pose.r = gymapi.Quat(*quat)
        actor_handle_indenter = gym.create_actor(env_handle, assets_indenters[i], pose, f"indenter_{i}", collision_group, collision_filter,segmentationId=10)
        actor_handles_indenters.append(actor_handle_indenter)
    return env_handles, actor_handles_biotacs, actor_handles_indenters, gripper_rotation_with_random_z,sampled_points_real

def create_sim(gym):
    """Set the simulation parameters and create a Sim object."""

    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.  # Control frequency
    sim_params.substeps = 4  # Physics simulation frequency (multiplier)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)

    # enable Von-Mises stress visualization
    sim_params.stress_visualization = True
    sim_params.stress_visualization_min = 0.0
    sim_params.stress_visualization_max = 1.e+5

    sim_params.flex.solver_type = 5  # PCR (GPU, global)
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 50
    sim_params.flex.relaxation = 0.7
    sim_params.flex.warm_start = 0.1
    sim_params.flex.shape_collision_distance = 5e-4
    sim_params.flex.contact_regularization = 1.0e-6
    sim_params.flex.shape_collision_margin = 1.0e-4  
  
    sim_params.flex.deterministic_mode = True    

    sim_params.flex.friction_mode = 2  # Friction about all 3 axes (including torsional)
    sim_params.flex.dynamic_friction = 1000
    sim_params.flex.static_friction = 1000

    gpu_physics = 0
    gpu_render = 0
    sim = gym.create_sim(gpu_physics, gpu_render, sim_type, sim_params)

    return sim

def create_viewer(gym, sim):
    """Create viewer and axes objects."""

    camera_props = gymapi.CameraProperties()
    # camera_props.horizontal_fov = 5.0
    camera_props.width = 1920
    camera_props.height = 1080
    viewer = gym.create_viewer(sim, camera_props)
    # camera_pos = gymapi.Vec3(0.075, 3.0, 10.0)
    # camera_target = gymapi.Vec3(0.075, 0.0, 0.0)
    camera_pos= gymapi.Vec3(0.5, 0.18, 0.5) #x ngang, y cao, z nhin xa
    camera_target = gymapi.Vec3(0.5, 0.00, 0.0)
    gym.viewer_camera_look_at(viewer, None, camera_pos, camera_target)

    axes_geom = gymutil.AxesGeometry(0.1)

    return viewer, axes_geom


def get_pose_and_draw(gym, env, viewer, axes, indenter):
    """Draw the pose of an indenter."""

    indenter_pose = gym.get_actor_rigid_body_states(env, indenter, gymapi.STATE_POS)['pose']
    pose_to_draw = gymapi.Transform.from_buffer(indenter_pose)
    gymutil.draw_lines(axes, gym, viewer, env, pose_to_draw)


def load_assets(gym, sim, base_dir, objects, options, fix=True, gravity=False):
    """Load assets from specified URDF files."""
    handles = []
    # options = gymapi.AssetOptions()
    for obj in objects:
        options.fix_base_link = True if fix else False
        options.disable_gravity = True if not gravity else False
        handle = gym.load_asset(sim, base_dir, obj + '.urdf', options)
        handles.append(handle)
    
    return handles

def reset_biotac_state(gym, sim, biotac_state):
    """Reset the BioTac particle states to their initial states."""

    gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(biotac_state))

def set_asset_options():
    """Set asset options common to all assets."""

    options = gymapi.AssetOptions()
    options.flip_visual_attachments = False
    options.armature = 0.0
    options.thickness = 0.0005
    options.linear_damping = 0.0
    options.angular_damping = 0.0
    options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
    options.min_particle_mass = 1e-20

    return options


def set_scene_props(num_envs, env_dim=0.25):
    """Set the scene and environment properties."""

    # envs_per_row = int(np.ceil(np.sqrt(num_envs)))
    envs_per_row = int(num_envs)
    env_lower = gymapi.Vec3(-env_dim, 0, -env_dim)
    env_upper = gymapi.Vec3(env_dim, env_dim, env_dim)
    scene_props = {'num_envs': num_envs,
                   'per_row': envs_per_row,
                   'lower': env_lower,
                   'upper': env_upper}

    return scene_props

if __name__ == "__main__":
    main()