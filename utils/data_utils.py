import numpy as np
import os
import h5py 
from isaacgym import gymapi

def extract_nodal_coords(gym, sim, particle_states):
    """Extract the nodal coordinates (state) for the sponge from each environment."""
 
    gym.refresh_particle_state_tensor(sim)
    num_envs = gym.get_env_count(sim)
    num_particles = len(particle_states)
    num_particles_per_env = int(num_particles /  num_envs)
    nodal_coords = np.zeros((num_envs, num_particles_per_env, 3))
    for global_particle_index, particle_state in enumerate(particle_states):
        pos = particle_state[:3]
        env_index = global_particle_index // num_particles_per_env
        local_particle_index = global_particle_index % num_particles_per_env
        nodal_coords[env_index][local_particle_index] = pos.numpy()
    
    return nodal_coords

def extract_net_forces(gym, sim):
    """Extract the net force vector on the BioTac for each environment."""

    contacts = gym.get_soft_contacts(sim)
    num_envs = gym.get_env_count(sim)
    net_force_vecs = np.zeros((num_envs, 3))
    for contact in contacts:
        rigid_body_index = contact[4]
        contact_normal = np.array([*contact[6]])
        contact_force_mag = contact[7]
        env_index = rigid_body_index // 3
        force_vec = contact_force_mag * contact_normal
        net_force_vecs[env_index] += force_vec
    net_force_vecs = -net_force_vecs
    return net_force_vecs

def write_metrics_to_h5(num_envs,h5_file_path, sponge_fsms):
    """Write metrics and features to h5 result file."""
    h5_file_name = h5_file_path
    print("Trying to write to", h5_file_name)

    if not os.path.exists(h5_file_name):
        print("Writing to new file", h5_file_name)
        os.makedirs(os.path.dirname(h5_file_name), exist_ok=True)
        with h5py.File(h5_file_name, "w") as hf:

            # Position and force on each node of sponge when reaching the desired force
            hf.create_dataset("sponge_position_at_force",
                              (num_envs, sponge_fsms[0].state_tensor_length, 3),
                              maxshape=(None,
                                        sponge_fsms[0].state_tensor_length, 3))
                                        
            # hf.create_dataset("normal_forces_on_nodes",
            #                   (num_envs, sponge_fsms[0].state_tensor_length),
            #                   maxshape=(None, sponge_fsms[0].state_tensor_length))
            # Input of the system, contact point location and orientation of sponge, and target force
            hf.create_dataset("contact_locations",
                              (num_envs, 1,3,),
                              maxshape=(None, 1,3))
            
            hf.create_dataset("z_angle", (num_envs,),
                              maxshape=(None,))

            # Desired force setpoints

            hf.create_dataset("pressed_forces",
                              (num_envs,),
                              maxshape=(None,))
            # Status 
            hf.create_dataset("action_success", (num_envs,), maxshape=(None,))


        with h5py.File(h5_file_name, 'a') as hf:
            for i, sponge_fsm in enumerate(sponge_fsms):

                sponge_position_at_force_dset = hf['sponge_position_at_force']
                sponge_position_at_force_dset[i, :, :] = sponge_fsm.sponge_position_at_force


                # normal_forces_on_nodes_dset = hf['normal_forces_on_nodes']
                # normal_forces_on_nodes_dset[i, :, :] = panda_fsm.normal_forces_on_nodes

                ######################

                contact_location_dset = hf['contact_locations']
                contact_location_dset[i, :, :] = sponge_fsm.contact_locations

                z_angle_dset = hf['z_angle']
                z_angle_dset[i] = sponge_fsm.z_angle

                pressed_forces_dset = hf['pressed_forces']
                pressed_forces_dset[i] = sponge_fsm.pressed_forces

                action_success_dset = hf['action_success']
                action_success_dset[i] = sponge_fsm.action_success

        hf.close()
