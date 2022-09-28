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

def extract_nodal_force(gym, sim, particle_states):
    contacts = gym.get_soft_contacts(sim)
    gym.refresh_particle_state_tensor(sim)
    num_envs = gym.get_env_count(sim)
    num_particles = len(particle_states)
    num_particles_per_env = int(num_particles /  num_envs)

    forces_on_nodes = np.zeros((num_particles,3))
    nodal_forces = np.zeros((num_envs,num_particles_per_env,3))
    contact_indexes_all = np.zeros((num_particles))
    contact_indexes_per_env = np.zeros((num_envs,num_particles_per_env))
    for contact in contacts:
        rigid_body_index = contact[4]
        contact_normal = np.array([*contact[6]])
        contact_force_mag = contact[7]
        env_index = rigid_body_index // 3
        force_vec = contact_force_mag * contact_normal
        for node, bary in zip(contact[2] , contact[3]):
            forces_on_nodes[node] += bary * np.abs(force_vec)
            contact_indexes_all[node] = 1
    for idx, forces_on_node in enumerate(forces_on_nodes):
        nodal_force = forces_on_nodes[idx,:3]
        env_index = idx // num_particles_per_env
        local_particle_index = idx % num_particles_per_env
        nodal_forces[env_index][local_particle_index] = nodal_force

        contact_indexes = contact_indexes_all[idx]
        contact_indexes_per_env[env_index][local_particle_index] = contact_indexes

    return nodal_forces, contact_indexes_per_env

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


def extract_elem_stresses(gym, sim, envs):
    """Extract the element-wise von Mises stresses on the BioTac from each environment."""

    (_, stresses) = gym.get_sim_tetrahedra(sim)
    num_envs = gym.get_env_count(sim)
    num_tets = len(stresses)
    num_tets_per_env = int(num_tets / num_envs)
    stresses_von_mises = np.zeros((num_envs, num_tets_per_env))

    for env_index, env in enumerate(envs):
        # Get tet range (start, count) for BioTac
        tet_range = gym.get_actor_tetrahedra_range(env, 0, 0)

        # Compute and store von Mises stress for each tet
        # TODO: Vectorize for speed
        for global_tet_index in range(tet_range.start, tet_range.start + tet_range.count):
            stress = stresses[global_tet_index]
            stress = np.matrix([(stress.x.x, stress.y.x, stress.z.x),
                                (stress.x.y, stress.y.y, stress.z.y),
                                (stress.x.z, stress.y.z, stress.z.z)])
            stress_von_mises = np.sqrt(0.5 * \
                                       ((stress[0, 0] - stress[1, 1]) ** 2 \
                                      + (stress[1, 1] - stress[2, 2]) ** 2 \
                                      + (stress[2, 2] - stress[0, 0]) ** 2 \
                                      + 6 * (stress[1, 2] ** 2 + stress[2, 0] ** 2 + stress[0, 1] ** 2)))
            local_tet_index = global_tet_index % num_tets_per_env
            stresses_von_mises[env_index][local_tet_index] = stress_von_mises
    
    return stresses_von_mises

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
                                        
            hf.create_dataset("normal_forces_on_nodes",
                              (num_envs, sponge_fsms[0].state_tensor_length,3),
                              maxshape=(None, sponge_fsms[0].state_tensor_length,3))

            hf.create_dataset("contact_indexes",
                              (num_envs, sponge_fsms[0].state_tensor_length),
                              maxshape=(None, sponge_fsms[0].state_tensor_length))                  
            # Input of the system, contact point location and orientation of sponge, and target force
            hf.create_dataset("press_locations",
                              (num_envs, 1,3,),
                              maxshape=(None, 1,3))
            
            hf.create_dataset("gripper_ori", 
                              (num_envs, 1,3,),
                              maxshape=(None, 1,3))

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


                normal_forces_on_nodes_dset = hf['normal_forces_on_nodes']
                normal_forces_on_nodes_dset[i, :, :] = sponge_fsm.normal_forces_on_nodes

                contact_indexes_dset = hf['contact_indexes']
                contact_indexes_dset[i,:] = sponge_fsm.contact_indexes
                ######################

                press_location_dset = hf['press_locations']
                press_location_dset[i, :, :] = sponge_fsm.press_locations

                gripper_ori_dset = hf['gripper_ori']
                gripper_ori_dset[i, :, :] = sponge_fsm.gripper_ori

                pressed_forces_dset = hf['pressed_forces']
                pressed_forces_dset[i] = sponge_fsm.pressed_forces

                action_success_dset = hf['action_success']
                action_success_dset[i] = sponge_fsm.action_success

        hf.close()
