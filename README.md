## Training deform contactnet models
# Pointnet++: 
`python train2.py --model deform_contactnet_pointnet2 --use_normals --log_dir deform_contactnet_pointnet2 --batch_size=8 --epoch=10 --use_wandb --gpu=1`
# Pointnet: 
`python train.py --model deform_contactnet_pointnet --use_normals --log_dir deform_contactnet_pointnet --batch_size=9 --use_wandb --epoch=10 --gpu=0` 

## Running sequence planning
`python planning.py --object=bowl_ycb --use_vis`

## Sponge Sim
The script provides a simple example of how to import the Sponge assets into NVIDIA Isaac Gym, launch an FEM simulation with multiple objects across multiple parallel environments, and extract useful features (net forces, nodal coordinates, and element-wise stresses).

## Installation:
- Clone repo
- Download [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym/download)
    - Follow provided instructions to create and activate `rlgpu` Conda environment for Isaac Gym
- Install `h5py` package via Conda

## Usage:
- Execute `sim_sponge.py --object="target_object" --num_envs=1 --youngs=1000`
    - See code for available command line switches
- View `results/object/object_youngs`
    - File structure is `action_success / contact_indexes / gripper_ori / normal_forces_on_nodes / press_locations / press_forces /sponge_position_at_force`

## FAQ:
- Error: `cannot open shared object file`
    - Add `/home/username/anaconda3/envs/rlgpu/lib` to `LD_LIBRARY_PATH`
- Warning: `Degenerate or inverted tet`
    - Safely ignore

## Additional:
- For questions related to NVIDIA Isaac Gym, see [official forum](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-gym/322)
