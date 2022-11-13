from numpy import real
import time
import click
import socket
import multiprocessing as mp
from chester.run_exp import run_experiment_lite, VariantGenerator
from haptic.shared.eval_script import run_task
from haptic.pointNet_geo.launchers.launch_train_contact import (
    get_fp_mlp_list, get_sa_mlp_list, get_linear_mlp_list, get_fp_k, get_sa_radius, get_sa_ratio
)

def get_plot_mode(real_world):
    if real_world:
        return 'matplotlib'
    else:
        return 'pyrender'

def get_feature_num(add_gripper_vel, add_cloth_vel, num_temporal_step):
    base = 3 + (num_temporal_step) * 3
    if add_gripper_vel:
        base += 3
    if add_cloth_vel:
        base += 3
    return base

def get_dir(mode, real_world):
    print("in get_dir, mode {} real_world {}".format(mode, real_world))
    if mode == 'dressing':
        if not real_world:
            # IROS one
            return '2022-0211-testset-dressing'
            # return '2022-0407-dressing-smaller-epsilon-test'
        else:
            return '2022-0216-dressing-bad-traj-1-fix-gripper'
            

            # return '2022-0216-dressing-bad-traj-1-segment-tuned'
            # return '2022-0216-dressing-traj-6-segment-tuned'
            # return '2022-0218-dressing-traj-4-segment-tuned'
            # return '2022-0218-dressing-traj-7-segment-tuned'

            # return '2022-0216-dressing-traj-6-voxel-in-camera'
            # return '2022-0216-dressing-bad-traj-1-voxel-in-camera'
            # return '2022-0216-dressing-bad-traj-2-voxel-in-camera'
            # return '2022-0218-dressing-traj-4-voxel-in-camera'

    elif mode == 'bathing':
        if not real_world:
            # IROS one
            return '2022-0211-testset-bathing'
            # return '2022-0407-bathing-smaller-epsilon-test'
        else:
            # return '2022-0205-bathing-6'
            return '2022-0205-bathing-good'
    elif mode == 'plate-clean':
        if not real_world:
            # return '2022-0119-plate-clean-no-voxelization-fix-loofah'
            return '2022-0211-testset-plate-clean' # IROS submission dataset
            # return '2022-0322-plate-clean-test-smaller-epsilon'
        else:
            return '2022-0201-plate-small-cloth' ### after IROS, small voxel size
            # return '2022-0126-plate-real-world'
            # return '2022-0201-plate-small-cloth' ### final one to be used in IROS paper
            # return '2022-0201-plate-small-cloth-higher'
    elif mode == 'primitive':
        # return '2022-0119-towel-multile-object'
        return 'IROS-primitive-testset'

def get_force_model(mode, use_gt_particle):
    if not use_gt_particle:
        if mode == 'bathing':
            return 'data/seuss/2022-0119-bathing-clean-correct/2022-0119-bathing-clean-correct_2022_01_19_02_07_30_0001/checkpoints/best_model.pth'
        elif mode == 'dressing':
            return 'data/seuss/2022-0120-pointnet-dressing/2022-0120-pointnet-dressing_2022_01_20_00_20_43_0001/checkpoints/best_model.pth'
        elif mode == 'plate-clean':
            return 'data/seuss/2022-0119-pointnet-plate-clean-correct/2022-0119-pointnet-plate-clean-correct_2022_01_19_01_44_50_0001/checkpoints/best_model.pth'
    else:
        if mode == 'bathing':
            return 'data/seuss/2022-0119-bathing-clean-correct/2022-0119-bathing-clean-correct_2022_01_19_02_07_30_0002/checkpoints/best_model.pth'
        elif mode == 'dressing':
            return 'data/seuss/2022-0119-dressing/2022-0119-dressing_2022_01_20_00_03_57_0002/checkpoints/best_model.pth'
        elif mode == 'plate-clean':
            return 'data/seuss/2022-0119-pointnet-plate-clean-correct/2022-0119-pointnet-plate-clean-correct_2022_01_19_01_44_50_0002/checkpoints/best_model.pth'

def get_force_model_more_data(mode, use_gt_particle):
    if not use_gt_particle:
        if mode == 'bathing':
            # IROS model
            # return 'data/seuss/2022-0209-pointnet-more-data-bathing/2022-0209-pointnet-more-data-bathing_2022_02_09_23_26_39_0001/checkpoints/best_model.pth'
            return 'data/seuss/2022-0323-pointnet-force-bathing-smaller-epsilon/2022-0323-pointnet-force-bathing-smaller-epsilon_2022_03_23_12_36_40_0001/checkpoints/best_model.pth'
        elif mode == 'dressing':
            # IROS model
            # return 'data/seuss/2022-0209-pointnet-more-data-dressing/2022-0209-pointnet-more-data-dressing_2022_02_09_12_11_17_0001/checkpoints/best_model.pth'
            return 'data/seuss/2022-0323-pointnet-force-dressing-smaller-epsilon/2022-0323-pointnet-force-dressing-smaller-epsilon_2022_03_23_12_41_58_0001/checkpoints/best_model.pth'
        elif mode == 'plate-clean':
            # large voxel size, used in IROS paper
            return 'data/seuss/2022-0209-pointnet-more-data-plate-clean/2022-0209-pointnet-more-data-plate-clean_2022_02_09_18_12_23_0001/checkpoints/best_model.pth'
            # small voxel size
            # return 'data/seuss/2022-0304-pointnet-force-plate-clean-small-voxel/2022-0304-pointnet-force-plate-clean-small-voxel_2022_03_06_15_23_37_0001/checkpoints/best_model.pth'
            # return 'data/seuss/2022-0313-pointnet-force-plate-clean-smaller-epsilon/2022-0313-pointnet-force-plate-clean-smaller-epsilon_2022_03_13_16_44_21_0001/checkpoints/best_model.pth'
        if mode == 'primitive':
            return 'data/seuss/2022-0123-pointnet-primitive/2022-0123-pointnet-primitive_2022_01_23_00_40_12_0001/checkpoints/best_model.pth'
    else:
        if mode == 'bathing':
            return 'data/seuss/2022-0209-pointnet-more-data-bathing/2022-0209-pointnet-more-data-bathing_2022_02_09_23_26_39_0002/checkpoints/best_model.pth'
        elif mode == 'dressing':
            return 'data/seuss/2022-0209-pointnet-more-data-dressing/2022-0209-pointnet-more-data-dressing_2022_02_09_12_11_17_0002/checkpoints/best_model.pth'
        elif mode == 'plate-clean':
            return 'data/seuss/2022-0209-pointnet-more-data-plate-clean/2022-0209-pointnet-more-data-plate-clean_2022_02_09_18_12_23_0002/checkpoints/best_model.pth'
        if mode == 'primitive':
            return 'data/seuss/2022-0123-pointnet-primitive/2022-0123-pointnet-primitive_2022_01_23_00_40_12_0002/checkpoints/best_model.pth'


def get_force_model_all_data(use_gt_particle):
    if not use_gt_particle:
        return 'data/seuss/2022-0208-pointnet-all-dataset/2022-0208-pointnet-all-dataset_2022_02_09_01_07_05_0001/checkpoints/best_model.pth'
    else:
        return 'data/seuss/2022-0208-pointnet-all-dataset/2022-0208-pointnet-all-dataset_2022_02_09_01_07_05_0002/checkpoints/best_model.pth'


def get_force_model_all_more_data(use_gt_particle):
    if not use_gt_particle:
        return 'data/seuss/2022-0213-pointnet-more-data-all/2022-0213-pointnet-more-data-all_2022_02_13_01_21_46_0001/checkpoints/best_model.pth'
    else:
        return 'data/seuss/2022-0213-pointnet-more-data-all/2022-0213-pointnet-more-data-all_2022_02_13_01_21_46_0002/checkpoints/best_model.pth'

def get_force_model_all_more_inverse_data(use_gt_particle):
    if not use_gt_particle:
        return 'data/seuss/2022-0215-pointnet-more-data-all-inverse-weight/2022-0215-pointnet-more-data-all-inverse-weight_2022_02_15_20_43_12_0001/checkpoints/best_model.pth'
    else:
        return ''


def get_contact_model(mode, use_gt_particle):
    if mode == 'bathing':
        # return 'data/seuss/2021-0120-pointnet-bathing-contact/2021-0120-pointnet-bathing-contact_2022_01_20_23_52_02_0001/checkpoints/best_model.pth'
        if not use_gt_particle:
            # IROS model
            # return 'data/seuss/2022-0213-pointnet-contact-bathing-more-data/2022-0213-pointnet-contact-bathing-more-data_2022_02_13_22_29_26_0001/checkpoints/best_model.pth'
            return 'data/seuss/2022-0323-pointnet-contact-bathing-smaller-epsilon/2022-0323-pointnet-contact-bathing-smaller-epsilon_2022_04_03_23_58_02_0001/checkpoints/best_model.pth'
        else:
            return 'data/seuss/2022-0213-pointnet-contact-bathing-more-data/2022-0213-pointnet-contact-bathing-more-data_2022_02_13_22_29_26_0002/checkpoints/best_model.pth'

    elif mode == 'dressing':
        if not use_gt_particle:
            # IROS
            # return 'data/seuss/2022-0120-pointnet-dressing-contact/2022-0120-pointnet-dressing-contact_2022_01_21_12_57_56_0001/checkpoints/best_model.pth'
            # return 'data/seuss/2022-0213-pointnet-contact-dressing-more-data/2022-0213-pointnet-contact-dressing-more-data_2022_02_13_22_27_02_0001/checkpoints/best_model.pth'
            return 'data/seuss/2022-0323-pointnet-contact-dressing-smaller-epsilon/2022-0323-pointnet-contact-dressing-smaller-epsilon_2022_03_23_12_41_42_0001/checkpoints/best_model.pth'
        else:
            return 'data/seuss/2022-0213-pointnet-contact-dressing-more-data/2022-0213-pointnet-contact-dressing-more-data_2022_02_13_22_27_02_0002/checkpoints/best_model.pth'
    elif mode == 'plate-clean':
        # return 'data/seuss/2021-0120-pointnet-plate-clean-contact/2021-0120-pointnet-plate-clean-contact_2022_01_20_23_52_44_0001/checkpoints/best_model.pth'
        if not use_gt_particle:
            # large voxel size, used in IROS paper
            return 'data/seuss/2022-0213-pointnet-contact-plate-clean-more-data/2022-0213-pointnet-contact-plate-clean-more-data_2022_02_13_22_34_03_0001/checkpoints/best_model.pth'
            # small voxel size
            # return 'data/seuss/2022-0304-pointnet-contact-plate-clean-small-voxel/2022-0304-pointnet-contact-plate-clean-small-voxel_2022_03_06_15_29_20_0001/checkpoints/best_model.pth'
            # return 'data/seuss/2022-0313-pointnet-contact-plate-clean-smaller-epsilon/2022-0313-pointnet-contact-plate-clean-smaller-epsilon_2022_03_13_16_42_12_0001/checkpoints/best_model.pth'
        else:
            return 'data/seuss/2022-0213-pointnet-contact-plate-clean-more-data/2022-0213-pointnet-contact-plate-clean-more-data_2022_02_13_22_34_03_0002/checkpoints/best_model.pth'
    if mode == 'primitive':
        if not use_gt_particle:
            return 'data/seuss/2022-0123-pointnet-primitive-contact/2022-0123-pointnet-primitive-contact_2022_01_23_20_53_57_0001/checkpoints/best_model.pth'
        else:
            return 'data/seuss/2022-0123-pointnet-primitive-contact/2022-0123-pointnet-primitive-contact_2022_02_23_19_18_27_0001/checkpoints/best_model.pth'


def get_contact_model_all(use_gt_particle):
    if not use_gt_particle:
        return 'data/seuss/2022-0213-pointnet-contact-all-data/2022-0213-pointnet-contact-all-data_2022_02_13_18_34_07_0001/checkpoints/best_model.pth'
    else:
        return 'data/seuss/2022-0213-pointnet-contact-all-data/2022-0213-pointnet-contact-all-data_2022_02_17_19_59_08_0001/checkpoints/best_model.pth'


def get_train_f1(contact_model_dir):
    # old task-specific model
    # if mode == 'bathing':
    #     return 0.75
    # elif mode == 'dressing':
    #     return 0.95
    # elif mode == 'plate-clean':
    #     return 0.75

    if contact_model_dir == 'data/seuss/2022-0323-pointnet-contact-bathing-smaller-epsilon/2022-0323-pointnet-contact-bathing-smaller-epsilon_2022_04_03_23_58_02_0001/checkpoints/best_model.pth':
        return 0.9
    if contact_model_dir == 'data/seuss/2022-0323-pointnet-contact-dressing-smaller-epsilon/2022-0323-pointnet-contact-dressing-smaller-epsilon_2022_03_23_12_41_42_0001/checkpoints/best_model.pth':
        return 0.95
    if contact_model_dir == 'data/seuss/2022-0313-pointnet-contact-plate-clean-smaller-epsilon/2022-0313-pointnet-contact-plate-clean-smaller-epsilon_2022_03_13_16_42_12_0001/checkpoints/best_model.pth':
        return 0.9

    if contact_model_dir == 'data/seuss/2022-0304-pointnet-contact-plate-clean-small-voxel/2022-0304-pointnet-contact-plate-clean-small-voxel_2022_03_06_15_29_20_0001/checkpoints/best_model.pth':
        return 0.
        # return 0.9 # best value from simulation

    if contact_model_dir == 'data/seuss/2022-0123-pointnet-primitive-contact/2022-0123-pointnet-primitive-contact_2022_02_23_19_18_27_0001/checkpoints/best_model.pth':
        return 0.7

    if contact_model_dir == 'data/seuss/2022-0213-pointnet-contact-all-data/2022-0213-pointnet-contact-all-data_2022_02_13_18_34_07_0001/checkpoints/best_model.pth':
        return 0.85
    elif contact_model_dir == 'data/seuss/2022-0213-pointnet-contact-bathing-more-data/2022-0213-pointnet-contact-bathing-more-data_2022_02_13_22_29_26_0001/checkpoints/best_model.pth':
        return 0.7
    elif contact_model_dir == 'data/seuss/2022-0213-pointnet-contact-dressing-more-data/2022-0213-pointnet-contact-dressing-more-data_2022_02_13_22_27_02_0001/checkpoints/best_model.pth':
        return 0.95
    if contact_model_dir == 'data/seuss/2022-0120-pointnet-dressing-contact/2022-0120-pointnet-dressing-contact_2022_01_21_12_57_56_0001/checkpoints/best_model.pth':
        return 0.95
    elif contact_model_dir == 'data/seuss/2022-0213-pointnet-contact-plate-clean-more-data/2022-0213-pointnet-contact-plate-clean-more-data_2022_02_13_22_34_03_0001/checkpoints/best_model.pth':
        return 0.75

    elif contact_model_dir == 'data/seuss/2022-0213-pointnet-contact-all-data/2022-0213-pointnet-contact-all-data_2022_02_17_19_59_08_0001/checkpoints/best_model.pth':
        return 0.9
    elif contact_model_dir == 'data/seuss/2022-0213-pointnet-contact-bathing-more-data/2022-0213-pointnet-contact-bathing-more-data_2022_02_13_22_29_26_0002/checkpoints/best_model.pth':
        return 0.75
    elif contact_model_dir == 'data/seuss/2022-0213-pointnet-contact-dressing-more-data/2022-0213-pointnet-contact-dressing-more-data_2022_02_13_22_27_02_0002/checkpoints/best_model.pth':
        return 0.95
    elif contact_model_dir == 'data/seuss/2022-0213-pointnet-contact-plate-clean-more-data/2022-0213-pointnet-contact-plate-clean-more-data_2022_02_13_22_34_03_0002/checkpoints/best_model.pth':
        return 0.75

    if contact_model_dir == 'data/seuss/2022-0123-pointnet-primitive-contact/2022-0123-pointnet-primitive-contact_2022_01_23_20_53_57_0001/checkpoints/best_model.pth':
        return 0.7

def get_max_force(mode, real_world):
    if mode == 'dressing':
        if real_world:
            # return 0.014373867925665529
            return 0.01
            # return None
        else:
            # return 0.8
            return 1
    elif mode == 'bathing':
        if real_world:
            return 0.01
        else:
            # return 0.8
            return 1
    elif mode == 'plate-clean':
        if real_world:
            return 0.005
        else:
            return 

def get_object_img_path(data_dir):
    # if data_dir in ['2022-0216-dressing-bad-traj-1-segment-tuned', '2022-0216-dressing-traj-6-segment-tuned']:
    print(data_dir)
    if 'sugar' in data_dir:
        return 'data/real-world/2022-0526-clean-plate-sugar-C-object-only/data_000020'
    

    if '2022-0216-dressing' in data_dir:
        return 'data/real-world/2022-0216-dressing-arm-only/data_000012'
    elif data_dir == '2022-0218-dressing-traj-4-segment-tuned':
        return 'data/real-world/2022-0218-dressing-arm-only/data_000012'
    elif data_dir == '2022-0218-dressing-traj-7-segment-tuned':
        return 'data/real-world/2022-0218-dressing-traj-7/data_000000'

    # if data_dir == '2022-0201-plate-small-cloth':
    if 'plate' in data_dir:
        return 'data/real-world/2022-0201-plate-small-cloth-plate-only/data_000012'

    if 'bathing' in data_dir:
        return 'data/real-world/2022-0205-bathing-manikin-only/data_000012'

   

def get_camera_matrix_path(data_dir):
    # if data_dir in ['2022-0216-dressing-bad-traj-1-segment-tuned', '2022-0216-dressing-traj-6-segment-tuned']:
    if 'sugar' in data_dir:
        return 'data/real-world/2022-0526-clean-plate-sugar-C-object-only/R_camera_to_world.npy'

    if '2022-0216-dressing' in data_dir:
        return 'data/real-world/2022-0216-dressing-arm-only/R_camera_to_world.npy'
    elif data_dir in ['2022-0218-dressing-traj-4-segment-tuned', '2022-0218-dressing-traj-7-segment-tuned']:
        return 'data/real-world/2022-0218-dressing-arm-only/R_camera_to_world.npy'

    if 'plate' in data_dir:
        return 'data/real-world/2022-0201-plate-small-cloth/R_camera_to_world.npy'

    if 'bathing' in data_dir:
        return 'data/real-world/2022-0205-bathing-manikin-only/R_camera_to_world.npy'



def get_p(mode, real_world):
    if not real_world:
        # return -0.7 # IROS value for all tasks
        if mode == 'plate-clean':
            # return -0.5
            return -0.7
        else:
            return -0.7
    else:
        if mode == 'dressing':
            return -0.5
        if mode == 'plate-clean':
            # return -0.5
            return -0.9
        if mode == 'bathing':
            return -0.5

def get_plot_epsilon(mode, real_world):
    if not real_world:
        return 0.00625 * 2
            
    else:
        if mode == 'dressing':
            return 0.00625 * 2.5
        elif mode == 'bathing':
            return 0.00625 * 2
        elif mode == 'plate-clean':
            return 0.00625 * 2 # for IROS paper submission
            # return 0.00625 * 1 
            # return 0.00625 * 1.5 # for after IROS small voxel size

def get_force_epsilon(mode, real_world):
    if not real_world:
        return 0.00625 * 3
    else:
        if mode in ['bathing', 'dressing']:
            return 0.00625 * 3
        elif mode == 'plate-clean':
            # return 0.00625 * 3 # IROS submission
            # return 0.00625 * 2
            return 0.00625 * 2
            # return 0.00625 * 0.75 # after IROS small voxelization

def get_baseline_contact_threshold(mode):
    if mode == 'plate-clean':
        # return 0.03125 # IROS submission
        # return 0.0046875
        return 0.00625 * 2.5 # IROS submission
    if mode == 'dressing':
        return 0.046875
    if mode == 'bathing':
        return 0.0375
    

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    vg = VariantGenerator()

    vg.add('algo', ['pointnet'])
    vg.add('epoch', [500])
    vg.add('split', ['valid'])
    vg.add('use_batch_norm', [False])
    vg.add('add_gripper_vel', [True])
    vg.add('add_cloth_vel', [False])
    vg.add('force_normalize_mean', [None])
    vg.add('force_normalize_std', [None])
    vg.add('remove_normalized_coordinate', [True])
    vg.add('force_feature_num', lambda add_gripper_vel, add_cloth_vel, num_temporal_step: [get_feature_num(add_gripper_vel, add_cloth_vel, num_temporal_step)])
    vg.add('contact_feature_num', [1])
    vg.add('num_temporal_step', [0])
    vg.add('neighbor_radius', [0.00625 * 6])
    # vg.add('neighbor_radius', [0.00625 * 1.8])
    vg.add('mask_radius', [0.00625 * 10])
    # vg.add('mask_radius', [0.00625 * 3])

    vg.add('contact_num_layer', [4])
    vg.add('contact_sa_radius', lambda contact_num_layer: [get_sa_radius(contact_num_layer)])
    vg.add('contact_sa_ratio', lambda contact_num_layer: [get_sa_ratio(contact_num_layer)])
    vg.add('contact_sa_mlp_list', lambda contact_num_layer: [get_sa_mlp_list(contact_num_layer)])
    vg.add('contact_fp_mlp_list', lambda contact_num_layer: [get_fp_mlp_list(contact_num_layer)])
    vg.add('contact_linear_mlp_list', lambda contact_num_layer: [get_linear_mlp_list(contact_num_layer)])
    vg.add('contact_fp_k', lambda contact_num_layer: [get_fp_k(contact_num_layer)])

    vg.add('force_num_layer', [4])
    vg.add('force_sa_radius', lambda force_num_layer: [get_sa_radius(force_num_layer)])
    vg.add('force_sa_ratio', lambda force_num_layer: [get_sa_ratio(force_num_layer)])
    vg.add('force_sa_mlp_list', lambda force_num_layer: [get_sa_mlp_list(force_num_layer)])
    vg.add('force_fp_mlp_list', lambda force_num_layer: [get_fp_mlp_list(force_num_layer)])
    vg.add('force_linear_mlp_list', lambda force_num_layer: [get_linear_mlp_list(force_num_layer)])
    vg.add('force_fp_k', lambda force_num_layer: [get_fp_k(force_num_layer)])
    vg.add('contact_residual', [False])
    vg.add('force_residual', [False])
    vg.add('cuda_idx', [0])
    vg.add('use_gt_contact', [False])
    vg.add('lr', [0.001])    
    vg.add('seed', [100])
    vg.add('plot_img_interval', [-1])

    vg.add('real_world', [1])
    # vg.add('mode', ['bathing'])
    # vg.add('mode', ['dressing'])
    vg.add('mode', ['plate-clean'])
    # vg.add('mode', ['primitive'])
    # vg.add('mode', ['bathing', 'plate-clean'])
    vg.add('voxel_size', [0.00625*2.5])
    vg.add('noise_level', [0])
    vg.add('save_force', [1])
    vg.add('data_dir', lambda mode, real_world: [
        # '2022-0523-dressing-epsilon-5-noise-test'
        # '2022-0526-clean-plate-sugar-C-3'
        '2022-0526-clean-plate-sugar-C-3-2'
        # get_dir(mode, real_world)
        # '2022-0205-bathing-good',
        # '2022-0205-bathing-3',
        # '2022-0205-bathing-4',
        # '2022-0205-bathing-6'
        # '2022-0216-dressing-bad-traj-2-voxel-in-camera',
        # '2022-0218-dressing-traj-4-segment-tuned',
        # '2022-0216-dressing-bad-traj-1-segment-tuned',
        # '2022-0216-dressing-traj-6-segment-tuned',
        # '2022-0218-dressing-traj-7-segment-tuned'
    ])
    vg.add('plot_video_interval', [1])
    vg.add('plot_traj_idx', [
        [0]
        # [0, 1, 2, 3, 4, 5]
    ])
    vg.add('p', lambda mode, real_world: [get_p(mode, real_world)]) # real-world dressing
    vg.add('plot_epsilon', lambda mode, real_world: [get_plot_epsilon(mode, real_world)])
    vg.add('force_epsilon', lambda mode, real_world: [get_force_epsilon(mode, real_world)])
    vg.add('baseline_contact_threshold', lambda mode: [get_baseline_contact_threshold(mode)])
    vg.add('object_img_path', lambda data_dir: [get_object_img_path(data_dir)])
    vg.add('camera_matrix_path', lambda data_dir: [get_camera_matrix_path(data_dir)])
    vg.add('max_force', lambda mode, real_world: [get_max_force(mode, real_world)])
    vg.add('plot_mode', lambda real_world: [get_plot_mode(real_world)])
    vg.add('plot_force_map', [True])

    vg.add('use_gt_particle', [False])
    ### task-specific model
    vg.add('load_contact_name', lambda mode, use_gt_particle: [get_contact_model(mode, use_gt_particle)])
    ### all dataset model
    # vg.add('load_contact_name', lambda use_gt_particle: [get_contact_model_all(use_gt_particle)])
    vg.add('train_f1', lambda load_contact_name: [get_train_f1(load_contact_name)])

    vg.add('use_force_for_contact', [False])
    ### task-specific model
    vg.add('load_force_name', lambda mode, use_gt_particle: [get_force_model_more_data(mode, use_gt_particle)])
    # vg.add('load_force_name', lambda mode, use_gt_particle: [get_force_model(mode, use_gt_particle)])
    ### all dataset model
    # vg.add('load_force_name', lambda use_gt_particle: [get_force_model_all_more_data(use_gt_particle)])
  
    vg.add('train_traj_num', [1000])
    vg.add('valid_traj_num', [200])
    vg.add('batch_size', [1]) # 8

    vg.add('force_loss_mode', ['contact'])
    vg.add('train_pos_label_weight', [11.2])
    vg.add('contact_loss_balance', [0])
    vg.add('save_interval', [20])
    vg.add('only_eval', [20])

    # exp_prefix = '2022-0407-post-IROS-plate-clean-real-world-small-epsilon'
    exp_prefix = '2022-0407-post-IROS-plate-clean-simulation-small-epsilon'
    exp_prefix = 'test-speed-and-point-cloud-size'
    # exp_prefix = '2022-0523-test-robustness-to-noise-pn'
    # exp_prefix = '2022-0526-clean-plate-sugar'
    exp_prefix = '2022-0528-clean-plate-sugar'
    
    print('Number of configurations: ', len(vg.variants()))
    print("exp_prefix: ", exp_prefix)

    hostname = socket.gethostname()

    sub_process_popens = []

    variations = set(vg.variations())

    task_per_gpu = 1
    all_vvs = vg.variants()
    slurm_nums = len(all_vvs) // task_per_gpu
    if len(all_vvs) % task_per_gpu != 0:
        slurm_nums += 1
    
    for idx in range(slurm_nums):
        beg = idx * task_per_gpu
        end = min((idx+1) * task_per_gpu, len(all_vvs))
        vvs = all_vvs[beg:end]
    # for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 10:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        if mode in ['seuss', 'autobot']:
            if idx == 0:
                # compile_script = 'compile.sh'  # For the first experiment, compile the current softgym
                compile_script = None  # For the first experiment, compile the current softgym
                wait_compile = None
            else:
                compile_script = None
                wait_compile = None  # Wait 30 seconds for the compilation to finish
        elif mode == 'ec2':
            compile_script = 'compile_1.0.sh'
            wait_compile = None
        else:
            compile_script = wait_compile = None
        if hostname.startswith('autobot') and gpu_num > 0:
            env_var = {'CUDA_VISIBLE_DEVICES': str(idx % gpu_num)}
        else:
            env_var = None
        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variants=vvs,
            # variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            variations=variations,
            env=env_var,
            task_per_gpu=task_per_gpu,
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
