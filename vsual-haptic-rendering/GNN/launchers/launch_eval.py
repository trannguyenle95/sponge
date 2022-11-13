import time
import click
import socket
import multiprocessing as mp

from pyparsing import Word
from chester.run_exp import run_experiment_lite, VariantGenerator
from haptic.shared.eval_script import run_task
from haptic.pointNet_geo.launchers.launch_eval import get_p, get_plot_epsilon, get_force_epsilon, get_object_img_path, get_camera_matrix_path, get_baseline_contact_threshold

def get_node_dim(add_gripper_vel, add_cloth_vel, num_temporal_step):
    base = (num_temporal_step) * 3 + 3
    if add_gripper_vel:
        base += 3
    if add_cloth_vel:
        base += 3
    return base

def get_dir(mode, real_world):
    if mode == 'dressing':
        if not real_world:
            # IROS
            return '2022-0211-testset-dressing'
            # return '2022-0407-dressing-smaller-epsilon-test'
        else:
            return '2022-0216-dressing-bad-traj-1-fix-gripper'

            # return '2022-0216-dressing-traj-6-voxel-in-camera'
            # return '2022-0216-dressing-bad-traj-1-voxel-in-camera'
            # return '2022-0216-dressing-bad-traj-2-voxel-in-camera'
            return '2022-0218-dressing-traj-4-voxel-in-camera'
            
    elif mode == 'bathing':
        if not real_world:
            # return '2022-0119-bathing-no-voxelization-fix-loofah'
            return '2022-0211-testset-bathing' # IROS
            # return '2022-0407-bathing-smaller-epsilon-test'

        else:
            return '2022-0205-bathing-6-voxel-in-camera'
            return '2022-0205-bathing-good-voxel-in-camera'
            return '2022-0205-bathing-good'
            return '2022-0205-bathing-good'
    elif mode == 'plate-clean':
        if not real_world:
            # return '2022-0119-plate-clean-no-voxelization-fix-loofah'
            return '2022-0211-testset-plate-clean'  # IROS one
            # return '2022-0322-plate-clean-test-smaller-epsilon'
        else:
            # return '2022-0307-plate-small-cloth-smaller-voxel' ### after IROS, small voxel size
            # return '2022-0126-plate-real-world'
            return '2022-0201-plate-small-cloth' ### final one to be used in paper
            # return '2022-0201-plate-small-cloth-higher'
    elif mode == 'primitive':
        # return '2022-0119-towel-multile-object'
        return 'IROS-primitive-testset'

def get_force_model(mode, use_gt_particle):
    if not use_gt_particle:
        if mode == 'bathing':
            return 'data/seuss/2022-0119-gnn-bathing-correct/2022-0119-gnn-bathing-correct_2022_01_19_01_46_45_0001/checkpoints/best_model.pth'
        elif mode == 'dressing':
            return 'data/seuss/2022-0120-gnn-dressing/2022-0120-gnn-dressing_2022_01_20_00_18_12_0001/checkpoints/best_model.pth'
        elif mode == 'plate-clean':
            return 'data/seuss/2022-0119-gnn-plate-clean-correct/2022-0119-gnn-plate-clean-correct_2022_01_19_01_38_44_0001/checkpoints/best_model.pth'
    else:
        if mode == 'bathing':
            return 'data/seuss/2022-0119-gnn-bathing-correct/2022-0119-gnn-bathing-correct_2022_01_19_02_07_17_0001/checkpoints/best_model.pth'
        elif mode == 'dressing':
            return 'data/seuss/2022-0120-gnn-dressing/2022-0120-gnn-dressing_2022_01_20_00_03_08_0001/checkpoints/best_model.pth'
        elif mode == 'plate-clean':
            return 'data/seuss/2022-0119-gnn-plate-clean-correct/2022-0119-gnn-plate-clean-correct_2022_01_19_01_50_32_0001/checkpoints/best_model.pth'

def get_force_model_more_data(mode, use_gt_particle):
    if not use_gt_particle:
        if mode == 'bathing':
            # IROS
            # return 'data/seuss/2022-0209-gnn-bathing-more-data/2022-0209-gnn-bathing-more-data_2022_02_09_23_23_53_0001/checkpoints/best_model.pth'
            return 'data/seuss/2022-0323-gnn-force-bathing-smaller-epsilon/2022-0323-gnn-force-bathing-smaller-epsilon_2022_03_23_12_37_24_0001/checkpoints/best_model.pth'

        elif mode == 'dressing':
            # IROS
            return 'data/seuss/2022-0209-gnn-dressing-more-data/2022-0209-gnn-dressing-more-data_2022_02_09_11_18_48_0001/checkpoints/best_model.pth'
            # return 'data/seuss/2022-0323-gnn-force-dressing-smaller-epsilon/2022-0323-gnn-force-dressing-smaller-epsilon_2022_03_23_15_17_47_0001/checkpoints/best_model.pth'
        elif mode == 'plate-clean':
            # after IROS
            # return 'data/seuss/2022-0309-gnn-force-plate-clean-small-epsilon/2022-0309-gnn-force-plate-clean-small-epsilon_2022_03_09_20_51_33_0001/checkpoints/best_model.pth'
            # return 'data/seuss/2022-0304-gnn-force-plate-clean-small-voxel/2022-0304-gnn-force-plate-clean-small-voxel_2022_03_06_15_24_19_0001/checkpoints/best_model.pth'
            # return 'data/seuss/2022-0313-gnn-force-plate-clean-smaller-epsilon/2022-0313-gnn-force-plate-clean-smaller-epsilon_2022_03_13_16_43_04_0001/checkpoints/best_model.pth'

            return 'data/seuss/2022-0209-gnn-plate-clean-more-data/2022-0209-gnn-plate-clean-more-data_2022_02_09_18_11_30_0001/checkpoints/best_model.pth' # IROS one
        if mode == 'primitive':
            return 'data/seuss/2022-0123-gnn-primitive/2022-0123-gnn-primitive_2022_01_23_00_39_26_0001/checkpoints/best_model.pth'
    else:
        if mode == 'bathing':
            return 'data/seuss/2022-0209-gnn-bathing-more-data/2022-0209-gnn-bathing-more-data_2022_02_09_23_23_53_0002/checkpoints/best_model.pth'
        elif mode == 'dressing':
            return 'data/seuss/2022-0209-gnn-dressing-more-data/2022-0209-gnn-dressing-more-data_2022_02_09_11_18_48_0002/checkpoints/best_model.pth'
        elif mode == 'plate-clean':
            # IROS one
            return 'data/seuss/2022-0209-gnn-plate-clean-more-data/2022-0209-gnn-plate-clean-more-data_2022_02_09_18_11_30_0002/checkpoints/best_model.pth'
        if mode == 'primitive':
            return 'data/seuss/2022-0123-gnn-primitive/2022-0123-gnn-primitive_2022_01_23_00_39_26_0002/checkpoints/best_model.pth'

def get_force_model_all_data(use_gt_particle):
    if not use_gt_particle:
        return 'data/seuss/2022-0208-gnn-all-dataset/2022-0208-gnn-all-dataset_2022_02_08_16_20_31_0001/checkpoints/best_model.pth'
    else:
        return 'data/seuss/2022-0208-gnn-all-dataset/2022-0208-gnn-all-dataset_2022_02_08_16_20_31_0002/checkpoints/best_model.pth'


def get_force_model_all_more_data(use_gt_particle):
    if not use_gt_particle:
        return 'data/seuss/2022-0213-gnn-more-data-all/2022-0213-gnn-more-data-all_2022_02_13_01_19_39_0001/checkpoints/best_model.pth'
    else:
        return 'data/seuss/2022-0213-gnn-more-data-all/2022-0213-gnn-more-data-all_2022_02_13_01_19_39_0002/checkpoints/best_model.pth'

def get_contact_model(mode, use_gt_particle):
    if mode == 'bathing':
        # return 'data/seuss/2022-0120-gnn-bathing-contact/2022-0120-gnn-bathing-contact_2022_01_20_20_52_13_0001/checkpoints/best_model.pth'
        if not use_gt_particle:
            # IROS model
            return 'data/seuss/2022-0213-gnn-contact-bathing-more-data/2022-0213-gnn-contact-bathing-more-data_2022_02_13_22_24_00_0001/checkpoints/best_model.pth'
            # return 'data/seuss/2022-0323-gnn-contact-bathing-smaller-epsilon/2022-0323-gnn-contact-bathing-smaller-epsilon_2022_03_23_12_14_43_0001/checkpoints/best_model.pth'
        else:
            return 'data/seuss/2022-0213-gnn-contact-bathing-more-data/2022-0213-gnn-contact-bathing-more-data_2022_02_13_22_24_00_0002/checkpoints/best_model.pth'

    elif mode == 'dressing':
        if not use_gt_particle:
            # IROS
            # return 'data/seuss/2022-0213-gnn-contact-dressing-more-data/2022-0213-gnn-contact-dressing-more-data_2022_02_13_20_53_30_0001/checkpoints/best_model.pth'
            return 'data/seuss/2022-0323-gnn-contact-dressing-smaller-epsilon/2022-0323-gnn-contact-dressing-smaller-epsilon_2022_03_23_15_17_06_0001/checkpoints/best_model.pth'
        else:
            return 'data/seuss/2022-0213-gnn-contact-dressing-more-data/2022-0213-gnn-contact-dressing-more-data_2022_02_13_21_56_12_0001/checkpoints/best_model.pth'

    elif mode == 'plate-clean':
        if not use_gt_particle:
            # return  'data/seuss/2022-0120-gnn-plate-clean-contact/2022-0120-gnn-plate-clean-contact_2022_01_20_20_52_40_0001/checkpoints/best_model.pth' # IROS one
            # return 'data/seuss/2022-0213-gnn-contact-plate-clean-more-data/2022-0213-gnn-contact-plate-clean-more-data_2022_02_13_22_25_09_0001/checkpoints/best_model.pth'
            
            ### after IROS one
            # return 'data/seuss/2022-0309-gnn-contact-plate-clean-small-epsilon/2022-0309-gnn-contact-plate-clean-small-epsilon_2022_03_09_20_49_55_0001/checkpoints/best_model.pth'
            # return 'data/seuss/2022-0304-gnn-contact-plate-clean-small-voxel/2022-0304-gnn-contact-plate-clean-small-voxel_2022_03_06_15_27_25_0001/checkpoints/best_model.pth'
            return 'data/seuss/2022-0313-gnn-contact-plate-clean-smaller-epsilon/2022-0313-gnn-contact-plate-clean-smaller-epsilon_2022_03_13_16_40_35_0001/checkpoints/best_model.pth'
        else:
            return 'data/seuss/2022-0213-gnn-contact-plate-clean-more-data/2022-0213-gnn-contact-plate-clean-more-data_2022_02_13_22_25_09_0002/checkpoints/best_model.pth'

    if mode == 'primitive':
        if not use_gt_particle:
            return 'data/seuss/2022-0123-gnn-primitive-contact/2022-0123-gnn-primitive-contact_2022_01_23_00_40_44_0001/checkpoints/best_model.pth'
        else:
            return 'data/seuss/2022-0123-gnn-primitive-contact/2022-0123-gnn-primitive-contact_2022_02_23_19_17_08_0001/checkpoints/best_model.pth'


def get_contact_all_model(use_gt_particle):
    if not use_gt_particle:
        return 'data/seuss/2022-0213-gnn-contact-all-data/2022-0213-gnn-contact-all-data_2022_02_13_18_14_22_0001/checkpoints/best_model.pth'
    else:
        return 'data/seuss/2022-0213-gnn-contact-all-data/2022-0213-gnn-contact-all-data_2022_02_17_19_56_58_0001/checkpoints/best_model.pth'

def get_train_f1(contact_model_dir):
    if contact_model_dir == 'data/seuss/2022-0323-gnn-contact-bathing-smaller-epsilon/2022-0323-gnn-contact-bathing-smaller-epsilon_2022_03_23_12_14_43_0001/checkpoints/best_model.pth':
        return 0.2

    if contact_model_dir == 'data/seuss/2022-0323-gnn-contact-dressing-smaller-epsilon/2022-0323-gnn-contact-dressing-smaller-epsilon_2022_03_23_15_17_06_0001/checkpoints/best_model.pth':
        return 0.35

    if contact_model_dir == 'data/seuss/2022-0313-gnn-contact-plate-clean-smaller-epsilon/2022-0313-gnn-contact-plate-clean-smaller-epsilon_2022_03_13_16_40_35_0001/checkpoints/best_model.pth':
        return 0.35

    if contact_model_dir == 'data/seuss/2022-0304-gnn-contact-plate-clean-small-voxel/2022-0304-gnn-contact-plate-clean-small-voxel_2022_03_06_15_27_25_0001/checkpoints/best_model.pth':
        return 0.4

    if contact_model_dir == 'data/seuss/2022-0309-gnn-contact-plate-clean-small-epsilon/2022-0309-gnn-contact-plate-clean-small-epsilon_2022_03_09_20_49_55_0001/checkpoints/best_model.pth':
        return 0.45

    if contact_model_dir == 'data/seuss/2022-0123-gnn-primitive-contact/2022-0123-gnn-primitive-contact_2022_02_23_19_17_08_0001/checkpoints/best_model.pth':
        return 0.45

    if contact_model_dir == 'data/seuss/2022-0120-gnn-plate-clean-contact/2022-0120-gnn-plate-clean-contact_2022_01_20_20_52_40_0001/checkpoints/best_model.pth':
        return 0.4

    if contact_model_dir == 'data/seuss/2022-0213-gnn-contact-all-data/2022-0213-gnn-contact-all-data_2022_02_13_18_14_22_0001/checkpoints/best_model.pth':
        # return 0.45
        return 0.65
    if contact_model_dir == 'data/seuss/2022-0213-gnn-contact-all-data/2022-0213-gnn-contact-all-data_2022_02_17_19_56_58_0001/checkpoints/best_model.pth':
        return 0.5

    if contact_model_dir == 'data/seuss/2022-0213-gnn-contact-bathing-more-data/2022-0213-gnn-contact-bathing-more-data_2022_02_13_22_24_00_0001/checkpoints/best_model.pth':
        return 0.35
    if contact_model_dir == 'data/seuss/2022-0213-gnn-contact-dressing-more-data/2022-0213-gnn-contact-dressing-more-data_2022_02_13_20_53_30_0001/checkpoints/best_model.pth':
        return 0.45
    if contact_model_dir == 'data/seuss/2022-0213-gnn-contact-plate-clean-more-data/2022-0213-gnn-contact-plate-clean-more-data_2022_02_13_22_25_09_0001/checkpoints/best_model.pth':
        return 0.4

    if contact_model_dir == 'data/seuss/2022-0213-gnn-contact-bathing-more-data/2022-0213-gnn-contact-bathing-more-data_2022_02_13_22_24_00_0002/checkpoints/best_model.pth':
        return 0.4
    if contact_model_dir == 'data/seuss/2022-0213-gnn-contact-dressing-more-data/2022-0213-gnn-contact-dressing-more-data_2022_02_13_21_56_12_0001/checkpoints/best_model.pth':
        return 0.4
    if contact_model_dir == 'data/seuss/2022-0213-gnn-contact-plate-clean-more-data/2022-0213-gnn-contact-plate-clean-more-data_2022_02_13_22_25_09_0002/checkpoints/best_model.pth':
        return 0.45

    if contact_model_dir == 'data/seuss/2022-0123-gnn-primitive-contact/2022-0123-gnn-primitive-contact_2022_01_23_00_40_44_0001/checkpoints/best_model.pth':
        return 0.45
    
    
def get_plot_mode(real_world):
    if real_world:
        return 'matplotlib'
    else:
        return 'pyrender'

def get_max_force(mode, real_world):
    if mode == 'dressing':
        if real_world:
            return 0.025848820434022056
        else:
            return 1
    elif mode == 'bathing':
        return 1
    elif mode == 'plate-clean':
        if real_world:
            return None # IROS
            # return 0.01
        else:
            return None

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    vg = VariantGenerator()

    vg.add('algo', ['GNN'])
    vg.add('epoch', [500])
    vg.add('add_gripper_vel', [True])
    vg.add('add_cloth_vel', [False])
    vg.add('force_normalize_mean', [None])
    vg.add('force_normalize_std', [None])
    vg.add('remove_normalized_coordinate', [True])
    vg.add('num_temporal_step', [0])

    vg.add('cuda_idx', [-1])
    vg.add('filter_edge', [2])

    vg.add('voxel_size', [0.00625*2.5])
    vg.add('neighbor_radius', [0.00625 * 6])
    # vg.add('neighbor_radius', [0.00625 * 1.8])
    vg.add('mask_radius', [0.00625 * 10])
    # vg.add('mask_radius', [0.00625 * 3])
    vg.add('auxiliary_loss_mode', ['None'])

    vg.add('node_dim', [6])
    vg.add('force_node_dim', lambda add_gripper_vel, add_cloth_vel, num_temporal_step: [get_node_dim(add_gripper_vel, add_cloth_vel, num_temporal_step)])
    vg.add('contact_node_dim', [1])
    vg.add('edge_dim', [4])
    
    vg.add('learning_rate', [0.001])    
    vg.add('contact_proc_layer', [4])
    vg.add('force_proc_layer', [4])
    vg.add('global_size', [128])
    
    vg.add('seed', [100])
    vg.add('train_traj_num', [1000])
    vg.add('valid_traj_num', [200]) # 200
    vg.add('split', ['valid'])
    vg.add('batch_size', [1]) # 8
  
    vg.add('force_loss_mode', ['contact']) 
    vg.add('train_pos_label_weight', [11.2])
    vg.add('contact_loss_balance', [0])
    vg.add('use_gt_contact', [0])
    vg.add('save_interval', [10])
    vg.add('plot_img_interval', [-1])
  
    vg.add('real_world', [0])
    vg.add('mode', ['plate-clean'])
    # vg.add('mode', ['bathing'])
    # vg.add('mode', ['primitive'])
    # vg.add('mode', ['dressing'])
    # vg.add('noise_level', [0.006, 0.007, 0.008, 0.009, 0.01, 0.012, 0.015])
    vg.add('noise_level', [0])
    vg.add('save_force', [0])
    # vg.add('mode', ['dressing', 'bathing', 'plate-clean'])
    vg.add('data_dir', lambda mode, real_world: [
        # '2022-0523-dressing-epsilon-5-noise-test'
        # '2022-0526-clean-plate-sugar-C-2',
        # '2022-0526-clean-plate-sugar-C-3'
        # '2022-0526-clean-plate-sugar-C-3-2'
        get_dir(mode, real_world)
        # '2022-0205-bathing-good',
        # '2022-0205-bathing-3',
        # '2022-0205-bathing-4',
        # '2022-0205-bathing-6'
        # '2022-0216-dressing-bad-traj-2-voxel-in-camera',
        # '2022-0216-dressing-bad-traj-1-segment-tuned',
        # '2022-0216-dressing-traj-6-segment-tuned',
        # '2022-0218-dressing-traj-4-segment-tuned',
        # '2022-0218-dressing-traj-7-segment-tuned'
    ])
    vg.add('plot_video_interval', [-1])
    vg.add('plot_traj_idx', [
        [0]
        # [0, 1, 2, 3, 4, 5]
        # [7, 8, 9, 10, 11, 12],
        # [14]
    ])
    vg.add('plot_force_map', [True])
    vg.add('p', lambda mode, real_world: [get_p(mode, real_world)]) # real-world dressing
    vg.add('plot_epsilon', lambda mode, real_world: [get_plot_epsilon(mode, real_world)])
    vg.add('force_epsilon', lambda mode, real_world: [get_force_epsilon(mode, real_world)])
    vg.add('baseline_contact_threshold', lambda mode: [get_baseline_contact_threshold(mode)])
    vg.add('max_force', lambda mode, real_world: [get_max_force(mode, real_world)])
    vg.add('object_img_path', lambda data_dir: [get_object_img_path(data_dir)])
    vg.add('camera_matrix_path', lambda data_dir: [get_camera_matrix_path(data_dir)])
    
    vg.add('use_gt_particle', [False])
    ### generalized contact model
    # vg.add('load_contact_name', ['data/seuss/2022-0123-gnn-primitive-contact/2022-0123-gnn-primitive-contact_2022_01_23_00_40_44_0001/checkpoints/model_49.pth'])
    ### task specific contact model
    vg.add('load_contact_name', lambda mode, use_gt_particle: [get_contact_model(mode, use_gt_particle)])
    ### task agnostic contact model:
    # vg.add('load_contact_name', lambda use_gt_particle: [get_contact_all_model(use_gt_particle)])
    vg.add('train_f1', lambda load_contact_name: [get_train_f1(load_contact_name)])

    vg.add('use_force_for_contact', [False])
    ### generalization force model
    # vg.add('load_force_name', ['data/seuss/2022-0123-gnn-primitive/2022-0123-gnn-primitive_2022_01_23_00_39_26_0001/checkpoints/model_59.pth'])
    ### specialized force models
    # vg.add('load_force_name', lambda mode, use_gt_particle: [get_force_model(mode, use_gt_particle)])
    ### trained on all dataset force models
    # vg.add('load_force_name', lambda use_gt_particle: [get_force_model_all_data(use_gt_particle)])
    ### trained on individual dataset, more data
    vg.add('load_force_name', lambda mode, use_gt_particle: [get_force_model_more_data(mode, use_gt_particle)])
    ### trained on all dataset, more data
    # vg.add('load_force_name', lambda use_gt_particle: [get_force_model_all_more_data(use_gt_particle)])
    vg.add('plot_mode', lambda real_world: [get_plot_mode(real_world)])

    exp_prefix = '2022-0407-gnn-real-world-plate-clean-smaller-epsilon'
    exp_prefix = '2022-0407-gnn-simulation-plate-clean-smaller-epsilon'
    exp_prefix = '2022-0407-gnn-simulation-bathing-smaller-epsilon'
    exp_prefix = 'test-speed-and-point-cloud-size'
    # exp_prefix = '2022-0523-test-robustness-to-noise-in-pc'
    # exp_prefix = '2022-0528-clean-plate-sugar'

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
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            env=env_var,
            variations=variations,
            task_per_gpu=task_per_gpu,
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        # if debug:
        #     break


if __name__ == '__main__':
    main()
