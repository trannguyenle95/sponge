import time
import click
import socket
import multiprocessing as mp
from chester.run_exp import run_experiment_lite, VariantGenerator
from haptic.shared.train_haptic_contact import run_task

def get_feature_num(add_gripper_vel, add_cloth_vel, remove_normalized_coordinate):
    base = 4
    if add_gripper_vel:
        base += 3
    if add_cloth_vel:
        base += 3
    if remove_normalized_coordinate:
        base -= 3
    return base

def get_sa_radius(layer):
    if layer == 3:
        return [0.05, 0.1]
    elif layer == 4:
        return [0.05, 0.1, 0.2]
    elif layer == 5:
        return [0.05, 0.1, 0.2, 0.4]

def get_sa_ratio(layer):
    if layer == 3:
        return  [0.4, 0.5]
    elif layer == 4:
        return [0.4, 0.5, 0.6]
    elif layer == 5:
        return [0.4, 0.5, 0.6, 0.7]

def get_sa_mlp_list(layer):
    if layer == 3:
        return  [[64, 64, 128], [128, 128, 256], [256, 512, 1024]]
    elif layer == 4:
        return [[64, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 1024]]
    elif layer == 5:
        return [[64, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 1024], [1024, 1024, 2048]]

def get_fp_mlp_list(layer):
    if layer == 3:
        return  [[256, 256], [256, 128], [128, 128, 128]]
    elif layer == 4:
        return  [[256, 256], [256, 256], [256, 256, 128], [128, 128, 128]]
    elif layer == 5:
        return [[256, 256], [256, 256], [256, 256, 128], [128, 128, 128], [128, 128, 128]]

def get_linear_mlp_list(layer):
    if layer == 3:
        return  [128, 128]
    elif layer == 4:
        return  [128, 128, 128]
    elif layer == 5:
        return  [128, 128, 128]

def get_fp_k(layer):
    if layer == 3:
        return  [1, 3, 3]
    elif layer == 4:
        return  [3, 3, 3, 3]
    elif layer == 5:
        return  [3, 3, 3, 3, 3]

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '2022-0514-pointnet-contact-epsilon-2.5'
    vg = VariantGenerator()

    vg.add('algo', ['pointnet'])
    vg.add('epoch', [500])
    if not debug:
        vg.add('train_traj_num', [2000])
        vg.add('valid_traj_num', [200])
    else:
        vg.add('train_traj_num', [5])
        vg.add('valid_traj_num', [1])
    vg.add('batch_size', [12])
    vg.add('use_batch_norm', [False])
    vg.add('add_gripper_vel', [False])
    vg.add('remove_normalized_coordinate', [True])
    vg.add('add_cloth_vel', [False])
    vg.add('feature_num', lambda add_gripper_vel, add_cloth_vel, remove_normalized_coordinate: [get_feature_num(add_gripper_vel, add_cloth_vel, remove_normalized_coordinate)])
    vg.add('use_parallel_gpu', [False])
    vg.add('neighbor_radius', [0.00625 * 6])
    # vg.add('neighbor_radius', [0.00625 * 1.8])
    vg.add('auxiliary_loss_mode', ['None'])
    vg.add('mask_radius', [0.00625 * 10])
    # vg.add('mask_radius', [0.00625 * 3.0])

    vg.add('contact_num_layer', [4])
    vg.add('contact_sa_radius', lambda contact_num_layer: [get_sa_radius(contact_num_layer)])
    vg.add('contact_sa_ratio', lambda contact_num_layer: [get_sa_ratio(contact_num_layer)])
    vg.add('contact_sa_mlp_list', lambda contact_num_layer: [get_sa_mlp_list(contact_num_layer)])
    vg.add('contact_fp_mlp_list', lambda contact_num_layer: [get_fp_mlp_list(contact_num_layer)])
    vg.add('contact_linear_mlp_list', lambda contact_num_layer: [get_linear_mlp_list(contact_num_layer)])
    vg.add('contact_fp_k', lambda contact_num_layer: [get_fp_k(contact_num_layer)])

    vg.add('contact_residual', [False])

    vg.add('cuda_idx', [0])
    vg.add('load_model', [False])
    vg.add('load_name', [None])

    vg.add('learning_rate', [0.0001])    
    
    vg.add('use_gt_particle', [False])
    vg.add('seed', [100])
    # vg.add('data_dir', [
    #     [
    #         # '2022-0119-dressing-no-voxelization-fix-hospital-gown',
    #         # '2022-0209-more-data-dressing',
    #         # '2022-0119-bathing-no-voxelization-fix-loofah',
    #         # '2022-0209-more-data-bathing',
    #         # '2022-0119-plate-clean-no-voxelization-fix-loofah',
    #         # '2022-0209-more-data-plate-clean',
    #         '2022-0119-towel-multile-object'
    #     ]
    # ])
    vg.add('data_dir', [
        # ['2022-0322-bathing-smaller-epsilon']
        # ['2022-0322-dressing-smaller-epsilon']
        ['2022-0430-dressing-epsilon-2.5'],
        ['2022-0430-bathing-epsilon-2.5'],
        ['2022-0430-plate-clean-epsilon-2.5'],
    ])

    vg.add('contact_loss_balance', [1])
    vg.add('train_pos_label_weight', [11.2]) 
    vg.add('plot_img_interval', [1000])
    vg.add('plot_video_interval', [1000])
    vg.add('save_interval', [20])

    print('Number of configurations: ', len(vg.variants()))
    print("exp_prefix: ", exp_prefix)

    hostname = socket.gethostname()

    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
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
            variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            env=env_var
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
