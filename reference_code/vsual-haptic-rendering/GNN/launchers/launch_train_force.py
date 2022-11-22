import time
import click
import socket
import multiprocessing as mp
from chester.run_exp import run_experiment_lite, VariantGenerator
from haptic.shared.train_haptic_force import run_task


def get_node_dim(add_gripper_vel, add_cloth_vel, remove_normalized_coordinate, num_temporal_step):
    base = (num_temporal_step) * 3 + 3
    if add_gripper_vel:
        base += 3
    if add_cloth_vel:
        base += 3
    return base

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '2022-0514-gnn-force-epsilon-2.5'
    vg = VariantGenerator()

    vg.add('algo', ['GNN'])
    vg.add('epoch', [500])
    if not debug:
        vg.add('train_traj_num', [2000])
        vg.add('valid_traj_num', [200])
    else:
        vg.add('train_traj_num', [10])
        vg.add('valid_traj_num', [2])

    vg.add('add_gripper_vel', [True])
    vg.add('use_gt_particle', [False])
    vg.add('loss_weight_inverse_mean', [False])
    vg.add('num_temporal_step', [0])
    vg.add('add_cloth_vel', [False])
    vg.add('remove_normalized_coordinate', [True])
    vg.add('force_normalize_mean', [None])
    vg.add('force_normalize_std', [None])
    # vg.add('auxiliary_loss_mode', ['smooth_feature', 'smooth'])
    vg.add('auxiliary_loss_mode', ['None'])
    vg.add('auxiliary_loss_w', [0])
    vg.add('pos_interval', [5])

    vg.add('cuda_idx', [0])
    vg.add('use_parallel_gpu', [0])
    vg.add('filter_edge', [2])

    vg.add('neighbor_radius', [0.00625 * 6])
    # vg.add('neighbor_radius', [0.00625 * 1.8])
    vg.add('mask_radius', [0.00625 * 10])
    # vg.add('mask_radius', [0.00625 * 3])

    vg.add('node_dim', lambda add_gripper_vel, add_cloth_vel, remove_normalized_coordinate, num_temporal_step: [get_node_dim(add_gripper_vel, add_cloth_vel, remove_normalized_coordinate, num_temporal_step)])
    vg.add('edge_dim', [4])
    
    vg.add('learning_rate', [0.0001])    
    vg.add('force_proc_layer', [4])

    vg.add('global_size', [128])
    vg.add('batch_size', [4])
    
    vg.add('seed', [100])
    # vg.add('data_dir', ['2022-0119-plate-clean-no-voxelization-fix-loofah'])
    # vg.add('data_dir', ['2022-0119-bathing-no-voxelization-fix-loofah'])
    # vg.add('data_dir', ['2022-0119-dressing-no-voxelization-fix-hospital-gown'])
    # vg.add('data_dir', ['2022-0119-towel-multile-object'])
    # vg.add('data_dir', [
    #     [
    #         # '2022-0119-dressing-no-voxelization-fix-hospital-gown',
    #         # '2022-0209-more-data-dressing',
    #         # '2022-0119-bathing-no-voxelization-fix-loofah',
    #         # '2022-0209-more-data-bathing',
    #         '2022-0119-plate-clean-no-voxelization-fix-loofah',
    #         '2022-0209-more-data-plate-clean',
    #     ]
    # ])
    vg.add('data_dir', [
        # ['2022-0322-bathing-smaller-epsilon']
        # ['2022-0322-dressing-smaller-epsilon']
        # ['2022-0430-dressing-epsilon-2.5'],
        # ['2022-0430-bathing-epsilon-2.5'],
        ['2022-0430-plate-clean-epsilon-2.5']
    ])

    vg.add('average_statistics', [True])
    vg.add('force_loss_mode', ['contact']) 
    vg.add('train_pos_label_weight', [11.2])
    vg.add('plot_img_interval', [1000])
    vg.add('plot_video_interval', [1000])
    vg.add('save_interval', [10])

    vg.add('load_model', [False])
    vg.add('load_name', ['data/local/1121-gnn-1000-local-train-force-only/1121-gnn-1000-local-train-force-only_2021_11_21_20_12_24_0001/checkpoints/best_model.pth'])


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
                wait_compile = 0
            else:
                compile_script = None
                wait_compile = 600 # Wait 30 seconds for the compilation to finish
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
