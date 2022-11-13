import time
import click
import socket
import multiprocessing as mp
from chester.run_exp import run_experiment_lite, VariantGenerator
from haptic.shared.train_haptic_contact import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '2022-0323-gnn-contact-dressing-smaller-epsilon'
    exp_prefix = '2022-0514-gnn-contact-epsilon-2.5'
    vg = VariantGenerator()

    vg.add('algo', ['GNN'])
    vg.add('epoch', [500])
    if not debug:
        vg.add('train_traj_num', [2000])
        vg.add('valid_traj_num', [200])
    else:
        vg.add('train_traj_num', [5])
        vg.add('valid_traj_num', [1])

    vg.add('add_gripper_vel', [False])
    vg.add('add_cloth_vel', [False])
    vg.add('remove_normalized_coordinate', [True])
    vg.add('auxiliary_loss_mode', ['None'])
    vg.add('mask_radius', [0.00625 * 10])
    # vg.add('mask_radius', [0.00625 * 3])
    vg.add('num_temporal_step', [0])

    vg.add('cuda_idx', [0])
    vg.add('use_parallel_gpu', [0])
    vg.add('filter_edge', [2])
    vg.add('use_gt_particle', [False])

    vg.add('neighbor_radius', [0.00625 * 6])
    # vg.add('neighbor_radius', [0.00625 * 1.8])

    vg.add('node_dim', [1])
    vg.add('edge_dim', [4])
    
    vg.add('learning_rate', [0.0001])    
    vg.add('contact_proc_layer', [4])

    vg.add('global_size', [128])
    vg.add('batch_size', [4])
    
    vg.add('seed', [100])
    vg.add('average_statistics', [1])
    # vg.add('data_dir', [
    #     [
    #         # '2022-0119-dressing-no-voxelization-fix-hospital-gown',
    #         # '2022-0209-more-data-dressing',
    #         # '2022-0119-bathing-no-voxelization-fix-loofah',
    #         # '2022-0209-more-data-bathing',
    #         # '2022-0119-plate-clean-no-voxelization-fix-loofah',
    #         # '2022-0209-more-data-plate-clean',
    #         '2022-0119-towel-multile-object',
    #     ]
    # ])
    vg.add('data_dir', [
        # ['2022-0312-plate-clean-smaller-epsilon']
        # ['2022-0322-bathing-smaller-epsilon']
        # ['2022-0322-dressing-smaller-epsilon']
        ['2022-0430-dressing-epsilon-2.5'],
        # ['2022-0430-bathing-epsilon-2.5'],
        # ['2022-0430-plate-clean-epsilon-2.5'],
    ])

    vg.add('contact_loss_balance', [0])
    vg.add('train_pos_label_weight', [11.2])
    vg.add('plot_img_interval', [1000])
    vg.add('plot_video_interval', [1000])
    vg.add('save_interval', [10])

    vg.add('load_name', [None])

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
