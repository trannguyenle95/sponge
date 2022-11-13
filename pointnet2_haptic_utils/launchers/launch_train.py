import time
import click
import socket
import multiprocessing as mp
from chester.run_exp import run_experiment_lite, VariantGenerator
from haptic.Pointnet_Pointnet2_pytorch.train_semseg_haptic import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '1106-pointNet-no-track-running-stats'
    exp_prefix = '1106-pointnet-simple-2-traj-2'
    exp_prefix = '1112-pointnet-100-mem-debug'
    vg = VariantGenerator()

    voxel_size = 0.00625 * 1.5
    if not debug:
        vg.add('debug', [False])
        vg.add('train_traj_num', [100])
        vg.add('valid_traj_num', [10])
        
        vg.add('use_batch_norm', [False])
        vg.add('dropout', [False])
        vg.add('set_eval_for_batch_norm', [False])
        vg.add('track_running_stats', [True])
        vg.add('adjust_momentum', [False])

        vg.add('downsample', [False])
        vg.add('radius_list', [
            # [[0.05, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.8]],
            # [[voxel_size, voxel_size*2], [voxel_size*2, voxel_size*4], [voxel_size*4, voxel_size*8], [voxel_size*8, voxel_size*16]],
            [[voxel_size*2, voxel_size*4], [voxel_size*4, voxel_size*8], [voxel_size*8, voxel_size*16], [voxel_size*16, voxel_size*32]],
            # [[voxel_size*4, voxel_size*8], [voxel_size*8, voxel_size*16], [voxel_size*16, voxel_size*32], [voxel_size*32, voxel_size*64]],
        ])
        vg.add('npoint_list', [
            # [1024, 256, 64, 16],
            # [2048, 512, 128, 32],
            # [4096, 1024, 256, 64],
            [6000, 1500, 375, 95],
            # [4096, 2048, 1024, 512],
        ])
        vg.add('layer', [
            1
        ])
        vg.add('sample_point_1_list', [
            [16, 16, 16, 16],
            # [64, 32, 16, 16],
            # [128, 64, 32, 16],
        ])
        vg.add('sample_point_2_list', [
            [32, 32, 32, 32],
        ])

        vg.add('npoint', [2500])
        vg.add('batch_size', [8])

        vg.add('correct_z_rotation', [2])
        vg.add('epoch', [500])
        vg.add('train_pos_label_weight', [8, 11.2, 15])
        vg.add('seed', [100])
        vg.add('manual_lr_adjust', [False])
        vg.add('schedule_lr', [True])
        vg.add('data_dir', ['2021-11-10-variation-1000'])
        vg.add('learning_rate', [0.001])
        vg.add('force_loss_mode', ['balance'])
        vg.add('normal_loss_mode', ['balance'])
        vg.add('force_loss_weight', [1])
        vg.add('normal_loss_weight', [1])
        vg.add('plot_interval', [25])
        vg.add('separate_model', [True])
        # vg.add('load_dir', ['./data/seuss/1023-pn-force-non-shared/1023-pn-force-non-shared_2021_10_23_23_59_23_0002'])
        vg.add('load_dir', [None])
        vg.add('train', [True])
        vg.add('num_worker', [8])
    else:
        vg.add('debug', [False])
        vg.add('train_traj_num', [100])
        vg.add('valid_traj_num', [10])
        
        vg.add('use_batch_norm', [False])
        vg.add('dropout', [False])
        vg.add('set_eval_for_batch_norm', [False])
        vg.add('track_running_stats', [True])
        vg.add('adjust_momentum', [False])

        vg.add('downsample', [False])
        vg.add('radius_list', [
            [[voxel_size*2, voxel_size*4], [voxel_size*4, voxel_size*8], [voxel_size*8, voxel_size*16], [voxel_size*16, voxel_size*32]],
        ])
        vg.add('npoint_list', [
            [6000, 1500, 375, 95],
        ])
        vg.add('layer', [
            1
        ])
        vg.add('sample_point_1_list', [
            [16, 16, 16, 16],
        ])
        vg.add('sample_point_2_list', [
            [32, 32, 32, 32],
        ])

        vg.add('npoint', [2500])
        vg.add('batch_size', [16])
        vg.add('mlp1_size', [[32, 64, 64]])
        vg.add('mlp2_size', [[32, 64, 64]])
        vg.add('interpolation_mlp_size', [[128, 128, 128]])

        vg.add('correct_z_rotation', [2])
        vg.add('epoch', [500])
        vg.add('num_worker', [1])
        vg.add('train_pos_label_weight', [8, 11.2, 15])
        vg.add('seed', [100])
        vg.add('manual_lr_adjust', [False])
        vg.add('schedule_lr', [True])
        vg.add('data_dir', ['2021-11-10-variation-1000'])
        vg.add('learning_rate', [0.001])
        vg.add('force_loss_mode', ['balance'])
        vg.add('normal_loss_mode', ['balance'])
        vg.add('force_loss_weight', [1])
        vg.add('normal_loss_weight', [1])
        vg.add('plot_interval', [25])
        vg.add('separate_model', [True])
        # vg.add('load_dir', ['./data/seuss/1023-pn-force-non-shared/1023-pn-force-non-shared_2021_10_23_23_59_23_0002'])
        vg.add('load_dir', [None])
        vg.add('train', [True])
        exp_prefix += 'debug'

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
