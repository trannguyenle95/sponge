import time
import click
import socket
import multiprocessing as mp
from chester.run_exp import run_experiment_lite, VariantGenerator
from haptic.pointNet_geo.pointnet2_segmentation_haptic import run_task

def get_feature_num(add_gripper_vel, add_cloth_vel):
    base_num = 4
    if add_gripper_vel:
        base_num += 3
    if add_cloth_vel:
        base_num += 3
    return base_num

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    # exp_prefix = '1119-geo-pointnet-larger-voxel-vary-force-layer'
    exp_prefix = '1125-geo-pointnet-larger-voxel-add-gripper-vel'
    exp_prefix = '1126-pointnet-larger-voxel-normalize-force-target'
    exp_prefix = '1128-pointnet-gt-particle'
    vg = VariantGenerator()

    if not debug:
        vg.add('epoch', [500])
        vg.add('train_traj_num', [1000])
        vg.add('valid_traj_num', [100])
        vg.add('batch_size', [16])
        vg.add('use_batch_norm', [False])
        vg.add('add_gripper_vel', [False])
        vg.add('add_cloth_vel', [True])
        vg.add('feature_num', lambda add_gripper_vel, add_cloth_vel: [get_feature_num(add_gripper_vel, add_cloth_vel)])
        vg.add('force_normalize_mean', [None])
        vg.add('force_normalize_std', [None])

        vg.add('contact_sa_radius', [
            [0.05, 0.1]
        ])
        vg.add('contact_num_layer', [3])
        vg.add('contact_sa_ratio', [[0.2, 0.25]])

        vg.add('force_num_layer', [4])
        vg.add('force_sa_radius', [
            [0.05, 0.1, 0.2]
            # [0.05, 0.1]
        ])
        vg.add('force_sa_ratio', [
            [0.4, 0.5, 0.6],
            # [0.4, 0.5],
        ])
        vg.add('force_sa_mlp_list', [
            [[64, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 1024]],
            # [[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
        ])
        vg.add('force_fp_mlp_list', [
            [[256, 256], [256, 256], [256, 256, 128], [128, 128, 128]],
            # [[256, 256], [256, 128], [128, 128, 128]]
        ])
        vg.add('force_linear_mlp_list', [
            [128, 128, 128]
            # [128, 128]
        ])
        vg.add('force_fp_k', [
            [3, 3, 3, 3]
            # [1, 3, 3]
        ])

        vg.add('contact_residual', [False])
        vg.add('force_residual', [False])

        vg.add('cuda_idx', [0])
        # vg.add('load_contact_name', ['/data/yufeiw2/softagent_perspective/data/local/1119-geo-pointnet-larger-voxel-vary-force-layer/1119-geo-pointnet-larger-voxel-vary-force-layer_2021_11_19_23_43_33_0001/checkpoints/best_model.pth'])
        vg.add('load_contact_name', [None])
        vg.add('load_force_name', [None])
        vg.add('train_contact', [True])
        vg.add('train_force', [True])
        vg.add('use_gt_contact', [False])

        vg.add('lr', [0.0001, 0.0005])    
        
        vg.add('seed', [100])
        # vg.add('data_dir', ['2021-11-10-variation-1000'])
        # vg.add('data_dir', ['2021-11-19-larger-voxel-size'])
        vg.add('data_dir', ['2021-11-27-gt-cloth-particle-with-vel'])
        vg.add('force_loss_mode', ['contact', 'balance'])
        vg.add('train_pos_label_weight', [11.2])
        vg.add('plot_img_interval', [1000])
        vg.add('plot_gif_interval', [1000])
        vg.add('save_interval', [1000])
        vg.add('only_eval', [False])
    else:
        vg.add('epoch', [500])
        vg.add('train_traj_num', [100])
        vg.add('valid_traj_num', [10])
        vg.add('batch_size', [16])
        vg.add('use_batch_norm', [False])
        vg.add('add_gripper_vel', [False])
        vg.add('add_cloth_vel', [True])
        vg.add('feature_num', lambda add_gripper_vel, add_cloth_vel: [get_feature_num(add_gripper_vel, add_cloth_vel)])
        vg.add('force_normalize_mean', [0.015])
        vg.add('force_normalize_std', [0.01])

        vg.add('contact_sa_radius', [
            [0.05, 0.1]
        ])
        vg.add('contact_num_layer', [3])
        vg.add('contact_sa_ratio', [[0.2, 0.25]])

        vg.add('force_num_layer', [4])
        vg.add('force_sa_radius', [
            [0.05, 0.1, 0.2]
            # [0.05, 0.1]
        ])
        vg.add('force_sa_ratio', [
            [0.4, 0.5, 0.6],
            # [0.4, 0.5],
        ])
        vg.add('force_sa_mlp_list', [
            [[64, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 1024]],
            # [[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
        ])
        vg.add('force_fp_mlp_list', [
            [[256, 256], [256, 256], [256, 256, 128], [128, 128, 128]],
            # [[256, 256], [256, 128], [128, 128, 128]]
        ])
        vg.add('force_linear_mlp_list', [
            [128, 128, 128]
            # [128, 128]
        ])
        vg.add('force_fp_k', [
            [3, 3, 3, 3]
            # [1, 3, 3]
        ])

        vg.add('contact_residual', [False])
        vg.add('force_residual', [False])

        vg.add('cuda_idx', [0])
        vg.add('load_contact_name', [None])
        vg.add('load_force_name', [None])
        vg.add('train_contact', [True])
        vg.add('train_force', [True])
        vg.add('use_gt_contact', [False])

        vg.add('lr', [0.0001])    
        
        vg.add('seed', [100])
        # vg.add('data_dir', ['2021-11-10-variation-1000'])
        # vg.add('data_dir', ['2021-11-19-larger-voxel-size'])
        vg.add('data_dir', ['2021-11-27-gt-cloth-particle-with-vel'])
        vg.add('force_loss_mode', ['contact'])
        vg.add('train_pos_label_weight', [11.2])
        vg.add('plot_img_interval', [1])
        vg.add('plot_gif_interval', [20])
        vg.add('save_interval', [20])
        vg.add('only_eval', [False])
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
