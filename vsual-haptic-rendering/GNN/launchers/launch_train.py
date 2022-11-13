import time
import click
import socket
import multiprocessing as mp
from chester.run_exp import run_experiment_lite, VariantGenerator
from haptic.GNN.train_haptic import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '1116-gnn-1000'
    exp_prefix = '1117-gnn-1000-force-tune-layer-load-contact'
    exp_prefix = '1117-gnn-1000-force-larger-radius'
    exp_prefix = '1120-gnn-1000-larger-voxel-vary-radius'
    exp_prefix = '1120-gnn-50-overfit'
    exp_prefix = '1121-gnn-1000-local'
    vg = VariantGenerator()

    if not debug:
        vg.add('algo', ['gnn'])
        vg.add('epoch', [500])
        vg.add('train_traj_num', [1000])
        vg.add('valid_traj_num', [100])

        vg.add('cuda_idx', [1])
        vg.add('filter_edge', [2])

        vg.add('neighbor_radius', [0.00625 * 6])

        vg.add('node_dim', [7])
        vg.add('edge_dim', [4])
        
        vg.add('learning_rate', [0.001])    
        vg.add('contact_proc_layer', [2])
        vg.add('force_proc_layer', [2])

        vg.add('global_size', [128])
        vg.add('batch_size', [32])
        
        vg.add('seed', [100])
        # vg.add('data_dir', ['2021-11-10-variation-1000'])
        vg.add('data_dir', ['2021-11-19-larger-voxel-size'])
        # vg.add('data_dir', ['2021-11-20-same-obj-different-traj-gnn'])
        vg.add('force_loss_mode', ['contact']) 
        vg.add('train_pos_label_weight', [11.2])
        vg.add('contact_loss_balance', [0])
        vg.add('plot_img_interval', [5])
        vg.add('plot_video_interval', [10])
        vg.add('save_interval', [10])

        vg.add('load_model', [True])
        vg.add('load_name', ['data/seuss/1120-gnn-1000-larger-voxel-vary-force-layer/1120-gnn-1000-larger-voxel-vary-force-layer_2021_11_19_23_14_48_0001/best_model.pth'])
        vg.add('load_contact', [True])
        vg.add('load_force', [False])

        vg.add('train_contact', [False])
        vg.add('train_force', [True])
    else:
        vg.add('algo', ['gnn'])
        vg.add('epoch', [500])
        vg.add('train_traj_num', [1000])
        vg.add('valid_traj_num', [100])

        vg.add('cuda_idx', [1])
        vg.add('filter_edge', [2])

        vg.add('neighbor_radius', [0.00625 * 6])

        vg.add('node_dim', [7])
        vg.add('edge_dim', [4])
        
        vg.add('learning_rate', [0.001])    
        vg.add('contact_proc_layer', [2])
        vg.add('force_proc_layer', [4])

        vg.add('global_size', [128])
        vg.add('batch_size', [15])
        
        vg.add('seed', [100])
        # vg.add('data_dir', ['2021-11-10-variation-1000'])
        vg.add('data_dir', ['2021-11-19-larger-voxel-size'])
        # vg.add('data_dir', ['2021-11-20-same-obj-different-traj-gnn'])
        vg.add('force_loss_mode', ['balance']) 
        vg.add('train_pos_label_weight', [11.2])
        vg.add('contact_loss_balance', [0])
        vg.add('plot_img_interval', [1000])
        vg.add('plot_video_interval', [1000])
        vg.add('save_interval', [10])

        vg.add('load_model', [True])
        vg.add('load_name', ['data/seuss/1120-gnn-1000-larger-voxel-vary-force-layer/1120-gnn-1000-larger-voxel-vary-force-layer_2021_11_19_23_14_48_0001/best_model.pth'])
        vg.add('load_contact', [True])
        vg.add('load_force', [False])

        vg.add('train_contact', [False])
        vg.add('train_force', [True])
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
