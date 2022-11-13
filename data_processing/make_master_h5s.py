"""Create master files for each of the object."""

import os
import shutil

import h5py
import numpy as np

import argparse


# Create command line flag options
parser = argparse.ArgumentParser(description='Options')
# parser.add_argument('--dir', required=True, help="Folder name for object results")
parser.add_argument('--object', required=True, help="Name of object")

args = parser.parse_args()
object_name = args.object
# if args.dir:
#     OBJ_RESULTS_DIR = args.dir

OBJ_RESULTS_DIR = "../results/"+ object_name +"_all"
MASTER_RESULTS_DIR = "../results/masters"

# Set this to the results directory for the object of interest.
all_files = os.listdir(OBJ_RESULTS_DIR)
mode_files = [os.path.join(OBJ_RESULTS_DIR, o) for o in all_files if o.endswith(
    ".h5") and "master" not in o and "iter" in o]

mode_master_file = object_name + "_master.h5"
shutil.copy(mode_files[0], os.path.join(MASTER_RESULTS_DIR, mode_master_file))

master_f = h5py.File(os.path.join(MASTER_RESULTS_DIR, mode_master_file), 'a')
dataset_names = master_f.keys()
print(dataset_names)
num_actions = master_f['action_success'].shape[0]
j = 0
for file in mode_files:
    f = h5py.File(file, 'r')
    print("======", file)
    for i in range(num_actions):
        if not np.all(f['action_success'][i] == 0.0):
            # Populate master CVS row with contents here
            for dataset_name in dataset_names:
                dataset = f[dataset_name]
                master_dataset = master_f[dataset_name]
                try:
                    master_dataset[i+j] = dataset[i]
                except BaseException:
                    pass
    j += 3
    f.close()

# # Patch in the holes
# for file in mode_files:
#     print("~~~~~~", file)
#     f = h5py.File(file, 'r')
#     for i in range(num_actions):
#         if not np.all(f['action_success'][i] == 0.0):
#             # Populate master CVS row with contents here
#             master_timedout = master_f['timed_out']
#             f_timedout = f['timed_out']

#             num_dirs = master_timedout[i].shape[0]

#             for d in range(num_dirs):
#                 if master_timedout[i][d] == 0:  # No timeout, so no need to replace
#                     continue
#                 master_timedout[i, d] = f_timedout[i, d]
#                 master_f['reorientation_meshes'][i, d, :, :,
#                                                     :] = f['reorientation_meshes'][i, d, :,
#                                                                                 :, :]
#                 master_f['shake_fail_accs'][i, d] = f['shake_fail_accs'][i, d]
#                 master_f['twist_fail_accs'][i, d] = f['twist_fail_accs'][i, d]

#     f.close()
master_f.close()
