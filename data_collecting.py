import subprocess
from colorama import Fore, Back, Style


objects = ["bowl_ycb","plate_ycb","bowl_shapenet","bowl_a_shapenet","bowl_b_shapenet","bowl_c_shapenet","bowl_d_shapenet","bowl_e_shapenet",
           "bowl_f_shapenet", "plate_a_shapenet"]
num_iters = 100
for object_name in objects:
    for i in range(num_iters):
        p = subprocess.Popen(['python', 'sim_sponge.py', '--object',
        object_name, '--num_envs', '10', '--run_headless', 'True'])
        p.wait()
        print(Fore.GREEN + "Done iter: " + str(i) + Style.RESET_ALL)
    print(Fore.GREEN + "DONE! Switching to new object"+ Style.RESET_ALL)
