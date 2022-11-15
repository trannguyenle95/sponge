import subprocess
from colorama import Fore, Back, Style


objects = ["plate_ycb","bowl_ycb","bowl_shapenet"]
num_iters = 100
for object_name in objects:
    for i in range(num_iters):
        p = subprocess.Popen(['python', 'sim_sponge.py', '--object',
        object_name, '--num_envs', '10', '--run_headless', 'True'])
        p.wait()
        print(Fore.GREEN + "Done iter: " + str(i) + Style.RESET_ALL)
    print(Fore.GREEN + "DONE! Switching to new object"+ Style.RESET_ALL)
