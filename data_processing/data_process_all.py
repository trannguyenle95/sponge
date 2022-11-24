import subprocess
from colorama import Fore, Back, Style


objects = ["plate_ycb","bowl_ycb","bowl_shapenet"]
for object_name in objects:
    p = subprocess.Popen(['python', 'data_process.py', '--object',object_name])
    p.wait()
print(Fore.GREEN + "DONE! Switching to new object"+ Style.RESET_ALL)
