import os
import shutil

path = '/opt/datasets/scans'
with open('examples/ScanNet/datasplit/scannetv2_val.txt', 'r') as source:
    val_scenes = source.read().splitlines()

    for scene in os.listdir(path):
        if scene in val_scenes:
            shutil.rmtree(os.path.join(path, scene))