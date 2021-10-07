import numpy as np

def read_file(path):
    with open(path) as file:
        lines = [s.strip().split(' ') for s in file.readlines()]
        X = np.array([float(line[8]) for line in lines])
        print(np.mean(X))


read_file('/opt/datasets/val/scenes/multi/predicted/reconstruction_speeds.txt')