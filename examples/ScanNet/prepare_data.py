# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob, plyfile, numpy as np, multiprocessing as mp, torch
import yaml
# config = yaml.safe_load(open('/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/config/0444_00_400_scaled_normalized_transformed.yaml'))
# config = yaml.safe_load(open('/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/config/scene0444_00_vh_clean_2.yaml'))

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper=np.ones(150)*(-100)
for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
    remapper[x]=i

def filter_train(files):
    with open('./failed_scenes_train.txt', 'r') as failed:
        failed_lines = [s.strip() for s in failed.readlines()]
        files = [f for f in files if f[len('/opt/datasets/train/scenes/multi/labeled/'):-len('.pth')] not in failed_lines]
        return files

def filter_val(files):
    with open('./failed_scenes_val.txt', 'r') as failed:
        failed_lines = [s.strip() for s in failed.readlines()]
        files = [f for f in files if f[len('/opt/datasets/val/scenes/multi/labeled/'):-len('.pth')] not in failed_lines]
        return files

# TODO read the .pth
files_train=sorted(filter_train(glob.glob('/opt/datasets/train/scenes/multi/labeled/*.pth')))
files_labeled_train=sorted(filter_train(glob.glob('/opt/datasets/train/scenes/multi/labeled/*.ply')))

files_val=sorted(filter_val(glob.glob('/opt/datasets/val/scenes/multi/labeled/*.pth')))
files_labeled_val=sorted(filter_val(glob.glob('/opt/datasets/val/scenes/multi/labeled/*.ply')))

SAVE_MASK_TRAIN = '/opt/datasets/train/scenes/multi/normalized/{}.pth'
SAVE_MASK_VAL = '/opt/datasets/val/scenes/multi/normalized/{}.pth'

assert len(files_train) == len(files_labeled_train)
assert len(files_val) == len(files_labeled_val)

def write_ply(coords, colors, name):
    vertices = np.empty(len(coords), dtype=[(
        'x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices['x'] = coords[:, 0].astype('f4')
    vertices['y'] = coords[:, 1].astype('f4')
    vertices['z'] = coords[:, 2].astype('f4')
    vertices['red'] = colors[:, 0].astype('u1')
    vertices['green'] = colors[:, 1].astype('u1')
    vertices['blue'] = colors[:, 2].astype('u1')

    ply = plyfile.PlyData(
        [plyfile.PlyElement.describe(vertices, 'vertex')], text=False)
    ply.write(name)


def f_train(fn):
    v = torch.load(fn)
    coords = np.ascontiguousarray((v[0]-v[0].mean(0)), dtype=np.float32)
    colors = np.ascontiguousarray(v[1], dtype=np.float32)/127.5-1
    w = np.ascontiguousarray(v[2], dtype=np.float32)
    id = fn[len('/opt/datasets/train/scenes/multi/labeled/'):-len('.pth')]
    torch.save((coords, colors, w), SAVE_MASK_TRAIN.format(id))

def f_val(fn):
    v = torch.load(fn)
    coords = np.ascontiguousarray((v[0]-v[0].mean(0)), dtype=np.float32)
    colors = np.ascontiguousarray(v[1], dtype=np.float32)/127.5-1
    w = np.ascontiguousarray(v[2], dtype=np.float32)
    id = fn[len('/opt/datasets/val/scenes/multi/labeled/'):-len('.pth')]
    torch.save((coords, colors, w), SAVE_MASK_VAL.format(id))


p = mp.Pool(processes=mp.cpu_count())
p.map(f_train,files_train)
p.close()
p.join()

p = mp.Pool(processes=mp.cpu_count())
p.map(f_val,files_val)
p.close()
p.join()
