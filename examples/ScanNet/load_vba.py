# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob, numpy as np, torch
import plyfile
import yaml
config = yaml.safe_load(open(
    '/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/config/0444_00_400_scaled_normalized.yaml'))
files_test = sorted(glob.glob(config['original_path']))[0]
scale = 0.005

def f_test(fn):
    v=torch.load(fn)
    coords = np.ascontiguousarray((v[:, :3]-v[:, :3].mean(0)) * scale)
    colors = np.ascontiguousarray(v[:, 3:6])/127.5-1
    torch.save((coords, colors), fn[:-4]+config['modifiers']+'.pth')
    # coords=np.ascontiguousarray(v[:, :3])
    # colors=np.ascontiguousarray(v[:, 3:6])
    # torch.save((coords,colors),fn[:-4]+'_reshaped.pth')
    
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
    ply.write(fn[:-4] + config['modifiers'] + ".ply")
    print(fn)


f_test(files_test)

# files_test = sorted(glob.glob(
#     '/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/scene0000_00_vh_clean_2.pth'))[0]


# def f_test(fn):
#     v = torch.load(fn)
#     coords = v[0]
#     colors = v[1]

#     vertices = np.empty(len(coords), dtype=[(
#         'x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
#     vertices['x'] = coords[:, 0].astype('f4')
#     vertices['y'] = coords[:, 1].astype('f4')
#     vertices['z'] = coords[:, 2].astype('f4')
#     vertices['red'] = colors[:, 0].astype('u1')
#     vertices['green'] = colors[:, 1].astype('u1')
#     vertices['blue'] = colors[:, 2].astype('u1')

#     ply = plyfile.PlyData(
#         [plyfile.PlyElement.describe(vertices, 'vertex')], text=False)
#     ply.write(fn[:-4] + "_normalized.ply")
#     print(fn)


# f_test(files_test)
