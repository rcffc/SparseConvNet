# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob, plyfile, numpy as np, multiprocessing as mp, torch
import yaml
# config = yaml.safe_load(open('/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/config/0444_00_400_scaled_normalized_transformed.yaml'))
config = yaml.safe_load(open('/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/config/scene0444_00_vh_clean_2.yaml'))

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper=np.ones(150)*(-100)
for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
    remapper[x]=i

# files=sorted(glob.glob('../data/*/*_vh_clean_2.ply'))
# files2=sorted(glob.glob('../data/*/*_vh_clean_2.labels.ply'))

# files_test=sorted(glob.glob('/opt/datasets/scannetv2_sparseconvnet/test/*_vh_clean_2.ply'))
# files_test=sorted(glob.glob('/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/0.ply'))
# files_test = sorted(glob.glob(
#     '/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/pointclouds/scene0444_00_vh_clean_2.ply'))
# files_test = sorted(glob.glob(
#     '/igd/a4/homestud/pejiang/ScanNet/scans/scene0444_00/scene0444_00_vh_clean_2_edited.ply'))
# files_test = sorted(glob.glob(
#     '/igd/a4/homestud/pejiang/scenes/multi/0444_00_400_edited.ply'))
files_test = sorted(glob.glob(config['ply']))

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

def f(fn):
    fn2 = fn[:-3]+'labels.ply'
    a=plyfile.PlyData().read(fn)
    v=np.array([list(x) for x in a.elements[0]])
    coords=np.ascontiguousarray(v[:,:3]-v[:,:3].mean(0))
    colors=np.ascontiguousarray(v[:,3:6])/127.5-1
    a=plyfile.PlyData().read(fn2)
    w=remapper[np.array(a.elements[0]['label'])]
    torch.save((coords, colors, w), fn[:-4]+'_normalized.pth')
    write_ply(coords,colors, fn[:-4] + "_normalized.ply")
    print(fn, fn2)

def f_test(fn):
    a=plyfile.PlyData().read(fn)
    v=np.array([list(x) for x in a.elements[0]])
    coords=np.ascontiguousarray(v[:,:3]-v[:,:3].mean(0))
    colors=np.ascontiguousarray(v[:,3:6])/127.5-1
    # coords=np.ascontiguousarray(v[:, :3])
    # colors=np.ascontiguousarray(v[:, 3:6])

    torch.save((coords, colors), fn[:-4] + config['modifiers'] + '.pth')
    # write_ply(coords, colors, fn[:-4] + "_normalized.ply")
    print(fn)

p = mp.Pool(processes=mp.cpu_count())
p.map(f_test,files_test)
p.close()
p.join()
