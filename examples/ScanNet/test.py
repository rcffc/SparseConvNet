# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
m = 32 # 16 or 32
residual_blocks=True #True or False
block_reps = 2 #Conv block repetition factor: 1 or 2

import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys, glob
import math
import numpy as np
import plyfile
import yaml
import multiprocessing as mp

# use_cuda = torch.cuda.is_available()
use_cuda = False
# exp_name=os.path.join('examples','ScanNet', 'results','unet_scale20_m16_rep1_notResidualBlocks')
exp_name = os.path.join('/app', 'results', 'unet_scale50_m32_rep2_residualBlocks_batchSize5_finetune')
# config = yaml.safe_load(open('/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/config/0444_00_400_scaled_normalized.yaml'))
# config = yaml.safe_load(open('/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/config/0444_00_100_scaled_normalized_0.9_modulo8.yaml'))
# config = yaml.safe_load(open('/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/config/scene0444_00_vh_clean_2.yaml'))

scale=50  #Voxel size = 1/scale
val_reps=3 # Number of test views, 1 or more
test_reps=1
batch_size=5
elastic_deformation=False
dimension=3
full_scale=4096 #Input field size


def testMerge(i):
    locs = []
    feats = []
    point_ids = []
    a, b, _ = i
    m = np.eye(3)
    m[0][0]=-1
    m[1][1]=-1
    m *= scale
    a = np.matmul(a,m)+full_scale/2
    idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
    a = torch.from_numpy(a).long()
    locs.append(torch.cat([a, torch.LongTensor(a.shape[0], 1).fill_(0)], 1))
    feats.append(torch.from_numpy(b))
    point_ids.append(torch.from_numpy(np.nonzero(idxs)[0]))
    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    point_ids=torch.cat(point_ids,0)
    return {'x': [locs, feats], 'id': i, 'point_ids': point_ids}

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension,full_scale, mode=4)).add(
           scn.SubmanifoldConvolution(dimension, 3, m, 3, False)).add(
               scn.UNet(dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(dimension))
        self.linear = nn.Linear(m, 20)
    def forward(self,x):
        x=self.sparseModel(x)
        x=self.linear(x)
        return x

unet=Model()
if use_cuda:
    unet=unet.cuda()

training_epochs=512
training_epoch=scn.checkpoint_restore(unet,exp_name,'unet',use_cuda)
optimizer = optim.Adam(unet.parameters())
print('#classifer parameters', sum([x.nelement() for x in unet.parameters()]))


def write_ply(labels, positions, path):
    global color_dict    
    colors = np.array([color_dict.__getitem__(label) for label in labels])

    vertices = np.empty(len(labels), dtype=[(
        'x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('label', 'i4')])
    vertices['x'] = positions[:, 0].astype('f4')
    vertices['y'] = positions[:, 1].astype('f4')
    vertices['z'] = positions[:, 2].astype('f4')
    vertices['red'] = colors[:, 0].astype('u1')
    vertices['green'] = colors[:, 1].astype('u1')
    vertices['blue'] = colors[:, 2].astype('u1')
    vertices['label'] = labels.astype('i4')
    
    id = path[len('/opt/datasets/val/scenes/multi/normalized/'):-len('.pth')]
    ply = plyfile.PlyData(
        [plyfile.PlyElement.describe(vertices, 'vertex')], text=False)
    ply.write('/opt/datasets/val/scenes/multi/predicted/' + id + '.ply')


color_dict={}
color_dict[0] = np.array([174, 199, 232])
color_dict[1] = np.array([152, 223, 138])
color_dict[2] = np.array([31, 119, 180])
color_dict[3] = np.array([255, 187, 120])
color_dict[4] = np.array([188, 189,  34])
color_dict[5] = np.array([140,  86,  75])
color_dict[6] = np.array([255, 152, 150])
color_dict[7] = np.array([214,  39,  40])
color_dict[8] = np.array([197, 176, 213])
color_dict[9] = np.array([148, 103, 189])
color_dict[10] = np.array([196, 156, 148])
color_dict[11] = np.array([23, 190, 207])
color_dict[12] = np.array([247, 182, 210])
color_dict[13] = np.array([219, 219, 141])
color_dict[14] = np.array([255, 127,  14])
color_dict[15] = np.array([158, 218, 229])
color_dict[16] = np.array([44, 160,  44])
color_dict[17] = np.array([112, 128, 144])
color_dict[18] = np.array([227, 119, 194])
color_dict[19] = np.array([82,  84, 163])

with torch.no_grad():
    unet.eval()
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    for path in glob.glob('/opt/datasets/val/scenes/multi/normalized/*.pth'):
        
        test = []
        for x in torch.utils.data.DataLoader(
            [path],
            collate_fn=lambda 
            x: torch.load(x[0]), 
            num_workers=1
            ):
            test.append(x) 

        test_data_loader = torch.utils.data.DataLoader(
            test,
            batch_size=None,
            collate_fn=testMerge,
            num_workers=1,
            shuffle=True,
            worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
        )

        start = time.time()
        for rep in range(1,1+test_reps):
            for batch in test_data_loader:
                if use_cuda:
                    batch['x'][1]=batch['x'][1].cuda()
                predictions=unet(batch['x'])
                labels = torch.argmax(predictions, 1)
                end = time.time() - start
                write_ply(labels.cpu().numpy(), batch['id'][0], path)
            print(training_epoch-1,rep,'Test MegaMulAdd=',scn.forward_pass_multiplyAdd_count/len(test)/1e6, 'MegaHidden',scn.forward_pass_hidden_states/len(test)/1e6,'time=', end,'s')
