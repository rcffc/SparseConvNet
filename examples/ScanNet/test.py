# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
m = 16 # 16 or 32
residual_blocks=False #True or False
block_reps = 1 #Conv block repetition factor: 1 or 2

import collections
import torch, data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys, glob
import math
import numpy as np
import plyfile

use_cuda = torch.cuda.is_available()
exp_name=os.path.join('examples','ScanNet', 'results','unet_scale20_m16_rep1_notResidualBlocks')

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(data.dimension,data.full_scale, mode=4)).add(
           scn.SubmanifoldConvolution(data.dimension, 3, m, 3, False)).add(
               scn.UNet(data.dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(data.dimension))
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


def write_ply(labels, positions):
    global color_dict    
    colors = np.array([color_dict.__getitem__(label) for label in labels])

    vertices = np.empty(len(labels), dtype=[(
        'x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices['x'] = positions[:, 0].astype('f4')
    vertices['y'] = positions[:, 1].astype('f4')
    vertices['z'] = positions[:, 2].astype('f4')
    vertices['red'] = colors[:, 0].astype('u1')
    vertices['green'] = colors[:, 1].astype('u1')
    vertices['blue'] = colors[:, 2].astype('u1')

    ply = plyfile.PlyData(
        [plyfile.PlyElement.describe(vertices, 'vertex')], text=False)
    ply.write("result.ply")


color_dict={}
for i in range(0, 20):
    color_dict[i] = np.random.randint(low=0, high=255, size=(3))

with torch.no_grad():
    unet.eval()
    store=torch.zeros(data.testOffsets[-1],20)
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    for rep in range(1,1+data.test_reps):
        for i,batch in enumerate(data.test_data_loader):
            if use_cuda:
                batch['x'][1]=batch['x'][1].cuda()
            predictions=unet(batch['x'])
            labels = torch.argmax(predictions, 1)
            end = time.time() - start
            write_ply(labels.cpu().numpy(), batch['x'][0].cpu().numpy())
            store.index_add_(0,batch['point_ids'],predictions.cpu())
        print(0,rep,'Test MegaMulAdd=',scn.forward_pass_multiplyAdd_count/len(data.test)/1e6, 'MegaHidden',scn.forward_pass_hidden_states/len(data.test)/1e6,'time=', end,'s')


