# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
scale=50  #Voxel size = 1/scale
val_reps=3 # Number of test views, 1 or more
test_reps=1
batch_size=5
elastic_deformation=False

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp, time
import yaml
# config = yaml.safe_load(open('/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/config/0444_00_400_scaled_normalized_transformed.yaml'))
# config = yaml.safe_load(open('/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/config/scene0444_00_vh_clean_2.yaml'))

dimension=3
full_scale=4096 #Input field size

# Class IDs have been mapped to the range {0,1,...,19}
# NYU_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

train,val,test=[],[],[]
# for x in torch.utils.data.DataLoader(
#         glob.glob('/opt/datasets/train/scenes/multi/normalized/*.pth'),
#         collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
#     train.append(x)
# for x in torch.utils.data.DataLoader(
#         glob.glob('/opt/datasets/val/scenes/multi/normalized/*.pth'),
#         collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
#     val.append(x)
for x in torch.utils.data.DataLoader(
        glob.glob('/opt/datasets/val/scenes/multi/normalized/*.pth'),
        collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
    test.append(x)

print('Training examples:', len(train))
print('Validation examples:', len(val))
print('Test examples:', len(test))
# Elastic distortion
blur0=np.ones((3,1,1)).astype('float32')/3
blur1=np.ones((1,3,1)).astype('float32')/3
blur2=np.ones((1,1,3)).astype('float32')/3
def elastic(x,gran,mag):
    bb=np.abs(x).max(0).astype(np.int32)//gran+3
    noise=[np.random.randn(bb[0],bb[1],bb[2]).astype('float32') for _ in range(3)]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    ax=[np.linspace(-(b-1)*gran,(b-1)*gran,b) for b in bb]
    interp=[scipy.interpolate.RegularGridInterpolator(ax,n,bounds_error=0,fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x+g(x)*mag

# def trainMerge(tbl):
#     locs=[]
#     feats=[]
#     labels=[]
#     for idx,i in enumerate(tbl):
#         a,b,c=train[i]
#         m=np.eye(3)+np.random.randn(3,3)*0.1
#         m[0][0]*=np.random.randint(0,2)*2-1
#         m*=scale
#         theta=np.random.rand()*2*math.pi
#         m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
#         a=np.matmul(a,m)
#         if elastic_deformation:
#             a=elastic(a,6*scale//50,40*scale/50)
#             a=elastic(a,20*scale//50,160*scale/50)
#         m=a.min(0)
#         M=a.max(0)
#         q=M-m
#         offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
#         a+=offset
#         idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
#         a=a[idxs]
#         b=b[idxs]
#         c=c[idxs]
#         a=torch.from_numpy(a).long()
#         locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
#         feats.append(torch.from_numpy(b)+torch.randn(3)*0.1)
#         labels.append(torch.from_numpy(c))
#     locs=torch.cat(locs,0)
#     feats=torch.cat(feats,0)
#     labels=torch.cat(labels,0)
#     return {'x': [locs,feats], 'y': labels.long(), 'id': tbl}
# train_data_loader = torch.utils.data.DataLoader(
#     list(range(len(train))),
#     batch_size=batch_size,
#     collate_fn=trainMerge,
#     num_workers=mp.cpu_count(), 
#     shuffle=True,
#     drop_last=True,
#     worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
# )

# valOffsets=[0]
# valLabels=[]
# for idx,x in enumerate(val):
#     valOffsets.append(valOffsets[-1]+x[2].size)
#     valLabels.append(x[2].astype(np.int32))
# valLabels=np.hstack(valLabels)

# def valMerge(tbl):
#     locs=[]
#     feats=[]
#     labels=[]
#     point_ids=[]
#     for idx,i in enumerate(tbl):
#         a,b,c=val[i]
#         m=np.eye(3)
#         m[0][0]*=np.random.randint(0,2)*2-1
#         m*=scale
#         theta=np.random.rand()*2*math.pi
#         m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
#         a=np.matmul(a,m)+full_scale/2+np.random.uniform(-2,2,3)
#         m=a.min(0)
#         M=a.max(0)
#         q=M-m
#         offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
#         a+=offset
#         idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
#         a=a[idxs]
#         b=b[idxs]
#         c=c[idxs]
#         a=torch.from_numpy(a).long()
#         locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
#         feats.append(torch.from_numpy(b))
#         labels.append(torch.from_numpy(c))
#         point_ids.append(torch.from_numpy(np.nonzero(idxs)[0]+valOffsets[i]))
#     locs=torch.cat(locs,0)
#     feats=torch.cat(feats,0)
#     labels=torch.cat(labels,0)
#     point_ids=torch.cat(point_ids,0)
#     return {'x': [locs,feats], 'y': labels.long(), 'id': tbl, 'point_ids': point_ids}
# val_data_loader = torch.utils.data.DataLoader(
#     list(range(len(val))),
#     batch_size=batch_size,
#     collate_fn=valMerge,
#     num_workers=mp.cpu_count(),
#     shuffle=True,
#     worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
# )


testOffsets=[0]
for idx,x in enumerate(test):
    testOffsets.append(testOffsets[-1]+x[1].size)

def testMerge(i):
    locs = []
    feats = []
    point_ids = []
    print(i[0].shape)
    a, b, _ = i[0]
    m = np.eye(3)
    m[0][0]=-1
    m[1][1]=-1
    m *= scale
    a = np.matmul(a,m)+full_scale/2
    idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
    a = torch.from_numpy(a).long()
    locs.append(torch.cat([a, torch.LongTensor(a.shape[0], 1).fill_(idx)], 1))
    feats.append(torch.from_numpy(b))
    point_ids.append(torch.from_numpy(np.nonzero(idxs)[0]+testOffsets[idx]))
    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    point_ids=torch.cat(point_ids,0)
    return {'x': [locs, feats], 'id': i, 'point_ids': point_ids}

# bla = testMerge(test[0])
test_data_loader = torch.utils.data.DataLoader(
    test,
    batch_size=None,
    collate_fn=testMerge,
    num_workers=1,
    shuffle=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)
