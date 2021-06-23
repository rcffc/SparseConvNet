# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob, numpy as np, torch

files_test=sorted(glob.glob('/igd/a4/homestud/pejiang/repos/SparseConvNet/examples/ScanNet/0707_00_1.pth'))[0]

def f_test(fn):
    v=torch.load(fn)
    coords=np.ascontiguousarray(v[:,:3]-v[:,:3].mean(0))
    colors=np.ascontiguousarray(v[:,4:7])/127.5-1
    torch.save((coords,colors),fn[:-4]+'_normalized.pth')
    print(fn)


f_test(files_test)
