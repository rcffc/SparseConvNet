#!/bin/bash
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
env MAX_JOBS=12 FORCE_CUDA=1 python setup.py develop
env MAX_JOBS=12 python examples/hello-world.py
