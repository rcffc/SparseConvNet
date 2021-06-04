FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel as base

RUN apt-get update && \
    apt-get install -y \
    git g++ ninja-build unrar

RUN conda install -c conda-forge plyfile debugpy scipy

RUN usermod -a -G video root

WORKDIR /app
RUN git clone https://github.com/facebookresearch/SparseConvNet
RUN mv SparseConvNet src
WORKDIR /app/src
ENV MAX_JOBS=12

RUN rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
RUN python setup.py develop
# CMD [ "python", "/app/src/setup.py", "develop"]
# ENTRYPOINT [ "python", "/app/src/setup.py", "develop"]
