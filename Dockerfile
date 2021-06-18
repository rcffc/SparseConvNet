FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel as base

RUN apt-get update && \
    apt-get install -y \
    git g++ ninja-build unrar

RUN conda install -c conda-forge plyfile debugpy scipy

RUN usermod -a -G video root

WORKDIR /app
RUN git clone https://github.com/rcffc/SparseConvNet
RUN mv SparseConvNet src
WORKDIR /app/src
ENV MAX_JOBS=12

RUN rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
RUN python setup.py develop
##################################################
FROM base as debugger

RUN apt-get update && \
    apt-get install -y \
    gdb
COPY . /app/src/
# ENTRYPOINT [ "python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "--log-to-stderr", "/app/src/examples/ScanNet/unet.py"]
# ENTRYPOINT [ "python -m debugpy --listen 0.0.0.0:5678 --wait-for-client --log-to-stderr /app/src/examples/ScanNet/unet.py"]

# python /app/src/setup.py develop && python -m debugpy --listen 0.0.0.0:5679 --wait-for-client --log-to-stderr /app/src/examples/ScanNet/data.py
#################################################
FROM base as prepare_data

COPY . /app/src/
ENTRYPOINT [ "python", "-m", "debugpy", "--listen", "0.0.0.0:5679", "--wait-for-client", "--log-to-stderr", "/app/src/examples/ScanNet/prepare_data.py"]



# RUN apt-get update && \
#     apt-get install -y \
#     wget

# # install miniconda
# ENV MINICONDA_VERSION latest
# ENV CONDA_DIR $HOME/miniconda3
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
#     chmod +x ~/miniconda.sh && \
#     ~/miniconda.sh -b -p $CONDA_DIR && \
#     rm ~/miniconda.sh
# ENV PATH=$CONDA_DIR/bin:$PATH

# RUN conda create -n sparseconvnet python=3.6 --channel free

# SHELL ["conda", "run", "-n", "sparseconvnet", "/bin/bash", "-c"]

# RUN conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge