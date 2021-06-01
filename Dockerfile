FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel as base

USER root

RUN apt-get update && \
    apt-get install -y \
    git g++ ninja-build unrar

RUN conda install -c conda-forge plyfile debugpy scipy

# RUN export PATH="$PATH:/home/user/.local/bin"
# RUN useradd -ms /bin/bash user && echo "user:password" | chpasswd

RUN usermod -a -G video root

WORKDIR /app
RUN git clone https://github.com/facebookresearch/SparseConvNet
RUN mv SparseConvNet src
WORKDIR /app/src
RUN bash develop.sh

##################################################
FROM base as debugger

COPY . /app/src
WORKDIR /app/src
ENTRYPOINT [ "python3", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "--log-to-stderr", "/app/src/examples/ScanNet/unet.py"]


###################################################
FROM base as cleanup
COPY . /app/src