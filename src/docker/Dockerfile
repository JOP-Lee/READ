# Base packages for ubuntu
# FROM nvidia/vulkan:1.1.121-cuda-10.1--ubuntu18.04
# FROM nvidia/cudagl:10.1-devel-ubuntu18.04

FROM nvidia/cudagl:11.3.1-devel-ubuntu18.04

# clean the libs list
RUN apt-get clean

# RUN apt-key adv --recv-keys --keyserver keyserver.ubuntu.com A4B469963BF863CC
# RUN packages='libsdl2-2.0 xserver-xorg libvulkan1' \
#   && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y $packages --no-install-recommends \
#     && VULKAN_API_VERSION=`dpkg -s libvulkan1 | grep -oP 'Version: [0-9|\.]+' | grep -oP '[0-9|\.]+'` && \
#   mkdir -p /etc/vulkan/icd.d/ && \
#   echo \
#   "{\
#     \"file_format_version\" : \"1.0.0\",\
#     \"ICD\": {\
#       \"library_path\": \"libGLX_nvidia.so.0\",\
#       \"api_version\" : \"${VULKAN_API_VERSION}\"\
#     }\
#   }" > /etc/vulkan/icd.d/nvidia_icd.json \
#   && rm -rf /var/lib/apt/lists/*
  
RUN apt-get update -q
RUN apt-get install -y \
    git \
    wget \
    bzip2 \
    htop \
    vim \
    nano \
    g++ \
    make \
    build-essential \
    software-properties-common \
    apt-transport-https \
    sudo \
    gosu \
    libgl1-mesa-glx \
    graphviz \
    curl \
    libglew-dev \
    libglfw3-dev \
    libjpeg-dev \
    libjsoncpp-dev \
    libpng-dev \
    mesa-utils \
    xorg-dev \
    xvfb \
    ne 
    # lxde \
    # x11vnc \

## Download and install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O /tmp/miniconda.sh

RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh 

ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN conda list 

## Additional packages
RUN pip install torch==1.10.2+cu113 \
                torchvision==0.11.3+cu113 \
                torchaudio==0.10.2+cu113 \
                -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install tensorboardX \
                numpy \
                pyyaml \
                pillow \
                munch \
                scipy \
                scikit-image \
                scikit-learn \
                matplotlib \
                trimesh \
                tqdm \
                h5py \
                Cython \
                opencv-python \
                PyOpenGL \
                PyOpenGL_accelerate \
                huepy \
                pyglet
                # -i https://pypi.tuna.tsinghua.edu.cn/simple
                
# RUN pip install glumpy numpy-quaternion
RUN pip install git+https://github.com/DmitryUlyanov/glumpy \
                numpy-quaternion


RUN echo "carla ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN useradd -m carla
USER carla
ENV SDL_VIDEODRIVER=x11
WORKDIR /home/carla
