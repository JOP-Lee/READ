#!/bin/bash

conda create -y -n READ python=3.9
conda activate READ 

conda install -y cudnn=8.2.1.32 cudatoolkit-dev=11.2 cudatoolkit=11.2 -c nvidia -c conda-forge
pip install  torch==1.9.0   torchvision==0.10.0   torchaudio==0.9.0  
 
#optional   
cd MyRender 
pip install -v -e .  
    
pip install \
    numpy \
    pyyaml \
    tensorboardX \
    munch \
    scipy \
    matplotlib \
    Cython \
    PyOpenGL \
    PyOpenGL_accelerate \
    trimesh \
    huepy \
    pillow \
    tqdm \
    scikit-learn

conda install opencv

# need to install separately
pip install \
    git+https://github.com/DmitryUlyanov/glumpy \
    numpy-quaternion

# pycuda
git clone https://github.com/inducer/pycuda
cd pycuda
git submodule update --init
export PATH=$PATH:/usr/local/cuda/bin
./configure.py --cuda-enable-gl
python setup.py install
cd ..