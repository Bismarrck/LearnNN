#!/usr/bin/env bash

# Add CUDA runtime to repo
if ! [ -f cuda-repo-ubuntu1604_8.0.44-1_amd64.deb ]
then
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
fi
dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
apt-get update

# Install components
apt-get install cuda python python-setuptools python-scipy python-h5py unzip

# Install python packages
easy_install pip
pip install pymatgen scikit-learn ipython

# Install tensorflow
if [ -f tensorflow_gpu-0.11.0-cp27-cp27mu-linux_x86_64.whl ]; then
    pip install tensorflow_gpu-0.11.0-cp27-cp27mu-linux_x86_64.whl
fi

# Setup cuDNN
if [ -f cudnn-8.0-linux-x64-v5.1.tgz ]; then
    if [ -d cuda ]; then
        mv cuda cuda.bak
    fi
    tar -xvf cudnn-8.0-linux-x64-v5.1.tgz
    cp -f cuda/include/cudnn.h /usr/local/cuda/include
    cp -f cuda/lib64/libcudnn* /usr/local/cuda/lib64/
    chmod a+r /usr/local/cuda/include/cudnn.h
    chmod a+r /usr/local/cuda/lib64/libcudnn*
fi

# Extract the training data
if [ -f b20pbe.zip ]; then
    unzip b20pbes.zip
    mv data training
fi

# Create the env profile
echo '#!/bin/sh' > profile.dev
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"' >> profile.dev
echo 'export CUDA_HOME=/usr/local/cuda' >> profile.dev
chmod 755 profile.dev
