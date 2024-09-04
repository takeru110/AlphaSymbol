#!/bin/bash
curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-py39_24.7.1-0-Linux-x86_64.sh
chmod +x ~/miniconda.sh
~/miniconda.sh -b -p /opt/conda
rm ~/miniconda.sh
/opt/conda/bin/conda install numpy pyyaml scipy cython jupyter ipython mkl mkl-include
/opt/conda/bin/conda install -c soumith magma-cuda90
/opt/conda/bin/conda install pytorch=0.4.1 -c pytorch
/opt/conda/bin/conda clean -ya