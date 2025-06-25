FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y wget make

ENV TZ=Europe/Berlin \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/src:"$PYTHONPATH" \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/opt/conda/lib \
    PATH=/opt/conda/bin:$PATH

WORKDIR /orbio

RUN mkdir -p /opt/conda  &&\
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/conda/miniconda.sh  &&\
    bash /opt/conda/miniconda.sh -b -u -p /opt/conda  &&\
    rm -rf /opt/conda/miniconda.sh  &&\
    /opt/conda/bin/conda init bash

RUN source ~/.bashrc

RUN conda update -n base conda  &&\
    conda install -n base conda-libmamba-solver  &&\
    conda config --set solver libmamba

RUN mkdir -p /orbio/methane-cv
COPY methane-cv/conda_env.yaml ./methane-cv/
RUN conda env create --file ./methane-cv/conda_env.yaml &&\
    conda clean --all -y
