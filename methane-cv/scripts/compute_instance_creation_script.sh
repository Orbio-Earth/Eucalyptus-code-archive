#!/bin/bash
set -e
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

set -e
cd "$parent_path"
echo $(pwd)

printf "\n\n# Added by compute instance creation script\n" >> ~/.bashrc
printf "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/anaconda/lib\n\n" >> ~/.bashrc

conda update -y -n base conda
conda install -y -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create --file ../conda_env.yaml

conda init
conda activate methane-cv

ipython kernel install --user --name methane-cv