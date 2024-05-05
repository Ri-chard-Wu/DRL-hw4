#!/bin/bash



apt-get update 

apt install -y wget git

apt-get install -y x11-apps xauth xfonts-base
apt-get install -y libgtk-3-dev
apt-get install -y qt5-default
apt-get install -y fonts-dejavu-core

# curl -O https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
# sha256sum Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh -b
echo -e "export PATH=$PATH:/root/anaconda3/bin" >> ~/.bashrc && source ~/.bashrc
conda init bash && source ~/.bashrc

conda create -n opensim-rl -c kidzik -c conda-forge -y opensim python=3.6.1

conda activate opensim-rl
pip install osim-rl

pip install -r requirements.txt

 