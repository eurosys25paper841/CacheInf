# CacheInf: Collaborative Edge-Cloud Cache System for Efficient Robotic Visual Model Inference

This is the repository for the reproduction of results in paper 841 submitted to Eurosys 2025 with title, CacheInf: Collaborative Edge-Cloud Cache System for Efficient Robotic Visual Model Inference.

## Installation

```bash
git clone --depth=1 https://github.com/eurosys25paper841/CacheInf.git

# Install CUDA Toolkit (12.1 for wsl  on Ubuntu for example)
wget -c https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

cd CacheInf
export work=$PWD    # We need an environment variable work set as the directory of CacheInf; 
# Add it to .bashrc by 'echo "export work=$PWD" >> ~/.bashrc'

# Install requirements including torch
pip install -r requirements.txt

cd $work/CacheInf/kernels
python3 setup.py install
cd $work
```

## Explanation on the codes
(Underconstruction)

## Evaluation

### Two machine evaluation setup
1. Run the installation procedure on both the robot and the server.

2. On the server
```bash
# Enable running commands of tc and nice without sudo
user=$USER
sudo echo "$USER ALL=(ALL) NOPASSWD: /bin/tc" >> /etc/sudoers
sudo echo "$USER ALL=(ALL) NOPASSWD: /bin/nice" >> /etc/sudoers
```
3. On the robot
```bash
# Get the dataset
mkdir $work/data
cd $work/data
wget -c https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
unzip DAVIS-data.zip
# Please configure the robot so that it can ssh server without prompting for the password, e.g., via ssh-copy-id

# Download kapao checkpoints
cd $work/third_parties/kapao
python3 data/scripts/download_models.py
cd $work
```

### Run the experiments
On the robot
```bash
bash $work/exp_utils/run_cache.sh [IP_SERVER] [PORT_SERVER] [USERNAME_SERVER] [WNIC_NAME_SERVER]
# Run the kapao experiments
```

### Collect the results
```bash
python3 $work/exp_utils/read_logs_all.py
```
You would find the statistics latex files and the plotted figures in ./Plot.

