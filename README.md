# CacheInf: Collaborative Edge-Cloud Cache System for Efficient Robotic Visual Model Inference

This is the repository for the reproduction of results in paper 841 submitted to Eurosys 2025 with title, CacheInf: Collaborative Edge-Cloud Cache System for Efficient Robotic Visual Model Inference.

## Installation

```bash
git clone --depth=1 https://github.com/eurosys25paper841/CacheInf.git

# Install CUDA Toolkit (12.2 for wsl  on Ubuntu for example)
wget -c https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
sudo sh cuda_12.2.2_535.104.05_linux.run

# Install requirements including torch
pip install -r requirements.txt
```

## Evaluation

### Two machine evaluation setup
Run the installation procedure on both the robot and the server.

On the server
```bash
# Enable running commands of tc and nice without sudo
user=$USER
sudo echo "$USER ALL=(ALL) NOPASSWD: /bin/tc" >> /etc/sudoers
sudo echo "$USER ALL=(ALL) NOPASSWD: /bin/nice" >> /etc/sudoers
# Configure tc control
```
### Run the experiments
On the robot
```bash
bash run_all.sh [IP_SERVER] [PORT_SERVER] [WNIC_NAME_SERVER]
# Run all the experiments
```

### Collect the results
```bash
python3 exp_utils/read_logs_all.py
```
You would find the statistics latex files and the plotted figures in ./Plot.

