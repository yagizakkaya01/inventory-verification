#!/bin/bash
# Setup script for Jetson Nano Developer Kit P3450 (JetPack 4.6.4)
# JetPack 4.6.4 comes with Ubuntu 18.04 and Python 3.6 by default.
# Ultralytics YOLO11 requires Python >= 3.8.
# This script installs Python 3.8 and a community-built PyTorch wheel for JetPack 4.6.

set -e

echo "====================================================="
echo " Setting up Inventory Verification for Jetson Nano   "
echo " (JetPack 4.6.4, L4T 32.7.4, CUDA 10.2)              "
echo "====================================================="

# 1. Install Python 3.8 and dependencies
echo -e "\n--- Installing Python 3.8 ---"
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-dev python3.8-venv python3.8-distutils libpython3.8-dev
curl https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py
sudo python3.8 get-pip.py
rm get-pip.py

# 2. Setup Virtual Environment
echo -e "\n--- Setting up Virtual Environment ---"
python3.8 -m venv .venv-jetson
source .venv-jetson/bin/activate
pip install --upgrade pip

# 3. Install PyTorch & Torchvision for JetPack 4.6 (Python 3.8)
# Since NVIDIA only provides Python 3.6 wheels for JP4.6, we use a pre-compiled Python 3.8 wheel
echo -e "\n--- Installing PyTorch & Torchvision ---"
sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev

# PyTorch 1.10.0 for Python 3.8 (built for JetPack 4.6 by community)
pip install wget
python -c "import wget; wget.download('https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl', 'torch-1.10.0-cp38-cp38-linux_aarch64.whl')"
pip install torch-1.10.0-cp38-cp38-linux_aarch64.whl
rm torch-1.10.0-cp38-cp38-linux_aarch64.whl

# Torchvision 0.11.1
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python setup.py install --user
cd ..
rm -rf torchvision

# 4. Install other requirements
echo -e "\n--- Installing project requirements ---"
# DepthAI needs some rules on Linux
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Install packages
pip install ultralytics opencv-python numpy pyyaml pydantic depthai blobconverter

echo -e "\n====================================================="
echo " Setup Complete! "
echo " Remember to activate the environment before running:"
echo "   source .venv-jetson/bin/activate"
echo "====================================================="
