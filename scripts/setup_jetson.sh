#!/bin/bash
# Setup script for Jetson Nano Developer Kit P3450 (JetPack 4.6.4)
# Uses the system Python 3.6 with JetPack's built-in TensorRT & CUDA.
# No PyTorch or Ultralytics needed at runtime.

set -e

echo "====================================================="
echo " Setting up Inventory Verification for Jetson Nano   "
echo " (JetPack 4.6.4, L4T 32.7.4, CUDA 10.2, TensorRT 8.2)"
echo "====================================================="

# 1. System dependencies
echo -e "\n--- Installing system dependencies ---"
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev libopencv-dev

# 2. Install pycuda (needed for TensorRT Python bindings)
echo -e "\n--- Installing pycuda ---"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip3 install pycuda

# 3. Install other Python packages
echo -e "\n--- Installing Python packages ---"
pip3 install -r requirements-jetson.txt

# 4. Setup udev rules for OAK-D camera
echo -e "\n--- Configuring OAK-D camera ---"
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

echo -e "\n====================================================="
echo " Setup Complete!"
echo ""
echo " Next steps:"
echo "   1. Copy best.onnx to models/checkpoints/inventory-yolo11s/weights/"
echo "   2. Build the TensorRT engine:"
echo "      /usr/src/tensorrt/bin/trtexec \\"
echo "        --onnx=models/checkpoints/inventory-yolo11s/weights/best.onnx \\"
echo "        --saveEngine=models/checkpoints/inventory-yolo11s/weights/best.engine \\"
echo "        --fp16"
echo "   3. Run the pipeline:"
echo "      python3 -m src.pipeline.run_jetson --config configs/jetson.yaml"
echo "====================================================="
