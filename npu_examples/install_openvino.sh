#!/bin/bash

# Enable verbose output to help diagnose issues
set -x

echo "========================================"
echo "OpenVINO Installation Script - Official Method (Updated)"
echo "========================================"

# Check for package manager locks
if pgrep -f "apt|dpkg" > /dev/null; then
    echo "WARNING: Package manager is currently in use by another process."
    echo "Please wait for it to complete or terminate it before continuing."
    echo "You can check active package processes with: ps aux | grep -E 'apt|dpkg'"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation aborted."
        exit 1
    fi
fi

# Step 1: Download the GPG key with --no-proxy to avoid proxy issues
echo "Downloading Intel GPG key..."
wget --no-proxy https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

# Step 2: Add the key to the system keyring (modern approach)
echo "Adding key to system keyring..."
sudo mkdir -p /etc/apt/keyrings
sudo gpg --dearmor < GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo tee /etc/apt/keyrings/intel-openvino.gpg > /dev/null
sudo chmod go+r /etc/apt/keyrings/intel-openvino.gpg

# Step 3: Add the appropriate repository based on Ubuntu version
echo "Adding OpenVINO repository..."
UBUNTU_VERSION=$(lsb_release -rs | cut -d. -f1)
echo "Detected Ubuntu $UBUNTU_VERSION"

if [ "$UBUNTU_VERSION" -ge 25 ]; then
    # Use Ubuntu 24 repository for Ubuntu 25 as it's the closest version
    echo "Using Ubuntu 24 repository for Ubuntu 25"
    echo "deb [signed-by=/etc/apt/keyrings/intel-openvino.gpg] https://apt.repos.intel.com/openvino ubuntu24 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list
elif [ "$UBUNTU_VERSION" -eq 24 ]; then
    echo "deb [signed-by=/etc/apt/keyrings/intel-openvino.gpg] https://apt.repos.intel.com/openvino ubuntu24 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list
elif [ "$UBUNTU_VERSION" -eq 22 ]; then
    echo "deb [signed-by=/etc/apt/keyrings/intel-openvino.gpg] https://apt.repos.intel.com/openvino ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list
elif [ "$UBUNTU_VERSION" -eq 20 ]; then
    echo "deb [signed-by=/etc/apt/keyrings/intel-openvino.gpg] https://apt.repos.intel.com/openvino ubuntu20 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list
else
    echo "Unsupported Ubuntu version: $UBUNTU_VERSION"
    exit 1
fi

# Step 4: Update the list of packages
echo "Updating package lists..."
sudo apt update

# Step 5: Verify that the APT repository is properly set up
echo "Verifying available OpenVINO packages:"
apt-cache search openvino | grep "^openvino"

# Step 6: Install the specific OpenVINO version
echo "Installing OpenVINO 2025.1.0..."
sudo apt install -y openvino-2025.1.0

# Verify installation
echo "Verifying installation..."
apt list --installed | grep openvino

# Clean up the GPG key file
rm -f GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

# Return to the NPU examples directory
cd "$HOME/co-processor-demo/npu_examples"

echo "========================================"
echo "Attempting to compile the NPU example..."
echo "========================================"
make clean || true
make

if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Compilation successful! Running the NPU example..."
    echo "========================================"
    ./npu_matrix_mul
else
    echo "========================================"
    echo "Compilation failed. Try running the sample Python script to verify OpenVINO installation:"
    echo "python3 /usr/share/openvino/samples/python/hello_query_device/hello_query_device.py"
    echo "========================================"
fi 