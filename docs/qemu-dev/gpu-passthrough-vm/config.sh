#!/bin/bash
# Configuration for GPU Passthrough VM

# GPU PCI IDs (NVIDIA RTX 5090)
GPU_PCI_ADDRESS="0000:02:00.0"
GPU_AUDIO_PCI_ADDRESS="0000:02:00.1"
GPU_VENDOR_ID="10de"
GPU_DEVICE_ID="2b85"
GPU_AUDIO_DEVICE_ID="22e8"

# VM Configuration
VM_NAME="ubuntu2404-gpu"
VM_MEMORY="16G"
VM_CPU_CORES="8"
VM_CPU_THREADS="2"
VM_SMP=$((VM_CPU_CORES * VM_CPU_THREADS))

# Disk Configuration
VM_DISK_SIZE="50G"
VM_DISK_IMAGE="./ubuntu2404-gpu.qcow2"

# OVMF/UEFI Firmware Paths
OVMF_CODE="/usr/share/OVMF/OVMF_CODE_4M.fd"
OVMF_VARS="./OVMF_VARS_4M.fd"

# Ubuntu ISO (will be downloaded if not present)
UBUNTU_ISO_URL="https://releases.ubuntu.com/24.04/ubuntu-24.04.1-desktop-amd64.iso"
UBUNTU_ISO="./ubuntu-24.04.1-desktop-amd64.iso"

# VNC Configuration
VNC_PORT="5900"
VNC_PASSWORD="gpu-vm-pass"

# Network Configuration (bridged network)
# Change to your network interface if different
HOST_NETWORK_INTERFACE="enp129s0"
VM_MAC_ADDRESS="52:54:00:12:34:56"

# VFIO Configuration Files
VFIO_CONF="/etc/modprobe.d/vfio.conf"
BLACKLIST_CONF="/etc/modprobe.d/blacklist-nvidia.conf"

# Logging
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/vm-$(date +%Y%m%d-%H%M%S).log"
