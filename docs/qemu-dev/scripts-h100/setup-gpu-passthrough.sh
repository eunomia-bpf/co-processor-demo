#!/bin/bash
# Setup GPU Passthrough for QEMU
# This script configures the H100 GPU for VFIO passthrough

set -e

GPU_BUS="0000:8a:00.0"
GPU_VENDOR="10de"
GPU_DEVICE="2336"

echo "=== GPU Passthrough Setup ==="
echo "GPU: $GPU_BUS (NVIDIA H100)"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Load VFIO modules
echo "Loading VFIO kernel modules..."
modprobe vfio
modprobe vfio_pci
modprobe vfio_iommu_type1

# Check IOMMU groups
echo ""
echo "IOMMU Group for GPU:"
for d in /sys/kernel/iommu_groups/*/devices/*; do
    n=${d#*/iommu_groups/*}
    n=${n%%/*}
    if [[ $(basename $d) == "$GPU_BUS" ]]; then
        echo "  Group $n: $(basename $d) ($(lspci -nns $(basename $d)))"
    fi
done

echo ""
echo "Current GPU driver binding:"
lspci -k -s 8a:00.0

echo ""
read -p "Do you want to unbind the GPU from nvidia and bind to vfio-pci? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Unbinding GPU from current driver..."

    # Unbind from nvidia if bound
    if [ -e /sys/bus/pci/devices/$GPU_BUS/driver ]; then
        echo "$GPU_BUS" > /sys/bus/pci/devices/$GPU_BUS/driver/unbind
    fi

    # Bind to vfio-pci
    echo "Binding to vfio-pci..."
    echo "$GPU_VENDOR $GPU_DEVICE" > /sys/bus/pci/drivers/vfio-pci/new_id

    echo ""
    echo "GPU bound to vfio-pci. New driver binding:"
    lspci -k -s 8a:00.0

    echo ""
    echo "SUCCESS: GPU is now ready for passthrough!"
else
    echo "Skipping driver rebinding."
fi

echo ""
echo "To make this persistent across reboots, add to /etc/modprobe.d/vfio.conf:"
echo "  options vfio-pci ids=$GPU_VENDOR:$GPU_DEVICE"
echo ""
echo "And add to /etc/modules:"
echo "  vfio"
echo "  vfio_iommu_type1"
echo "  vfio_pci"
