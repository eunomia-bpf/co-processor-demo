#!/bin/bash
# QEMU Development Environment Setup Script with GPU Passthrough
# This script sets up a QEMU VM for safe kernel development with H100 GPU access

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QEMU_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_DIR="$QEMU_DIR/images"
KERNEL_DIR="$QEMU_DIR/kernel"

# VM Configuration
VM_NAME="kernel-dev"
VM_MEMORY="16G"
VM_CPUS="8"
DISK_SIZE="50G"
DISK_IMAGE="$IMAGE_DIR/${VM_NAME}.qcow2"

# GPU Information (H100)
GPU_BUS="8a:00.0"
GPU_VENDOR="10de"
GPU_DEVICE="2336"

echo "=== QEMU Development Environment Setup ==="
echo "VM Name: $VM_NAME"
echo "Memory: $VM_MEMORY"
echo "CPUs: $VM_CPUS"
echo "Disk Size: $DISK_SIZE"
echo "GPU: NVIDIA H100 ($GPU_BUS)"
echo ""

# Create necessary directories
mkdir -p "$IMAGE_DIR" "$KERNEL_DIR"

# Check if QEMU is installed
if ! command -v qemu-system-x86_64 &> /dev/null; then
    echo "ERROR: QEMU is not installed."
    echo "Install with: apt-get install qemu-kvm qemu-system-x86 libvirt-daemon-system virtinst"
    exit 1
fi

# Check IOMMU support
if ! dmesg | grep -q "IOMMU"; then
    echo "WARNING: IOMMU may not be properly enabled."
    echo "Current kernel cmdline: $(cat /proc/cmdline)"
    echo "You may need to enable IOMMU in BIOS and kernel parameters."
fi

# Create disk image if it doesn't exist
if [ ! -f "$DISK_IMAGE" ]; then
    echo "Creating VM disk image: $DISK_IMAGE"
    qemu-img create -f qcow2 "$DISK_IMAGE" "$DISK_SIZE"
    echo "Disk image created. You'll need to install an OS before using GPU passthrough."
    echo "Run ./install-os.sh to install Ubuntu/Debian."
else
    echo "Disk image already exists: $DISK_IMAGE"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. If you haven't installed an OS yet, run: ./install-os.sh"
echo "2. To start the VM with GPU passthrough, run: ./start-vm.sh"
echo "3. To build and test a custom kernel, use: ./build-kernel.sh"
echo ""
echo "NOTE: For GPU passthrough to work, you may need to:"
echo "  - Unbind the GPU from the host driver (nvidia)"
echo "  - Bind it to vfio-pci driver"
echo "  - Enable IOMMU in BIOS and kernel cmdline"
echo "  Use ./setup-gpu-passthrough.sh to configure this automatically."
