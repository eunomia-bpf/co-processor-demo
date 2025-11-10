#!/bin/bash
# Start QEMU VM with GPU Passthrough
# This script launches the development VM with H100 GPU access

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QEMU_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_DIR="$QEMU_DIR/images"
KERNEL_DIR="$QEMU_DIR/kernel"

VM_NAME="kernel-dev"
VM_MEMORY="16G"
VM_CPUS="8"
DISK_IMAGE="$IMAGE_DIR/${VM_NAME}.qcow2"

# GPU Configuration
GPU_BUS="8a:00.0"

# Check if disk image exists
if [ ! -f "$DISK_IMAGE" ]; then
    echo "ERROR: Disk image not found: $DISK_IMAGE"
    echo "Run ./setup-vm.sh first to create the VM."
    exit 1
fi

# Determine if we should use custom kernel
CUSTOM_KERNEL=""
CUSTOM_INITRD=""
if [ -f "$KERNEL_DIR/bzImage" ]; then
    CUSTOM_KERNEL="-kernel $KERNEL_DIR/bzImage"
    if [ -f "$KERNEL_DIR/initrd.img" ]; then
        CUSTOM_INITRD="-initrd $KERNEL_DIR/initrd.img"
    fi
    CUSTOM_KERNEL="$CUSTOM_KERNEL $CUSTOM_INITRD -append 'root=/dev/sda1 console=ttyS0 console=tty0'"
    echo "Using custom kernel: $KERNEL_DIR/bzImage"
fi

# Check if GPU passthrough is requested
USE_GPU_PASSTHROUGH=false
if [ "$1" == "--gpu" ] || [ "$1" == "-g" ]; then
    USE_GPU_PASSTHROUGH=true
    echo "GPU Passthrough enabled"

    # Verify GPU is bound to vfio-pci
    if ! lspci -k -s $GPU_BUS | grep -q "vfio-pci"; then
        echo "WARNING: GPU is not bound to vfio-pci driver"
        echo "Run ./setup-gpu-passthrough.sh first"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

echo "Starting VM: $VM_NAME"
echo "Memory: $VM_MEMORY"
echo "CPUs: $VM_CPUS"
echo "Disk: $DISK_IMAGE"
echo ""

# Build QEMU command
QEMU_CMD="qemu-system-x86_64 \
    -name $VM_NAME \
    -machine type=q35,accel=kvm \
    -cpu host \
    -smp $VM_CPUS \
    -m $VM_MEMORY \
    -drive file=$DISK_IMAGE,format=qcow2,if=virtio \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device virtio-net-pci,netdev=net0 \
    -display gtk \
    -vga virtio \
    -serial stdio \
    -monitor telnet:127.0.0.1:5555,server,nowait"

# Add custom kernel if available
if [ -n "$CUSTOM_KERNEL" ]; then
    QEMU_CMD="$QEMU_CMD $CUSTOM_KERNEL"
fi

# Add GPU passthrough if requested
if [ "$USE_GPU_PASSTHROUGH" = true ]; then
    QEMU_CMD="$QEMU_CMD \
        -device vfio-pci,host=$GPU_BUS,multifunction=on"
fi

# Enable IOMMU
QEMU_CMD="$QEMU_CMD -device intel-iommu,intremap=on"

echo "Starting QEMU..."
echo "Monitor available on: telnet localhost 5555"
echo "SSH forwarding: localhost:2222 -> VM:22"
echo ""
echo "Press Ctrl+C to stop (in this terminal)"
echo ""

eval $QEMU_CMD
