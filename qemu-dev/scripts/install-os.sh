#!/bin/bash
# Install OS in QEMU VM
# This script helps install Ubuntu/Debian in the development VM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QEMU_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_DIR="$QEMU_DIR/images"

VM_NAME="kernel-dev"
VM_MEMORY="16G"
VM_CPUS="8"
DISK_IMAGE="$IMAGE_DIR/${VM_NAME}.qcow2"

echo "=== OS Installation for QEMU VM ==="
echo ""

# Check if disk image exists
if [ ! -f "$DISK_IMAGE" ]; then
    echo "ERROR: Disk image not found: $DISK_IMAGE"
    echo "Run ./setup-vm.sh first to create the VM."
    exit 1
fi

# Check for ISO file
ISO_FILE=""
if [ -n "$1" ]; then
    ISO_FILE="$1"
elif [ -f "$IMAGE_DIR/ubuntu.iso" ]; then
    ISO_FILE="$IMAGE_DIR/ubuntu.iso"
elif [ -f "$IMAGE_DIR/debian.iso" ]; then
    ISO_FILE="$IMAGE_DIR/debian.iso"
fi

if [ -z "$ISO_FILE" ] || [ ! -f "$ISO_FILE" ]; then
    echo "Usage: $0 <path-to-iso>"
    echo ""
    echo "Download an ISO first:"
    echo "  Ubuntu Server: https://ubuntu.com/download/server"
    echo "  Debian: https://www.debian.org/distrib/netinst"
    echo ""
    echo "Then run: $0 /path/to/ubuntu-server.iso"
    exit 1
fi

echo "Installing OS from: $ISO_FILE"
echo "Target disk: $DISK_IMAGE"
echo ""
echo "Starting installation VM..."
echo "Follow the on-screen installer instructions."
echo ""

qemu-system-x86_64 \
    -name ${VM_NAME}-install \
    -machine type=q35,accel=kvm \
    -cpu host \
    -smp $VM_CPUS \
    -m $VM_MEMORY \
    -drive file=$DISK_IMAGE,format=qcow2,if=virtio \
    -cdrom "$ISO_FILE" \
    -boot d \
    -netdev user,id=net0 \
    -device virtio-net-pci,netdev=net0 \
    -display gtk \
    -vga virtio

echo ""
echo "Installation complete! You can now start the VM with ./start-vm.sh"
