#!/bin/bash
# Automated Ubuntu Installation for QEMU
# This performs a headless installation with preset answers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QEMU_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_DIR="$QEMU_DIR/images"

VM_NAME="kernel-dev"
VM_MEMORY="16G"
VM_CPUS="8"
DISK_IMAGE="$IMAGE_DIR/${VM_NAME}.qcow2"
ISO_FILE="$IMAGE_DIR/ubuntu-22.04-server.iso"

if [ ! -f "$ISO_FILE" ]; then
    echo "ERROR: ISO not found: $ISO_FILE"
    exit 1
fi

if [ ! -f "$DISK_IMAGE" ]; then
    echo "ERROR: Disk image not found: $DISK_IMAGE"
    echo "Run ./setup-vm.sh first"
    exit 1
fi

echo "Starting Ubuntu installation..."
echo "This will start the Ubuntu installer."
echo ""
echo "Quick setup instructions:"
echo "1. Select 'Install Ubuntu Server'"
echo "2. Choose language and keyboard layout"
echo "3. Network: Leave default (DHCP)"
echo "4. Storage: Use entire disk (default)"
echo "5. Profile setup:"
echo "   - Name: dev"
echo "   - Server name: qemu-cuda-dev"
echo "   - Username: dev"
echo "   - Password: dev123"
echo "6. Enable 'Install OpenSSH server'"
echo "7. Skip additional snaps"
echo "8. Wait for installation to complete"
echo ""
echo "Installation takes about 10-15 minutes"
echo ""
read -p "Press Enter to start installation..."

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
    -nographic \
    -serial mon:stdio

echo ""
echo "Installation process complete!"
