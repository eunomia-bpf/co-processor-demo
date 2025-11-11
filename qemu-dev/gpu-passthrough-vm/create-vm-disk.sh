#!/bin/bash
# Create Ubuntu 24.04 VM disk and download ISO

set -e

# Load configuration
source "$(dirname "$0")/config.sh"

echo "========================================="
echo "Create Ubuntu 24.04 VM Disk"
echo "========================================="
echo ""

# Create logs directory
mkdir -p "$LOG_DIR"

echo "Step 1: Checking for existing VM disk..."
if [ -f "$VM_DISK_IMAGE" ]; then
    echo "WARNING: VM disk already exists: $VM_DISK_IMAGE"
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$VM_DISK_IMAGE"
        echo "✓ Deleted existing disk"
    else
        echo "Keeping existing disk. Exiting."
        exit 0
    fi
fi
echo ""

echo "Step 2: Creating VM disk image (${VM_DISK_SIZE})..."
qemu-img create -f qcow2 "$VM_DISK_IMAGE" "$VM_DISK_SIZE"
echo "✓ Created $VM_DISK_IMAGE"
echo ""

echo "Step 3: Checking for Ubuntu ISO..."
if [ -f "$UBUNTU_ISO" ]; then
    echo "✓ Ubuntu ISO already downloaded: $UBUNTU_ISO"
else
    echo "Downloading Ubuntu 24.04 ISO..."
    echo "URL: $UBUNTU_ISO_URL"
    echo "This may take a while (approximately 6GB download)..."

    if command -v wget &> /dev/null; then
        wget -O "$UBUNTU_ISO" "$UBUNTU_ISO_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$UBUNTU_ISO" "$UBUNTU_ISO_URL"
    else
        echo "ERROR: Neither wget nor curl found. Please install one of them."
        echo "Or manually download Ubuntu 24.04 ISO to: $UBUNTU_ISO"
        exit 1
    fi

    echo "✓ Downloaded Ubuntu ISO"
fi
echo ""

echo "Step 4: Creating OVMF variables file..."
if [ -f "$OVMF_CODE" ]; then
    if [ -f "$OVMF_VARS" ]; then
        echo "✓ OVMF variables file already exists"
    else
        cp /usr/share/OVMF/OVMF_VARS_4M.fd "$OVMF_VARS"
        echo "✓ Created OVMF variables file"
    fi
else
    echo "ERROR: OVMF firmware not found at $OVMF_CODE"
    echo "Install with: sudo apt install ovmf"
    exit 1
fi
echo ""

echo "========================================="
echo "VM Disk Setup Complete!"
echo "========================================="
echo ""
echo "VM Configuration:"
echo "  - Disk: $VM_DISK_IMAGE (${VM_DISK_SIZE})"
echo "  - ISO: $UBUNTU_ISO"
echo "  - Memory: $VM_MEMORY"
echo "  - CPUs: $VM_SMP ($VM_CPU_CORES cores x $VM_CPU_THREADS threads)"
echo ""
echo "Next steps:"
echo "  1. Ensure GPU is configured for VFIO: sudo ./setup-vfio.sh"
echo "  2. Reboot if you ran setup-vfio.sh"
echo "  3. Start VM: sudo ./start-vm.sh"
echo ""
