#!/bin/bash
# Build Custom Kernel for QEMU VM
# This script helps compile and prepare a custom kernel for testing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QEMU_DIR="$(dirname "$SCRIPT_DIR")"
KERNEL_DIR="$QEMU_DIR/kernel"

echo "=== Custom Kernel Build Script ==="
echo ""

# Check if kernel source is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-kernel-source>"
    echo ""
    echo "Example:"
    echo "  1. Download kernel source:"
    echo "     cd $KERNEL_DIR"
    echo "     wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.8.tar.xz"
    echo "     tar xf linux-6.8.tar.xz"
    echo ""
    echo "  2. Build kernel:"
    echo "     $0 $KERNEL_DIR/linux-6.8"
    echo ""
    echo "Or to use the host kernel config:"
    echo "  $0 <kernel-source> --use-host-config"
    exit 1
fi

KERNEL_SRC="$1"
USE_HOST_CONFIG=false

if [ "$2" == "--use-host-config" ]; then
    USE_HOST_CONFIG=true
fi

if [ ! -d "$KERNEL_SRC" ]; then
    echo "ERROR: Kernel source directory not found: $KERNEL_SRC"
    exit 1
fi

echo "Kernel source: $KERNEL_SRC"
echo "Output directory: $KERNEL_DIR"
echo ""

cd "$KERNEL_SRC"

# Configure kernel
if [ ! -f .config ]; then
    if [ "$USE_HOST_CONFIG" = true ] && [ -f "/boot/config-$(uname -r)" ]; then
        echo "Using host kernel config..."
        cp "/boot/config-$(uname -r)" .config
        make olddefconfig
    else
        echo "Creating default config..."
        make defconfig
    fi

    # Enable important options for QEMU and GPU
    echo "Enabling QEMU and GPU support options..."
    scripts/config --enable CONFIG_VIRTIO_PCI
    scripts/config --enable CONFIG_VIRTIO_BLK
    scripts/config --enable CONFIG_VIRTIO_NET
    scripts/config --enable CONFIG_VIRTIO_CONSOLE
    scripts/config --enable CONFIG_DRM
    scripts/config --enable CONFIG_DRM_NOUVEAU
    scripts/config --enable CONFIG_VFIO
    scripts/config --enable CONFIG_VFIO_PCI
    scripts/config --enable CONFIG_VFIO_PCI_VGA
    scripts/config --enable CONFIG_IOMMU_SUPPORT
    scripts/config --enable CONFIG_INTEL_IOMMU
    scripts/config --enable CONFIG_INTEL_IOMMU_SVM

    make olddefconfig
fi

# Ask for configuration changes
read -p "Do you want to modify kernel config with menuconfig? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    make menuconfig
fi

# Build kernel
echo ""
echo "Building kernel (this may take a while)..."
NCPUS=$(nproc)
make -j$NCPUS

if [ $? -ne 0 ]; then
    echo "ERROR: Kernel build failed"
    exit 1
fi

# Copy kernel to QEMU directory
echo ""
echo "Copying kernel to $KERNEL_DIR..."
mkdir -p "$KERNEL_DIR"
cp arch/x86/boot/bzImage "$KERNEL_DIR/"

# Create initrd if tools are available
if command -v mkinitramfs &> /dev/null; then
    echo "Creating initrd..."
    TEMP_DIR=$(mktemp -d)
    make modules_install INSTALL_MOD_PATH="$TEMP_DIR"
    mkinitramfs -o "$KERNEL_DIR/initrd.img" "$(make kernelrelease)" -d "$TEMP_DIR"
    rm -rf "$TEMP_DIR"
    echo "initrd created: $KERNEL_DIR/initrd.img"
else
    echo "WARNING: mkinitramfs not found, skipping initrd creation"
    echo "The VM may need initrd for proper boot"
fi

echo ""
echo "SUCCESS: Kernel built and ready!"
echo "  Kernel: $KERNEL_DIR/bzImage"
if [ -f "$KERNEL_DIR/initrd.img" ]; then
    echo "  initrd: $KERNEL_DIR/initrd.img"
fi
echo ""
echo "Start the VM with your custom kernel using: ./start-vm.sh"
