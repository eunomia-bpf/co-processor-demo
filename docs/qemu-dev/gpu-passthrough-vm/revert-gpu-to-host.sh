#!/bin/bash
# Revert NVIDIA GPU to host use (undo VFIO passthrough)

set -e

# Load configuration
source "$(dirname "$0")/config.sh"

echo "========================================="
echo "Revert NVIDIA GPU to Host Use"
echo "========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Please run as root (use sudo)"
    exit 1
fi

echo "This will remove VFIO configuration and allow host to use NVIDIA GPU"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo "Step 1: Removing VFIO configuration files..."

if [ -f "$VFIO_CONF" ]; then
    rm -f "$VFIO_CONF"
    echo "✓ Removed $VFIO_CONF"
else
    echo "  (already removed)"
fi

if [ -f "$BLACKLIST_CONF" ]; then
    rm -f "$BLACKLIST_CONF"
    echo "✓ Removed $BLACKLIST_CONF"
else
    echo "  (already removed)"
fi
echo ""

echo "Step 2: Removing VFIO modules from initramfs..."

if [ -f "/etc/initramfs-tools/modules" ]; then
    # Remove VFIO entries
    sed -i '/^vfio$/d; /^vfio_iommu_type1$/d; /^vfio_pci$/d' /etc/initramfs-tools/modules
    echo "✓ Removed VFIO modules from initramfs"
fi
echo ""

echo "Step 3: Updating initramfs..."
update-initramfs -u
echo "✓ Initramfs updated"
echo ""

echo "========================================="
echo "Configuration Removed!"
echo "========================================="
echo ""
echo "IMPORTANT: You must REBOOT for changes to take effect"
echo ""
echo "After reboot:"
echo "  - NVIDIA GPU will be available on host"
echo "  - nvidia-smi should work"
echo "  - You can use GPU for CUDA, rendering, etc."
echo ""
echo "To reboot now, run: sudo reboot"
echo ""
