#!/bin/bash
# Setup NVIDIA GPU for VFIO Passthrough

set -e

# Load configuration
source "$(dirname "$0")/config.sh"

echo "========================================="
echo "NVIDIA GPU VFIO Passthrough Setup"
echo "========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Please run as root (use sudo)"
    exit 1
fi

echo "Step 1: Verifying IOMMU is enabled..."
if ! grep -q "iommu=pt" /proc/cmdline; then
    echo "WARNING: IOMMU might not be properly enabled in kernel parameters"
    echo "Current kernel parameters: $(cat /proc/cmdline)"
fi

if [ ! -d "/sys/kernel/iommu_groups" ]; then
    echo "ERROR: IOMMU groups not found. Please enable VT-d in BIOS and add intel_iommu=on to kernel parameters"
    exit 1
fi

echo "✓ IOMMU is enabled"
echo ""

echo "Step 2: Checking GPU IOMMU group..."
IOMMU_GROUP=$(basename $(dirname $(readlink /sys/bus/pci/devices/${GPU_PCI_ADDRESS}/iommu_group)))
echo "GPU is in IOMMU group: $IOMMU_GROUP"

echo "Devices in this IOMMU group:"
for dev in /sys/kernel/iommu_groups/${IOMMU_GROUP}/devices/*; do
    echo "  - $(basename $dev): $(lspci -s $(basename $dev))"
done
echo ""

echo "Step 3: Creating VFIO configuration..."

# Create VFIO config to bind GPU at boot
cat > "$VFIO_CONF" <<EOF
# Bind NVIDIA RTX 5090 to VFIO-PCI driver
options vfio-pci ids=${GPU_VENDOR_ID}:${GPU_DEVICE_ID},${GPU_VENDOR_ID}:${GPU_AUDIO_DEVICE_ID}

# Enable unsafe interrupts if needed (usually not required)
# options vfio_iommu_type1 allow_unsafe_interrupts=1
EOF

echo "✓ Created $VFIO_CONF"
cat "$VFIO_CONF"
echo ""

echo "Step 4: Blacklisting NVIDIA drivers..."

# Blacklist NVIDIA drivers to prevent them from binding to GPU
cat > "$BLACKLIST_CONF" <<EOF
# Prevent NVIDIA drivers from binding to GPU (for VFIO passthrough)
blacklist nvidia
blacklist nvidia_drm
blacklist nvidia_modeset
blacklist nvidia_uvm
blacklist nouveau
EOF

echo "✓ Created $BLACKLIST_CONF"
cat "$BLACKLIST_CONF"
echo ""

echo "Step 5: Ensuring VFIO modules load early..."

# Add VFIO modules to initramfs
if ! grep -q "vfio" /etc/initramfs-tools/modules 2>/dev/null; then
    cat >> /etc/initramfs-tools/modules <<EOF
# VFIO modules for GPU passthrough
vfio
vfio_iommu_type1
vfio_pci
EOF
    echo "✓ Added VFIO modules to initramfs"
else
    echo "✓ VFIO modules already in initramfs"
fi
echo ""

echo "Step 6: Updating initramfs..."
update-initramfs -u
echo "✓ Initramfs updated"
echo ""

echo "Step 7: Current GPU driver status:"
lspci -k -s ${GPU_PCI_ADDRESS} | grep -A 3 "VGA"
echo ""

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "IMPORTANT: You must REBOOT for changes to take effect"
echo ""
echo "After reboot, run ./check-status.sh to verify VFIO binding"
echo ""
echo "To reboot now, run: sudo reboot"
echo ""
