#!/bin/bash
# Script to enable IOMMU for GPU passthrough

set -e

echo "=== Enable IOMMU for GPU Passthrough ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
fi

# Backup current GRUB config
echo "[1/4] Backing up GRUB configuration..."
cp /etc/default/grub /etc/default/grub.backup.$(date +%Y%m%d-%H%M%S)
echo "  ✓ Backup created"

# Check current settings
echo ""
echo "[2/4] Current kernel parameters:"
cat /proc/cmdline
echo ""

# Modify GRUB config
echo "[3/4] Updating GRUB configuration..."
echo ""

# Remove intel_iommu=off and ensure intel_iommu=on
sed -i 's/intel_iommu=off/intel_iommu=on/g' /etc/default/grub

# If intel_iommu is not present at all, add it
if ! grep -q "intel_iommu=on" /etc/default/grub; then
    sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="intel_iommu=on /g' /etc/default/grub
fi

# Ensure iommu=pt is present
if ! grep -q "iommu=pt" /etc/default/grub; then
    sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="iommu=pt /g' /etc/default/grub
fi

echo "New GRUB_CMDLINE_LINUX_DEFAULT:"
grep "GRUB_CMDLINE_LINUX_DEFAULT" /etc/default/grub

echo ""
echo "[4/4] Updating GRUB..."
update-grub

echo ""
echo "=== IOMMU Configuration Complete ==="
echo ""
echo "⚠️  IMPORTANT: You must REBOOT for changes to take effect!"
echo ""
echo "After reboot:"
echo "  1. Verify IOMMU: dmesg | grep -i iommu"
echo "  2. Check groups: ls /sys/kernel/iommu_groups/"
echo "  3. Start VM with GPU: cd /root/co-processor-demo/qemu-dev/scripts && ./start-with-gpu.sh"
echo ""
echo "Reboot now? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Rebooting..."
    reboot
else
    echo "Reboot cancelled. Remember to reboot manually: sudo reboot"
fi
