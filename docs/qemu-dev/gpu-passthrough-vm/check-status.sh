#!/bin/bash
# Check GPU and VFIO status

# Load configuration
source "$(dirname "$0")/config.sh"

echo "========================================="
echo "GPU and VFIO Status Check"
echo "========================================="
echo ""

echo "1. IOMMU Status:"
if [ -d "/sys/kernel/iommu_groups" ]; then
    IOMMU_COUNT=$(find /sys/kernel/iommu_groups/ -maxdepth 1 -type d | wc -l)
    echo "   ✓ IOMMU is enabled ($((IOMMU_COUNT - 1)) groups found)"
else
    echo "   ✗ IOMMU is NOT enabled"
fi
echo ""

echo "2. Kernel Parameters:"
echo "   $(cat /proc/cmdline)"
echo ""

echo "3. NVIDIA GPU Status (${GPU_PCI_ADDRESS}):"
echo "   Device Info:"
lspci -v -s ${GPU_PCI_ADDRESS} | head -10 | sed 's/^/   /'
echo ""

echo "4. GPU Driver Binding:"
CURRENT_DRIVER=$(lspci -k -s ${GPU_PCI_ADDRESS} | grep "Kernel driver in use:" | awk '{print $5}')
if [ "$CURRENT_DRIVER" = "vfio-pci" ]; then
    echo "   ✓ GPU is bound to vfio-pci (READY FOR PASSTHROUGH)"
elif [ "$CURRENT_DRIVER" = "nvidia" ]; then
    echo "   ⚠ GPU is bound to nvidia driver (HOST USE)"
    echo "   Run sudo ./setup-vfio.sh and reboot to enable passthrough"
else
    echo "   ⚠ GPU is bound to: $CURRENT_DRIVER"
fi
echo ""

echo "5. GPU Audio Status (${GPU_AUDIO_PCI_ADDRESS}):"
AUDIO_DRIVER=$(lspci -k -s ${GPU_AUDIO_PCI_ADDRESS} | grep "Kernel driver in use:" | awk '{print $5}')
if [ "$AUDIO_DRIVER" = "vfio-pci" ]; then
    echo "   ✓ GPU Audio is bound to vfio-pci"
elif [ -n "$AUDIO_DRIVER" ]; then
    echo "   ⚠ GPU Audio is bound to: $AUDIO_DRIVER"
else
    echo "   ⚠ GPU Audio has no driver"
fi
echo ""

echo "6. VFIO Configuration Files:"
if [ -f "$VFIO_CONF" ]; then
    echo "   ✓ VFIO config exists: $VFIO_CONF"
    cat "$VFIO_CONF" | sed 's/^/     /'
else
    echo "   ✗ VFIO config NOT found: $VFIO_CONF"
fi
echo ""

if [ -f "$BLACKLIST_CONF" ]; then
    echo "   ✓ Blacklist config exists: $BLACKLIST_CONF"
else
    echo "   ✗ Blacklist config NOT found: $BLACKLIST_CONF"
fi
echo ""

echo "7. VFIO Modules:"
if lsmod | grep -q vfio_pci; then
    echo "   ✓ vfio_pci module is loaded"
    lsmod | grep vfio | sed 's/^/     /'
else
    echo "   ✗ vfio_pci module is NOT loaded"
fi
echo ""

echo "8. IOMMU Group for GPU:"
if [ -e "/sys/bus/pci/devices/${GPU_PCI_ADDRESS}/iommu_group" ]; then
    IOMMU_GROUP=$(basename $(dirname $(readlink /sys/bus/pci/devices/${GPU_PCI_ADDRESS}/iommu_group)))
    echo "   GPU is in IOMMU group: $IOMMU_GROUP"
    echo "   Devices in this group:"
    for dev in /sys/kernel/iommu_groups/${IOMMU_GROUP}/devices/*; do
        echo "     - $(basename $dev): $(lspci -s $(basename $dev))"
    done
else
    echo "   ✗ Cannot determine IOMMU group"
fi
echo ""

echo "9. Intel GPU (Host Display):"
INTEL_DRIVER=$(lspci -k -s 00:02.0 | grep "Kernel driver in use:" | awk '{print $5}')
echo "   Driver: $INTEL_DRIVER"
if [ "$INTEL_DRIVER" = "i915" ]; then
    echo "   ✓ Intel GPU is running host display"
fi
echo ""

echo "10. QEMU/KVM Status:"
if command -v qemu-system-x86_64 &> /dev/null; then
    QEMU_VERSION=$(qemu-system-x86_64 --version | head -1)
    echo "   ✓ $QEMU_VERSION"
else
    echo "   ✗ QEMU not found"
fi

if lsmod | grep -q kvm_intel; then
    echo "   ✓ KVM Intel module loaded"
else
    echo "   ✗ KVM Intel module NOT loaded"
fi
echo ""

echo "========================================="
echo "Summary:"
echo "========================================="

if [ "$CURRENT_DRIVER" = "vfio-pci" ] && [ "$AUDIO_DRIVER" = "vfio-pci" ]; then
    echo "✓ GPU is READY for passthrough!"
    echo "  Run: sudo ./start-vm.sh"
elif [ "$CURRENT_DRIVER" = "nvidia" ]; then
    echo "⚠ GPU is currently used by host"
    echo "  To enable passthrough: sudo ./setup-vfio.sh && sudo reboot"
else
    echo "⚠ GPU configuration incomplete"
    echo "  Run: sudo ./setup-vfio.sh && sudo reboot"
fi
echo ""
