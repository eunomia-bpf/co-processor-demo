#!/bin/bash
# Quick test to verify GPU passthrough readiness

source "$(dirname "$0")/config.sh"

echo "=========================================
GPU Passthrough Pre-Flight Check
========================================="
echo ""

ERRORS=0
WARNINGS=0

# Test 1: IOMMU enabled
echo -n "[ 1/10 ] IOMMU enabled ... "
if [ -d "/sys/kernel/iommu_groups" ]; then
    echo "✓ PASS"
else
    echo "✗ FAIL - Enable VT-d in BIOS"
    ERRORS=$((ERRORS + 1))
fi

# Test 2: KVM loaded
echo -n "[ 2/10 ] KVM module loaded ... "
if lsmod | grep -q kvm_intel; then
    echo "✓ PASS"
else
    echo "✗ FAIL - KVM not loaded"
    ERRORS=$((ERRORS + 1))
fi

# Test 3: QEMU installed
echo -n "[ 3/10 ] QEMU installed ... "
if command -v qemu-system-x86_64 &> /dev/null; then
    echo "✓ PASS"
else
    echo "✗ FAIL - Install qemu-system-x86"
    ERRORS=$((ERRORS + 1))
fi

# Test 4: OVMF firmware
echo -n "[ 4/10 ] OVMF firmware exists ... "
if [ -f "$OVMF_CODE" ]; then
    echo "✓ PASS"
else
    echo "✗ FAIL - Install ovmf package"
    ERRORS=$((ERRORS + 1))
fi

# Test 5: GPU exists
echo -n "[ 5/10 ] NVIDIA GPU detected ... "
if lspci -s ${GPU_PCI_ADDRESS} | grep -q NVIDIA; then
    echo "✓ PASS"
else
    echo "✗ FAIL - GPU not found at ${GPU_PCI_ADDRESS}"
    ERRORS=$((ERRORS + 1))
fi

# Test 6: GPU driver binding
echo -n "[ 6/10 ] GPU bound to vfio-pci ... "
CURRENT_DRIVER=$(lspci -k -s ${GPU_PCI_ADDRESS} 2>/dev/null | grep "Kernel driver in use:" | awk '{print $5}')
if [ "$CURRENT_DRIVER" = "vfio-pci" ]; then
    echo "✓ PASS"
elif [ "$CURRENT_DRIVER" = "nvidia" ]; then
    echo "⚠ WARN - GPU using nvidia driver (run setup-vfio.sh)"
    WARNINGS=$((WARNINGS + 1))
else
    echo "⚠ WARN - GPU driver: $CURRENT_DRIVER"
    WARNINGS=$((WARNINGS + 1))
fi

# Test 7: VFIO modules
echo -n "[ 7/10 ] VFIO modules available ... "
if modinfo vfio-pci &> /dev/null; then
    echo "✓ PASS"
else
    echo "✗ FAIL - VFIO modules not available"
    ERRORS=$((ERRORS + 1))
fi

# Test 8: IOMMU group clean
echo -n "[ 8/10 ] GPU in clean IOMMU group ... "
if [ -e "/sys/bus/pci/devices/${GPU_PCI_ADDRESS}/iommu_group" ]; then
    IOMMU_GROUP=$(basename $(dirname $(readlink /sys/bus/pci/devices/${GPU_PCI_ADDRESS}/iommu_group 2>/dev/null) 2>/dev/null) 2>/dev/null)
    if [ -n "$IOMMU_GROUP" ]; then
        DEVICE_COUNT=$(ls -1 /sys/kernel/iommu_groups/${IOMMU_GROUP}/devices/ 2>/dev/null | wc -l)
        if [ "$DEVICE_COUNT" -le 2 ]; then
            echo "✓ PASS (group $IOMMU_GROUP, $DEVICE_COUNT devices)"
        else
            echo "⚠ WARN (group $IOMMU_GROUP has $DEVICE_COUNT devices)"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo "⚠ WARN - Cannot determine IOMMU group"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "✗ FAIL - Cannot access IOMMU group"
    ERRORS=$((ERRORS + 1))
fi

# Test 9: Intel GPU for host
echo -n "[ 9/10 ] Intel GPU for host display ... "
INTEL_DRIVER=$(lspci -k -s 00:02.0 2>/dev/null | grep "Kernel driver in use:" | awk '{print $5}')
if [ "$INTEL_DRIVER" = "i915" ]; then
    echo "✓ PASS"
else
    echo "⚠ WARN - Intel GPU driver: $INTEL_DRIVER"
    WARNINGS=$((WARNINGS + 1))
fi

# Test 10: User in right groups
echo -n "[ 10/10 ] User in libvirt/kvm group ... "
if groups | grep -q libvirt || groups | grep -q kvm; then
    echo "✓ PASS"
else
    echo "⚠ WARN - Add user to libvirt group for non-root access"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""
echo "========================================="
echo "Test Results:"
echo "========================================="
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✓✓✓ ALL TESTS PASSED ✓✓✓"
    echo ""
    echo "System is READY for GPU passthrough!"
    echo ""
    if [ "$CURRENT_DRIVER" = "vfio-pci" ]; then
        echo "Next step: Create VM disk"
        echo "  ./create-vm-disk.sh"
    else
        echo "Next step: Configure VFIO"
        echo "  sudo ./setup-vfio.sh && sudo reboot"
    fi
elif [ $ERRORS -eq 0 ]; then
    echo "✓ PASSED with $WARNINGS warning(s)"
    echo ""
    echo "System is ready but has minor issues."
    echo "Review warnings above."
elif [ $ERRORS -le 2 ]; then
    echo "⚠ FAILED with $ERRORS error(s), $WARNINGS warning(s)"
    echo ""
    echo "Fix errors before proceeding."
    echo "See TROUBLESHOOTING.md for help."
else
    echo "✗ FAILED with $ERRORS error(s), $WARNINGS warning(s)"
    echo ""
    echo "System not ready for GPU passthrough."
    echo "Review errors above and consult documentation."
fi

echo ""
exit $ERRORS
