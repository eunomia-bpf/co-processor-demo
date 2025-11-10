#!/bin/bash
# Restore GPU to host for normal use

echo "=== Restore H100 GPU to Host ==="
echo ""

# Stop any running VMs
echo "[1/3] Stopping any running QEMU VMs..."
pkill -9 -f "qemu-system.*kernel-dev" 2>/dev/null || true
sleep 2

# Unbind from VFIO if bound
echo "[2/3] Unbinding GPU from VFIO (if bound)..."
if [ -e "/sys/bus/pci/devices/0000:8a:00.0/driver" ]; then
    CURRENT_DRIVER=$(basename $(readlink /sys/bus/pci/devices/0000:8a:00.0/driver))
    echo "  Current driver: $CURRENT_DRIVER"

    if [ "$CURRENT_DRIVER" = "vfio-pci" ]; then
        echo "0000:8a:00.0" > /sys/bus/pci/devices/0000:8a:00.0/driver/unbind 2>/dev/null || true
        echo "  ✓ Unbound from vfio-pci"
    fi
else
    echo "  GPU not currently bound to any driver"
fi

# The cleanest way to restore GPU is to reboot
echo ""
echo "[3/3] GPU restoration options:"
echo ""
echo "Option 1 - REBOOT (Recommended - cleanest)"
echo "  This will fully reset the GPU state and bind to nvidia driver"
echo "  Command: sudo reboot"
echo ""
echo "Option 2 - Try manual rebind (may not work if GPU is in bad state)"
echo "  modprobe nvidia && echo '0000:8a:00.0' > /sys/bus/pci/drivers/nvidia/bind"
echo ""
echo "Current GPU status:"
lspci -k -s 8a:00.0
echo ""
echo "Test nvidia-smi:"
nvidia-smi 2>&1 | head -5

echo ""
echo "=== Current Status ==="
if nvidia-smi &>/dev/null; then
    echo "✅ GPU is accessible on host - nvidia-smi works!"
else
    echo "⚠️  GPU needs reboot to fully restore"
    echo ""
    echo "Recommended: sudo reboot"
    echo ""
    echo "After reboot:"
    echo "  - GPU will work normally on host"
    echo "  - nvidia-smi will work"
    echo "  - CUDA programs will work"
    echo "  - VM scripts remain ready for future GPU passthrough"
fi
