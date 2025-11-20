#!/bin/bash
# Start Ubuntu 24.04 VM with NVIDIA GPU Passthrough

set -e

# Load configuration
source "$(dirname "$0")/config.sh"

echo "========================================="
echo "Starting VM with GPU Passthrough"
echo "========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Please run as root (use sudo)"
    exit 1
fi

# Create logs directory
mkdir -p "$LOG_DIR"

# Check if GPU is bound to vfio-pci
CURRENT_DRIVER=$(lspci -k -s ${GPU_PCI_ADDRESS} | grep "Kernel driver in use:" | awk '{print $5}')
if [ "$CURRENT_DRIVER" != "vfio-pci" ]; then
    echo "ERROR: GPU is not bound to vfio-pci (current: $CURRENT_DRIVER)"
    echo "Run: sudo ./setup-vfio.sh && sudo reboot"
    exit 1
fi

echo "✓ GPU is bound to vfio-pci"
echo ""

# Check if VM disk exists
if [ ! -f "$VM_DISK_IMAGE" ]; then
    echo "ERROR: VM disk not found: $VM_DISK_IMAGE"
    echo "Run: ./create-vm-disk.sh"
    exit 1
fi

echo "✓ VM disk found: $VM_DISK_IMAGE"
echo ""

# Check if OVMF firmware exists
if [ ! -f "$OVMF_CODE" ] || [ ! -f "$OVMF_VARS" ]; then
    echo "ERROR: OVMF firmware not found"
    echo "Install with: sudo apt install ovmf"
    exit 1
fi

echo "✓ OVMF firmware found"
echo ""

# Determine if this is first boot (ISO install) or regular boot
CDROM_ARGS=""
if [ -f "$UBUNTU_ISO" ]; then
    # Check if ISO should be used (ask user or auto-detect if disk is empty)
    DISK_SIZE=$(qemu-img info "$VM_DISK_IMAGE" | grep "virtual size" | awk '{print $3}')

    # If disk is brand new (50G), assume first install
    if [ "$DISK_SIZE" = "50" ]; then
        echo "First boot detected - will boot from ISO"
        CDROM_ARGS="-cdrom $UBUNTU_ISO -boot d"
    else
        read -p "Boot from ISO for installation? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            CDROM_ARGS="-cdrom $UBUNTU_ISO -boot d"
        fi
    fi
fi

echo "Starting VM with configuration:"
echo "  - Name: $VM_NAME"
echo "  - Memory: $VM_MEMORY"
echo "  - CPUs: $VM_SMP"
echo "  - GPU: $GPU_PCI_ADDRESS (NVIDIA RTX 5090)"
echo "  - GPU Audio: $GPU_AUDIO_PCI_ADDRESS"
echo "  - VNC: localhost:$VNC_PORT"
echo "  - Log: $LOG_FILE"
echo ""

# Create VNC password file
VNC_PASSWORD_FILE="/tmp/qemu-vnc-passwd-$$"
echo "$VNC_PASSWORD" > "$VNC_PASSWORD_FILE"

# Build QEMU command
QEMU_CMD="qemu-system-x86_64 \
  -name $VM_NAME \
  -machine q35,accel=kvm \
  -cpu host,kvm=off,hv_vendor_id=1234567890ab,hv_relaxed,hv_spinlocks=0x1fff,hv_vapic,hv_time \
  -smp $VM_SMP,sockets=1,cores=$VM_CPU_CORES,threads=$VM_CPU_THREADS \
  -m $VM_MEMORY \
  -drive if=pflash,format=raw,readonly=on,file=$OVMF_CODE \
  -drive if=pflash,format=raw,file=$OVMF_VARS \
  -drive file=$VM_DISK_IMAGE,if=virtio,cache=writeback,discard=unmap \
  $CDROM_ARGS \
  -device vfio-pci,host=$GPU_PCI_ADDRESS,multifunction=on,x-vga=on \
  -device vfio-pci,host=$GPU_AUDIO_PCI_ADDRESS \
  -device virtio-net-pci,netdev=net0,mac=$VM_MAC_ADDRESS \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -device virtio-keyboard-pci \
  -device virtio-mouse-pci \
  -device qemu-xhci,id=xhci \
  -vnc :0,password-secret=vncsec \
  -object secret,id=vncsec,data=$VNC_PASSWORD \
  -monitor unix:/tmp/qemu-monitor-$VM_NAME.sock,server,nowait \
  -serial file:$LOG_FILE \
  -daemonize \
  -pidfile /tmp/qemu-$VM_NAME.pid"

echo "Launching QEMU..."
echo "$QEMU_CMD" > "$LOG_DIR/last-qemu-command.txt"

# Execute QEMU
eval $QEMU_CMD

# Clean up VNC password file
rm -f "$VNC_PASSWORD_FILE"

# Wait a moment for VM to start
sleep 2

# Check if VM is running
if [ -f "/tmp/qemu-$VM_NAME.pid" ]; then
    PID=$(cat /tmp/qemu-$VM_NAME.pid)
    if ps -p $PID > /dev/null; then
        echo ""
        echo "========================================="
        echo "VM Started Successfully!"
        echo "========================================="
        echo ""
        echo "VM Information:"
        echo "  - Process ID: $PID"
        echo "  - VNC: localhost:$VNC_PORT (password: $VNC_PASSWORD)"
        echo "  - Monitor: /tmp/qemu-monitor-$VM_NAME.sock"
        echo "  - Log: $LOG_FILE"
        echo ""
        echo "Connect with VNC client:"
        echo "  vncviewer localhost:$VNC_PORT"
        echo ""
        echo "Or use remote desktop:"
        echo "  remmina vnc://localhost:$VNC_PORT"
        echo ""
        echo "To stop VM:"
        echo "  sudo kill $PID"
        echo "  or: echo 'quit' | sudo socat - UNIX-CONNECT:/tmp/qemu-monitor-$VM_NAME.sock"
        echo ""
        echo "After Ubuntu installation, install NVIDIA drivers:"
        echo "  sudo apt update"
        echo "  sudo ubuntu-drivers install"
        echo "  sudo reboot"
        echo ""
    else
        echo "ERROR: VM process died immediately. Check logs:"
        echo "  tail -f $LOG_FILE"
        exit 1
    fi
else
    echo "ERROR: VM failed to start. Check logs:"
    echo "  tail -f $LOG_FILE"
    exit 1
fi
