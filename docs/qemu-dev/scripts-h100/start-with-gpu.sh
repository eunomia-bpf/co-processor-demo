#!/bin/bash
# Complete script to start VM with H100 GPU passthrough and test CUDA

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QEMU_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_DIR="$QEMU_DIR/images"

cd "$IMAGE_DIR"

echo "=== QEMU VM with H100 GPU Passthrough ==="
echo ""

# GPU details
GPU_BUS="8a:00.0"
GPU_VENDOR="10de"
GPU_DEVICE="2336"

# Check if VM is already running
if pgrep -f "qemu-system-x86_64.*kernel-dev" > /dev/null; then
    echo "VM is already running. Stopping it..."
    pkill -9 -f "qemu-system-x86_64.*kernel-dev" || true
    sleep 3
fi

# Step 1: Unbind GPU from nvidia driver
echo "[1/5] Unbinding H100 GPU from NVIDIA driver..."
if [ -e "/sys/bus/pci/devices/0000:$GPU_BUS/driver" ]; then
    echo "0000:$GPU_BUS" | tee /sys/bus/pci/devices/0000:$GPU_BUS/driver/unbind 2>/dev/null || echo "  (already unbound)"
fi
sleep 1

# Step 2: Load VFIO modules
echo "[2/5] Loading VFIO kernel modules..."
modprobe vfio 2>/dev/null || true
modprobe vfio_pci 2>/dev/null || true
modprobe vfio_iommu_type1 2>/dev/null || true
sleep 1

# Step 3: Bind GPU to vfio-pci
echo "[3/5] Binding H100 GPU to vfio-pci..."
if ! lspci -k -s $GPU_BUS | grep -q "vfio-pci"; then
    echo "$GPU_VENDOR $GPU_DEVICE" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null || true
    sleep 2
fi

# Verify binding
if lspci -k -s $GPU_BUS | grep -q "vfio-pci"; then
    echo "  ✓ GPU successfully bound to vfio-pci"
else
    echo "  ⚠ GPU binding to vfio-pci may have failed, but continuing..."
fi

# Step 4: Start VM with GPU
echo "[4/5] Starting QEMU VM with GPU passthrough..."
echo "  Memory: 16GB"
echo "  CPUs: 8"
echo "  GPU: H100 at $GPU_BUS"
echo ""
echo "VM will start in background. To access:"
echo "  SSH: ssh -p 2222 ubuntu@localhost (password: ubuntu)"
echo "  Monitor console output: tail -f /tmp/qemu-gpu.log"
echo ""

qemu-system-x86_64 \
    -name kernel-dev-gpu \
    -machine type=q35,accel=kvm \
    -cpu host \
    -smp 8 \
    -m 16G \
    -drive file=kernel-dev.qcow2,format=qcow2,if=virtio \
    -drive file=cloud-init.iso,format=raw,if=virtio \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device virtio-net-pci,netdev=net0 \
    -device vfio-pci,host=$GPU_BUS \
    -nographic \
    -serial mon:stdio > /tmp/qemu-gpu.log 2>&1 &

VM_PID=$!
echo "VM started with PID: $VM_PID"
echo ""

# Step 5: Wait for boot and test
echo "[5/5] Waiting for VM to boot (60 seconds)..."
sleep 60

echo ""
echo "Testing SSH connection..."
if timeout 10 ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -p 2222 ubuntu@localhost "echo 'SSH OK'" 2>/dev/null; then
    echo "  ✓ SSH connection successful!"
    echo ""
    echo "Checking for GPU in VM..."
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 2222 ubuntu@localhost "lspci | grep -i nvidia" 2>/dev/null || echo "  (GPU detection pending - may need driver installation)"
else
    echo "  ⚠ SSH not ready yet. VM may still be booting."
    echo "  Try: ssh -p 2222 ubuntu@localhost"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps to test CUDA:"
echo "  1. SSH into VM: ssh -p 2222 ubuntu@localhost"
echo "  2. Check GPU: lspci | grep -i nvidia"
echo "  3. Install NVIDIA drivers: sudo apt install -y nvidia-driver-535"
echo "  4. Install CUDA: sudo apt install -y nvidia-cuda-toolkit"
echo "  5. Test: nvidia-smi"
echo ""
echo "Monitor VM: tail -f /tmp/qemu-gpu.log"
echo "Stop VM: pkill -f 'qemu-system.*kernel-dev'"
