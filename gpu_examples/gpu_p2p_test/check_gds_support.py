#!/usr/bin/env python3
"""
Check for GPU P2P and GPUDirect Storage support
"""

import subprocess
import os
import sys

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_nvidia_fs_module():
    """Check if nvidia-fs kernel module is loaded"""
    ret, out, err = run_command("lsmod | grep nvidia_fs")
    if ret == 0 and "nvidia_fs" in out:
        return True, out.strip()
    return False, None

def check_gds_proc():
    """Check for GPUDirect Storage proc filesystem"""
    gds_paths = [
        "/proc/driver/nvidia-fs/stats",
        "/proc/driver/nvidia-fs/version"
    ]
    results = {}
    for path in gds_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    results[path] = f.read().strip()
            except:
                results[path] = "Found but cannot read"
        else:
            results[path] = "Not found"
    return results

def check_cufile_config():
    """Check cuFile configuration"""
    config_path = "/etc/cufile.json"
    if os.path.exists(config_path):
        return True, config_path
    return False, None

def check_nvme_devices():
    """Check for NVMe storage devices"""
    ret, out, err = run_command("lsblk -o NAME,SIZE,TYPE,TRAN | grep nvme")
    devices = []
    if ret == 0:
        for line in out.strip().split('\n'):
            if line:
                devices.append(line)
    return devices

def check_pci_p2p():
    """Check PCI P2P DMA capability"""
    # Check if kernel supports P2P
    ret, out, err = run_command("zgrep CONFIG_PCI_P2PDMA /proc/config.gz 2>/dev/null || echo 'config not accessible'")
    return out.strip()

print("=" * 60)
print("GPU P2P and GPUDirect Storage Support Check")
print("=" * 60)
print()

# Check nvidia-smi
print("[1] GPU Information:")
ret, out, err = run_command("nvidia-smi -L")
if ret == 0:
    print(out)
else:
    print("  nvidia-smi failed or not found")
print()

# Check nvidia-fs module
print("[2] nvidia-fs Kernel Module:")
loaded, info = check_nvidia_fs_module()
if loaded:
    print(f"  ✓ nvidia-fs module is LOADED")
    print(f"  {info}")
else:
    print(f"  ✗ nvidia-fs module is NOT loaded")
    print(f"  This is required for GPUDirect Storage with local NVMe")
print()

# Check GDS proc filesystem
print("[3] GPUDirect Storage Proc Filesystem:")
gds_proc = check_gds_proc()
for path, status in gds_proc.items():
    if status != "Not found":
        print(f"  ✓ {path}")
        if status != "Found but cannot read":
            for line in status.split('\n')[:5]:  # First 5 lines
                print(f"    {line}")
    else:
        print(f"  ✗ {path} - Not found")
print()

# Check cuFile
print("[4] cuFile Library and Configuration:")
ret, out, err = run_command("ldconfig -p | grep libcufile.so")
if ret == 0 and "libcufile" in out:
    print(f"  ✓ libcufile libraries found:")
    for line in out.strip().split('\n')[:3]:
        print(f"    {line.strip()}")
else:
    print(f"  ✗ libcufile libraries not found")

config_exists, config_path = check_cufile_config()
if config_exists:
    print(f"  ✓ cuFile config exists: {config_path}")
else:
    print(f"  ✗ cuFile config not found")
print()

# Check NVMe devices
print("[5] NVMe Storage Devices:")
nvme_devs = check_nvme_devices()
if nvme_devs:
    print(f"  ✓ Found {len(nvme_devs)} NVMe devices:")
    for dev in nvme_devs:
        print(f"    {dev}")
else:
    print(f"  ✗ No NVMe devices found")
print()

# Check PCI P2P config
print("[6] Kernel PCI P2P DMA Support:")
p2p_config = check_pci_p2p()
if "CONFIG_PCI_P2PDMA=y" in p2p_config:
    print(f"  ✓ Kernel has CONFIG_PCI_P2PDMA enabled")
elif "CONFIG_PCI_P2PDMA=m" in p2p_config:
    print(f"  ✓ Kernel has CONFIG_PCI_P2PDMA as module")
else:
    print(f"  ? Unable to determine: {p2p_config}")
print()

# Summary
print("=" * 60)
print("SUMMARY:")
print("=" * 60)

has_gpu = ret == 0
has_nvme = len(nvme_devs) > 0
has_cufile = config_exists
has_nvidia_fs = loaded

print(f"GPU Available: {'✓ YES' if has_gpu else '✗ NO'}")
print(f"NVMe Storage: {'✓ YES' if has_nvme else '✗ NO'}")
print(f"cuFile Library: {'✓ YES' if has_cufile else '✗ NO'}")
print(f"nvidia-fs Module: {'✓ YES' if has_nvidia_fs else '✗ NO'}")
print()

if has_gpu and has_nvme and has_cufile:
    if has_nvidia_fs:
        print("✓ Full GPUDirect Storage (GDS) support is AVAILABLE")
        print("  - Direct GPU-to-NVMe data transfers enabled")
    else:
        print("⚠ Partial GDS support (compatibility mode)")
        print("  - cuFile library will use fallback paths")
        print("  - To enable full GDS, load nvidia-fs module:")
        print("    sudo modprobe nvidia-fs")
else:
    print("⚠ GPUDirect Storage requirements not fully met")
    if not has_nvidia_fs:
        print("  - Missing: nvidia-fs kernel module")
    if not has_cufile:
        print("  - Missing: cuFile library/config")

print()
print("For GPU P2P (multi-GPU) support, run CUDA device queries")
print("with the compiled check_p2p_gds program.")
