#!/bin/bash
# Connect to QEMU VM
# This script provides easy access to the running VM

echo "=== QEMU VM Connection ==="
echo ""
echo "Available connection methods:"
echo "  1. SSH (requires VM to be running and SSH configured)"
echo "  2. QEMU Monitor (control VM)"
echo "  3. Serial Console (view boot messages)"
echo ""

read -p "Select method (1-3): " -n 1 -r
echo

case $REPLY in
    1)
        echo "Connecting via SSH..."
        echo "Default: ssh -p 2222 user@localhost"
        read -p "Username: " username
        if [ -z "$username" ]; then
            username="user"
        fi
        ssh -p 2222 ${username}@localhost
        ;;
    2)
        echo "Connecting to QEMU Monitor..."
        echo "Useful commands:"
        echo "  info status - VM status"
        echo "  info registers - CPU registers"
        echo "  info pci - PCI devices"
        echo "  system_reset - Reset VM"
        echo "  quit - Stop VM"
        echo ""
        telnet localhost 5555
        ;;
    3)
        echo "Serial console is shown in the terminal where you started the VM"
        echo "If you started with ./start-vm.sh, check that terminal window"
        ;;
    *)
        echo "Invalid selection"
        exit 1
        ;;
esac
