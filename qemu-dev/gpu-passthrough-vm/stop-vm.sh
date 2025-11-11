#!/bin/bash
# Stop running VM

# Load configuration
source "$(dirname "$0")/config.sh"

echo "========================================="
echo "Stop VM: $VM_NAME"
echo "========================================="
echo ""

PID_FILE="/tmp/qemu-$VM_NAME.pid"
MONITOR_SOCK="/tmp/qemu-monitor-$VM_NAME.sock"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")

    if ps -p $PID > /dev/null 2>&1; then
        echo "Found VM process: $PID"

        # Try graceful shutdown via QEMU monitor
        if [ -S "$MONITOR_SOCK" ]; then
            echo "Attempting graceful shutdown via QEMU monitor..."
            echo "system_powerdown" | sudo socat - UNIX-CONNECT:$MONITOR_SOCK

            echo "Waiting for VM to shutdown (30 seconds)..."
            for i in {1..30}; do
                if ! ps -p $PID > /dev/null 2>&1; then
                    echo "✓ VM shut down gracefully"
                    rm -f "$PID_FILE" "$MONITOR_SOCK"
                    exit 0
                fi
                sleep 1
            done

            echo "VM did not shutdown gracefully, forcing..."
        fi

        # Force kill if graceful shutdown failed
        echo "Killing VM process: $PID"
        sudo kill $PID

        sleep 2

        if ps -p $PID > /dev/null 2>&1; then
            echo "Force killing VM process..."
            sudo kill -9 $PID
        fi

        echo "✓ VM stopped"
        rm -f "$PID_FILE" "$MONITOR_SOCK"
    else
        echo "VM process not running (stale PID file)"
        rm -f "$PID_FILE" "$MONITOR_SOCK"
    fi
else
    echo "VM is not running (no PID file found)"
fi
echo ""
