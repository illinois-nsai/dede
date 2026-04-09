#!/bin/bash

# Prerequisites:
# 1. ssh key for git loaded into ssh-agent
# 2. ssh config as follows (to enable forwarding the git key):
#
# Host *cloudlab.us *emulab.net
#         User <user>
#         IdentityFile <cloudlab ssh key>
#         ForwardAgent yes
#         StrictHostKeyChecking no
#
# DO NOT expose the dashboard host to the public, see the ShadowRay attack
# head node: ray start --head --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265
# worker node: ray start --address='<HEAD_NODE_IP>:6379'
# forward the port to the local machine by running ssh -L 5000:localhost:8265 user@host

set -e

BRANCH="${1:-ianz/cloudlab-experiments}"

MACHINES=(
    "yianz@pc525.emulab.net"
    "yianz@pc432.emulab.net"
    "yianz@pc544.emulab.net"
    "yianz@pc502.emulab.net"
    "yianz@pc512.emulab.net"
    "yianz@pc485.emulab.net"
    "yianz@pc486.emulab.net"
    "yianz@pc531.emulab.net"
    "yianz@pc547.emulab.net"
    "yianz@pc433.emulab.net"
    "yianz@pc444.emulab.net"
    "yianz@pc438.emulab.net"
    "yianz@pc434.emulab.net"
    "yianz@pc520.emulab.net"
    "yianz@pc543.emulab.net"
    "yianz@pc423.emulab.net"
)

LOG_DIR=$(mktemp -d)
echo "Logs: $LOG_DIR"

pids=()
for machine in "${MACHINES[@]}"; do
    echo "=== Launching $machine ==="
    ssh "$machine" bash -s >"$LOG_DIR/$machine.log" 2>&1 <<EOF &
cd work/dede
source .venv/bin/activate
ray start --address='155.98.38.125:6379'
EOF
    pids+=($!)
done

# Wait for all, collect exit codes
failed=()
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        failed+=("${MACHINES[$i]}")
    fi
done

echo ""
if [ ${#failed[@]} -eq 0 ]; then
    echo "=== All machines completed successfully ==="
else
    echo "=== FAILED on: ${failed[*]} ==="
    echo "Check logs in $LOG_DIR"
    exit 1
fi