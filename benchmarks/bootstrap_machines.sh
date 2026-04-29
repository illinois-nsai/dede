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
# If on cloudlab, make sure to only use the local subnet (use ip addr show), then use iptables to block all non-subnet trafific to 6379
# sudo iptables -A INPUT -p tcp --dport 6379 -s 10.10.1.0/24 -j ACCEPT
# sudo iptables -A INPUT -p tcp --dport 6379 -j DROP
# head node: ray start --head --port=6379 --node-ip-address <local ip> --dashboard-host=127.0.0.1 --dashboard-port=8265
# worker node: ray start --address='<HEAD_NODE_IP>:6379'
# forward the port to the local machine by running ssh -L 5000:localhost:8265 user@host

set -e

BRANCH="${1:-ianz/cloudlab-experiments}"

MACHINES=(
    # "yianz@ms1327.utah.cloudlab.us"
    "yianz@ms1339.utah.cloudlab.us"
    "yianz@ms1325.utah.cloudlab.us"
    "yianz@ms1328.utah.cloudlab.us"
)

LOG_DIR=$(mktemp -d)
echo "Logs: $LOG_DIR"

pids=()
for machine in "${MACHINES[@]}"; do
    echo "=== Launching $machine ==="
    ssh "$machine" bash -s >"$LOG_DIR/$machine.log" 2>&1 <<EOF &
cd work/dede
source .venv/bin/activate
ray start --address='10.10.1.1:6379'

# mkdir work
# cd work
# GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:illinois-nsai/dede.git
# cd dede
# git checkout $BRANCH
# sudo apt update
# sudo apt install -y python3.10-dev python3.10-venv tmux
# python3 -m venv .venv
# source .venv/bin/activate
# pip3 install -e .[dev]
# pip3 install ray[default]

# sudo apt install -y linux-tools-common linux-tools-5.15.0-168-generic linux-cloud-tools-5.15.0-168-generic 
# sudo cpupower frequency-set -u 2.0GHz
# echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost || true
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