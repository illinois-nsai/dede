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
# head node: ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
# worker node: ray start --address='<HEAD_NODE_IP>:6379'

set -e

BRANCH="${1:-ianz/cloudlab-experiments}"

MACHINES=(
    "yianz@ms0620.utah.cloudlab.us"
    "yianz@ms0607.utah.cloudlab.us"
    "yianz@ms0629.utah.cloudlab.us"
    "yianz@ms0602.utah.cloudlab.us"
)

for machine in "${MACHINES[@]}"; do
    echo "=== Running on $machine ==="
    ssh "$machine" bash -s <<EOF
mkdir work
cd work
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:illinois-nsai/dede.git
cd dede
git checkout $BRANCH
sudo apt update
sudo apt install -y python3.10-dev python3.10-venv tmux
python3 -m venv .venv
source .venv/bin/activate
pip3 install -e .[dev]
pip3 install ray[default]
EOF
    echo "=== Done with $machine ==="
done
