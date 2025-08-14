#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.


IMAGE_NAME=${1:-"iris-dev"}
WORKSPACE_DIR=${2:-"$(pwd)"}

alias drun='docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME:$HOME -v $HOME/.ssh:/root/.ssh:ro -w $HOME --shm-size=16G --ulimit memlock=-1 --ulimit stack=67108864'
drun $IMAGE_NAME
