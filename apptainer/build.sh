#!/bin/bash

# Default ROCm version
ROCM_VERSION=${1:-6.2.3}

# Build the SIF image with the specified ROCm version
export ROCM_VERSION
IMAGE_NAME="rocshmem_rocm_${ROCM_VERSION}.sif"
apptainer build apptainer/images/"$IMAGE_NAME" apptainer/triton.def

echo "Built image: $IMAGE_NAME with ROCm version: $ROCM_VERSION"
