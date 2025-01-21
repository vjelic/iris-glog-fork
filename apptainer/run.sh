#!/bin/bash

ROCM_VERSION=${1:-6.2.3}

# Set image name
IMAGE_NAME="rocshmem_rocm_${ROCM_VERSION}.sif"
IMAGE_PATH="apptainer/images/${IMAGE_NAME}"

# Check if the image exists
if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Error: Image $IMAGE_PATH does not exist."
  exit 1
fi

export ROCM_VERSION

# --pwd "$(pwd)/rocSHMEM"

apptainer exec --cleanenv  "$IMAGE_PATH" bash

echo "Executed image: $IMAGE_NAME with ROCm version: $ROCM_VERSION"
