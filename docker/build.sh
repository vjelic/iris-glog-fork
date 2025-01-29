#!/bin/bash

IMAGE_NAME=${1:-"sk-pyrocshmem"}

docker build -t $IMAGE_NAME .