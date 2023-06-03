#!/bin/bash

TRITON_TAG="22.03-py3"
MODEL_REPO_HOST_PATH=${PWD}/utils/triton
MODEL_REPO_CONTAINER_PATH=/triton

docker run \
    --gpus all \
    -it \
    --rm \
    --ipc host \
    --net host \
    -v ${MODEL_REPO_HOST_PATH}:${MODEL_REPO_CONTAINER_PATH} \
    "nvcr.io/nvidia/tritonserver:${TRITON_TAG}" \
    ./bin/tritonserver \
        --model-repository=${MODEL_REPO_CONTAINER_PATH} \
        --backend-directory=./backends \
        --backend-config=tensorrt,coalesce-request-input=true \
        --model-control-mode=none \
        --allow-grpc=true \
        --grpc-port=8000 \
        --allow-http=false