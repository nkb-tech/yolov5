#!/bin/bash

export DOCKER_UID=$(id -u)
export DOCKER_GID=$(id -g)
export DOCKER_USER_PASSWORD=user
export DOCKER_DEFAULT_PATH=/home
export DOCKER_PROJECT_PATH=src
export DOCKER_IMAGE_NAME=ilyabasharov
export DOCKER_IMAGE_VERSION=yolo-deploy:v1.0.0
export DOCKER_CONTAINER_NAME=yolo-deploy
export DOCKER_USER_NAME=${USER}
export DOCKER_PROJECT_NAME=project
export DEFAULT_DATA_PATH=/home/alexander/nkbtech/petsearch/data
export DOCKER_DATA_PATH=datasets
export DOCKERFILE_PATH=utils/docker/Dockerfile-ds