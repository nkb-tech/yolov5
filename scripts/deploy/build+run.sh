#!/bin/bash

export DOCKER_UID
export DOCKER_GID
export DOCKER_IMAGE_NAME
export DOCKER_IMAGE_VERSION
export DOCKER_USER_NAME
export DOCKER_USER_PASSWORD
export DOCKER_PROJECT_PATH
export DOCKER_PROJECT_NAME
export DOCKER_DEFAULT_PATH
export DOCKER_CONTAINER_NAME
export DEFAULT_DATA_PATH
export DOCKER_DATA_PATH
export DOCKERFILE_PATH

docker build \
    --tag ${DOCKER_IMAGE_NAME}/${DOCKER_IMAGE_VERSION} \
    --file ${DOCKERFILE_PATH} \
    .

docker run \
    -itd \
    --ipc host \
    --gpus all \
    --name ${DOCKER_CONTAINER_NAME} \
    --net host \
    --env "DISPLAY" \
    --env "QT_X11_NO_MITSHM=1" \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --volume $PWD:/usr/src/app/:rw \
    --volume ${DEFAULT_DATA_PATH}:/usr/src/app/${DOCKER_DATA_PATH}:rw \
    --privileged \
    ${DOCKER_IMAGE_NAME}/${DOCKER_IMAGE_VERSION}

docker start ${DOCKER_CONTAINER_NAME}
docker exec \
    -it ${DOCKER_CONTAINER_NAME} \
    bash -c "cd /usr/src/app/; bash"