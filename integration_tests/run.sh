#!/usr/bin/env bash

cd "$(dirname "$0")"

LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
LOCAL_IMAGE_NAME="air-quality-prediction:${LOCAL_TAG}"

docker build -t ${LOCAL_IMAGE_NAME} ..

docker run -it --rm -p 9696:9696 ${LOCAL_IMAGE_NAME}

pipenv run python test_docker.py