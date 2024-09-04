#!/bin/bash

docker run -itd \
--gpus all \
--name suragnair_no_port \
-v $(pwd)/share:/share \
nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
