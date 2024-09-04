#!/bin/bash
apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ssh \
         tzdata \
         ca-certificates \
         libjpeg-dev \
         libsm6 \
         libxext6 \
         libxrender-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*