FROM nvidia/cuda:11.1.1-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Update apt, add deadsnakes PPA, and install system packages and Python 3.9.
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
        git \
        ffmpeg \
        python3.9 \
        python3.9-distutils \
        python3-pip && \
    ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# Install the cuDNN runtime (libcudnn8 is available on focal/20.04)
RUN apt-get update && apt-get install -y libcudnn8 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
