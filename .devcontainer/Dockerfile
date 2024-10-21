# FROM ubuntu:22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y software-properties-common build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev libgdbm-dev wget curl && \
    cd /usr/bin && \
    wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz && \
    tar -xf Python-3.8.10.tgz && \
    cd Python-3.8.10 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall && \
    cp ./python ../python3.8 && \
    python3.8 --version && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py pip==24.0 && \
    rm get-pip.py

RUN pip install tensorboard
RUN pip install torch
RUN pip install torchvision
RUN pip install jupyter
RUN pip install notebook
RUN pip install opencv-python
RUN pip install ultralytics
RUN pip install torch-geometric==2.0.4
RUN pip install torch-geometric-temporal

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    zlib1g-dev \
    libopenmpi-dev \
    ffmpeg \
    software-properties-common \
    lsb-release \
    sed \
    tmux \
    python3-opengl \
    && rm -rf /var/lib/apt/lists/*
