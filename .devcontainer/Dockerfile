FROM ubuntu:22.04

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
    
RUN pip install wheel==0.38.4
RUN pip install setuptools==66.0.0
RUN pip install gym==0.21.0
RUN pip install stable_baselines3==1.8.0
RUN pip install tensorboard
RUN pip install opencv-python
RUN pip install torch
RUN pip install torchvision
RUN pip install torchaudio
RUN pip install jupyter
RUN pip install notebook
RUN pip install numpy
RUN pip install gym_super_mario_bros==7.3.0
RUN pip install nes_py
RUN pip install gym[atari,accept-rom-license]==0.21.0

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    vim \
    git \
    zlib1g-dev \
    libopenmpi-dev \
    ffmpeg \
    software-properties-common \
    lsb-release \
    sed \
    tmux \
    python3-opengl \
    x11-apps \
    x11-utils \
    xauth \
    ruby-full \
    && rm -rf /var/lib/apt/lists/*

RUN pip install jupyterlite-core

# # ------------ PERSONAL SETUP ----------
# # Install neovim
# RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz && \
#     rm -rf /opt/nvim && \
#     tar -C /opt -xzf nvim-linux64.tar.gz && \
#     rm nvim-linux64.tar.gz

# ENV PATH="$PATH:/opt/nvim-linux64/bin"

# # Vim
# RUN mkdir -p /root/.config/nvim
# RUN curl -L https://github.com/ucl98/config/raw/main/nvim/init.lua -o /root/.config/nvim/init.lua

# # Tmux
# RUN git clone https://github.com/ucl98/config.git /tmp/config \
#     && cp -r /tmp/config/tmux /root/.config/tmux \
#     && rm -rf /tmp/config

# # Note taking
# RUN sed -i "s|builtin.find_files({ cwd = \"~/Documents/obsidian/quickview\" })|builtin.find_files({ cwd = \"/workspaces/StreetFighterRL/notes\" })|" ~/.config/nvim/init.lua
