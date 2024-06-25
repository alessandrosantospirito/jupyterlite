FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ruby-full \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc \
    && echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc \
    && echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc

ENV GEM_HOME="/root/gems"
ENV PATH="/root/gems/bin:$PATH"

RUN gem install jekyll bundler \
    && gem install jekyll-paginate

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
