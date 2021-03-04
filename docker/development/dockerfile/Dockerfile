FROM nvidia/cudagl:10.2-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    libopencv-dev \
    cmake \
    freeglut3-dev \
    wget \
    sudo \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python-openssl

# Install cudnn
# Set default to 8.0.2.39
ARG CUDNN_VERSION=8.0.2.39
RUN apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda10.2 \
    libcudnn8-dev=$CUDNN_VERSION-1+cuda10.2 && \
    apt-mark hold libcudnn8

# Add sudo user
ARG USER
ARG USER_ID
RUN groupadd -g 1000 developer && \
    useradd  -g      developer -G sudo -m -u $USER_ID -s /bin/bash ${USER} && \
    echo "${USER}:${USER}" | chpasswd

RUN echo "Defaults visiblepw"             >> /etc/sudoers
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV HOME /home/$USER

# Install pyenv
# Set default to python 3.6.0
ARG PYTHON_VERSION_MAJOR=3
ARG PYTHON_VERSION_MINOR=6
ARG PYTHON_VERSION_MICRO=0
ENV PYVERNAME=${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.${PYTHON_VERSION_MICRO}

RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv

ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
RUN echo 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> $HOME/.bashrc

RUN pyenv install ${PYVERNAME}
RUN pyenv global ${PYVERNAME}
RUN pyenv rehash
RUN chmod -R a+w $HOME/.pyenv/shims

# Install requirements
RUN $HOME/.pyenv/shims/pip install --upgrade pip
ADD requirements.txt $HOME/tmp/deps/
RUN $HOME/.pyenv/shims/pip install -U -r $HOME/tmp/deps/requirements.txt && rm -rf /tmp/*

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}display,compute

# Remove apt caches
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Switch user
USER $USER

WORKDIR /home/$USER/nnabla-rl-dev

CMD [ "bash" ]
