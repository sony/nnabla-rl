FROM nvidia/cudagl:10.2-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    libopencv-dev \
    cmake \
    freeglut3-dev \
    wget \
    sudo

# Install cudnn
# Set default to 7.6.5.32
ARG CUDNN_VERSION=7.6.5.32
RUN apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.1 \
    libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 && \
    apt-mark hold libcudnn7

# Install conda and create env of specified python version
# Set default to python 3.6
ARG PYTHON_VERSION_MAJOR=3
ARG PYTHON_VERSION_MINOR=6
ENV PYVERNAME=${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}

ADD requirements.txt /tmp/deps/

RUN umask 0 \
    && mkdir -p /tmp/deps \
    && cd /tmp/deps \
    && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm -rf Miniconda3-latest-Linux-x86_64.sh \
    && . /opt/miniconda3/bin/activate \
    && conda install numpy \
    && conda create -n nnabla-rl-dev python=${PYVERNAME} \
    && conda activate nnabla-rl-dev \
    && pip install -U -r /tmp/deps/requirements.txt \
    && conda clean -y --all \
    && cd / \
    && rm -rf /tmp/*

ENV PATH /opt/miniconda3/envs/nnabla-rl-dev/bin:$PATH

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}display,compute

# Remove apt caches
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Add sudo user
ARG USER
ARG USER_ID
RUN groupadd -g 1000 developer && \
    useradd  -g      developer -G sudo -m -u $USER_ID -s /bin/bash ${USER} && \
    echo "${USER}:${USER}" | chpasswd

RUN echo "Defaults visiblepw"             >> /etc/sudoers
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch user
USER $USER

WORKDIR /home/nnabla-rl-dev

CMD [ "bash" ]
