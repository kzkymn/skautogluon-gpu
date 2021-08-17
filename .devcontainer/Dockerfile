ARG nvidia_cuda_version=10.1-cudnn7-runtime-ubuntu18.04
FROM nvidia/cuda:${nvidia_cuda_version}

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ARG REQUIREMENTS_TXT=requirements_with_gpu.txt

RUN apt update && apt -y full-upgrade \
    && apt -y install build-essential swig \
    && apt -y install apt-utils \
    && apt -y install --no-install-recommends apt-utils dialog 2>&1 \
    && apt -y install git iproute2 procps lsb-release \
    && apt -y install \
    python3.7 \
    python3-pip \
    python3-venv \
    python3.7-venv \
    && apt -y install python3.7-dev \
    && python3.7 -m pip install -U pip \
    && python3.7 -m pip install setuptools wheel \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt -y install sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*
    
COPY $REQUIREMENTS_TXT /tmp/
RUN python3.7 -m pip install -r /tmp/$REQUIREMENTS_TXT
