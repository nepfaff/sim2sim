# Define base image.
FROM dromni/nerfstudio:0.1.14
ENV SHELL /bin/bash
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New\ York

SHELL ["/bin/bash", "-c"]
USER root

# Remove preinstalled QT and Install required apt packages.
# RUN apt remove libqt* qt* -y
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    git \
    git-lfs \
    sudo \
    wget \
    libxcb-xfixes0-dev \
    build-essential \
    libfontconfig1-dev \
    libdbus-1-dev \
    libfreetype6-dev \
    libicu-dev \
    libinput-dev \
    libxkbcommon-dev \
    libsqlite3-dev \
    libssl-dev \
    libpng-dev \
    libjpeg-dev \
    libglib2.0-dev \
    tzdata

# Switch to new user to sudoers without password.
RUN usermod -aG sudo user && echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to new user and workdir.
USER 1000:1000
WORKDIR /home/user

RUN cd /home/user && git clone https://github.com/RobotLocomotion/drake.git 
RUN sudo ./drake/setup/ubuntu/install_prereqs.sh -y 
WORKDIR /home/user
RUN mkdir drake-build && cd drake-build && cmake ../drake && make -j

RUN python3 -m pip install --upgrade pip setuptools pathtools promise 
RUN python3 -m pip install numpy matplotlib  pre-commit  py  pytest  pytest-parallel
ENV PYTHONPATH /home/user/drake-build/install/lib/python3.10/site-packages:${PYTHONPATH} 

# Change working directory
WORKDIR /workspace

