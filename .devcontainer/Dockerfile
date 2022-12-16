# Define base image.
FROM dromni/nerfstudio:0.1.12
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"]

USER root

# Remove preinstalled QT and Install required apt packages.
RUN apt remove libqt* qt* -y
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
    libglib2.0-dev

# Switch to new user to sudoers without password.
RUN usermod -aG sudo user && echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to new user and workdir.
USER 1000:1000
WORKDIR /home/user

# Download required QT version, build and install.
RUN wget https://download.qt.io/archive/qt/5.15/5.15.7/single/qt-everywhere-opensource-src-5.15.7.tar.xz && \
    tar xvf qt-everywhere-opensource-src-5.15.7.tar.xz
RUN cd qt-everywhere-src-5.15.7 && \
    ./configure -opensource -confirm-license -nomake tests -nomake examples -skip qtmultimedia \
    -skip qtquickcontrols -skip qtwebengine && \
    make -j && \
    sudo make install && \
    cd .. && \
    rm -rf qt-everywhere*

RUN cd /home/user && git clone https://github.com/RobotLocomotion/drake.git
RUN sudo ./drake/setup/ubuntu/install_prereqs.sh -y 
WORKDIR /home/user
RUN mkdir drake-build && cd drake-build && cmake --verbose_failures ../drake && make -j

RUN python3.8 -m pip install --upgrade pip setuptools pathtools promise 
RUN python3.8 -m pip install numpy matplotlib  pre-commit  py  pytest  pytest-parallel
# RUN export PYTHONPATH=${PWD}/install/lib/python3.8/site-packages:${PYTHONPATH} 
ENV PYTHONPATH /home/user/drake-build/install/lib/python3.8/site-packages:${PYTHONPATH} 

# Change working directory
WORKDIR /workspace



