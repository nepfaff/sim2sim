# Define base image.
FROM dromni/nerfstudio:0.1.12
SHELL ["/bin/bash", "-c"]

USER root

RUN apt remove libqt* qt* -y
RUN apt install -y git-lfs sudo

# Create non root user and setup environment.
RUN usermod -aG sudo user
RUN echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to new user and workdir.
USER 1000:1000
# WORKDIR /home/user

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/user/.local/bin"

# Copy nerfstudio folder and give ownership to user.
#ADD . /home/user/nerfstudio
#USER root
#RUN chown -R user:user /home/user/nerfstudio
# RUN cd /home/user && git clone https://github.com/nerfstudio-project/nerfstudio.git

# Install nerfstudio dependencies.
RUN python3.8 -m pip install -U pip
# RUN cd nerfstudio && \
#     python3.8 -m pip install -e . && \
#     cd ..

# Change working directory
WORKDIR /workspace

RUN python3.8 -m pip install drake numpy matplotlib  pre-commit  pyqt5 py  pytest  pytest-parallel  trimesh  manipulation pytorch3d  open3d  scipy 

# Install nerfstudio cli auto completion and enter shell if no command was provided.
CMD ns-install-cli --mode install && /bin/bash



