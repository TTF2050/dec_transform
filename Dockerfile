# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    wget \
    xpra \
    xserver-xorg-dev \
    python-is-python3 \
    python3-pip \
    python3-venv \
    patchelf \
    cmake \
    unzip \
    tensorrt \
    # build-essential \
    # libtbb-dev \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

    # vim \
    # virtualenv \
# RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
# RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.6-dev python3.6 python3-pip
# RUN virtualenv --python=python3.6 env

# RUN rm /usr/bin/python
# RUN ln -s /env/bin/python3.6 /usr/bin/python
# RUN ln -s /env/bin/pip3.6 /usr/bin/pip
# RUN ln -s /env/bin/pytest /usr/bin/pytest

# RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
#     && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /opt/mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /opt/mujoco \
    && rm mujoco.tar.gz

ENV MUJOCO_PY_MUJOCO_PATH /opt/mujoco/mujoco210
ENV LD_LIBRARY_PATH /opt/mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# COPY vendor/Xdummy /usr/local/bin/Xdummy
# RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
# COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# RUN git clone --recursive https://github.com/opencv/opencv-python.git \
#     && cd opencv-python \
#     && CMAKE_ARGS="-DCMAKE_BUILD_TYPE=RELEASE-D WITH_TBB=ON \
# -D ENABLE_FAST_MATH=1 \
# -D CUDA_FAST_MATH=1 \
# -D WITH_CUBLAS=1 \
# -D WITH_CUDA=ON \
# -D BUILD_opencv_cudacodec=OFF \
# -D WITH_CUDNN=ON \
# -D OPENCV_DNN_CUDA=ON \
# -D INSTALL_PYTHON_EXAMPLES=OFF \
# -D INSTALL_C_EXAMPLES=OFF " \
#         pip wheel . --verbose

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-469.0.0-linux-x86_64.tar.gz
RUN tar -xf google-cloud-cli-469.0.0-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh --usage-reporting false -q 



RUN curl -O https://www.atarimania.com/roms/Atari-2600-VCS-ROM-Collection.zip 
RUN unzip Atari-2600-VCS-ROM-Collection.zip -d Atari-2600-VCS-ROM-Collection || true

# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
# COPY ./requirements.txt /mujoco_py/
# COPY ./requirements.dev.txt /mujoco_py/
# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.dev.txt

# Delay moving in the entire code until the very end.
# ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
# CMD ["pytest"]
# COPY . /mujoco_py
# RUN python setup.py install

# COPY entrypoint.sh /
# ENTRYPOINT ["/entrypoint.sh"]



RUN useradd -m -s /bin/bash user && \
    echo user ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/user \
    && chmod 0440 /etc/sudoers.d/user
USER user
