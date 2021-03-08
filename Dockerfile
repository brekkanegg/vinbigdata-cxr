# Usage
# docker build --tag minki/cxr:v1.0 -f Dockerfile . \
# --build-arg USER_ID=$(id -u ${USER}) --build-arg GROUP_ID=$(id -g ${USER})
FROM pytorch/pytorch:1.7.0-cuda10.1-cudnn7-devel 
MAINTAINER brekkanegg@gmail.com

RUN apt-get update && apt-get -y --no-install-recommends install \
    curl \
    sudo \
    python3-dev \
    python3-pip \
    python3-setuptools \
    vim \
    git \
    wget \
    bzip2 \
    nginx \
    libgdcm-tools \
    libsm6 \
    libxext6 \
    libxrender-dev \ 
    libglib2.0-0 

RUN pip3 install --upgrade pip

COPY requirements.txt /usr/local/requirements.txt
RUN pip install -r /usr/local/requirements.txt

# https://jtreminio.com/blog/running-docker-containers-as-current-host-user/
# --build-arg USER_ID=$(id -u ${USER})
# --build-arg GROUP_ID=$(id -g ${USER})
ARG USER_ID
ARG GROUP_ID
ARG USER_NAME=minki
ARG GROUP_NAME=cxr
RUN groupadd -g ${GROUP_ID} ${GROUP_NAME}
RUN useradd -u ${USER_ID} ${USER_NAME} -g ${GROUP_NAME} && adduser ${USER_NAME} sudo
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir /home/${USER_NAME}
RUN export HOME=/home/${USER_NAME}
# RUN export HOME=/workspace
RUN chown -R ${USER_NAME}:${GROUP_NAME} /home/${USER_NAME}
# RUN chown -R ${USER_NAME}:${GROUP_NAME} /workspace
RUN chown -R ${USER_NAME} ~/.cache

USER ${USER_NAME}

CMD ["/bin/bash"]


