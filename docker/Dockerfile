FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

RUN apt-get update \
  && apt-get install python3.11 python3-pip -y \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN pip install \
    torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126  \
    --index-url https://download.pytorch.org/whl/cu126

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY run-evaluation.sh /opt/run-evaluation.sh

