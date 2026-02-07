FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt/CosyVoice

# Use faster mirrors
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

# Install system dependencies
RUN apt-get update -y && apt-get install -y git unzip git-lfs g++
RUN git lfs install

# Clone CosyVoice
RUN git clone --recursive https://github.com/pageinfoabout/cosy_custom.git

# Install Python dependencies
# IMPORTANT: install requirements first, then force HyperPyYAML + ruamel.yaml versions
RUN cd CosyVoice && \
    pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com --no-cache-dir
RUN pip install --force-reinstall HyperPyYAML==1.1.0 ruamel.yaml==0.17.21
# Compile gRPC protos
RUN cd CosyVoice/runtime/python/grpc && \
    python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto



