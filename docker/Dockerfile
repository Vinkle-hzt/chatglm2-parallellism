FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 更换源
RUN sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
    && sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

# 更新源
RUN apt update

# 安装 git
RUN apt install -y git
RUN apt install -y wget

# 安装 conda310
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.9.0-0-Linux-x86_64.sh
RUN chmod +x Miniconda3-py310_23.9.0-0-Linux-x86_64.sh
RUN ./Miniconda3-py310_23.9.0-0-Linux-x86_64.sh -b -p /opt/conda
RUN rm Miniconda3-py310_23.9.0-0-Linux-x86_64.sh

# 设置环境变量
ENV PATH /opt/conda/bin:$PATH

# 安装 torch+cu118 numpy
RUN /opt/conda/bin/python3 -m pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN /opt/conda/bin/python3 -m pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN apt update && \
    apt install -y \
    wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
    libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*