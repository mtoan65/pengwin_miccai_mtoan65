# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)
FROM --platform=linux/amd64 pytorch/pytorch

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

# Add environment variables
ENV nnUNet_raw="../nnUNet_Dataset/nnUNet_raw/"
ENV nnUNet_preprocessed="../nnUNet_Dataset/nnUNet_preprocessed/"
ENV nnUNet_results="./Model/exp_3_200_epoch"
ENV nnUNet_n_proc_DA=4

# Install Python 3.10 and pip
USER root
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3-pip \
    build-essential \
    wget

# Set python3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create a user and set up the environment
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

# Set the working directory
WORKDIR /opt/app

# Copy the requirements file and install additional Python dependencies
COPY --chown=user:user ./requirements.txt /opt/app/

# Install Python dependencies using python3.10 and pip
RUN python3.10 -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --use-deprecated=legacy-resolver \
    --requirement /opt/app/requirements.txt

# Copy all files and directories from the host machine's current directory into the container's current working directory
COPY --chown=user:user . /opt/app/

# Set the entry point for the container
ENTRYPOINT ["sh", "-c", "python3.10 inference.py && python3.10 evaluation_CT_core.py"]

