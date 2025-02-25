# Use the NVIDIA PyTorch container as the base image
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Set the working directory
WORKDIR /workspace

# Install additional Python packages
RUN pip install --no-cache-dir transformers datasets accelerate deepspeed
RUN pip install --no-cache-dir tensorboardX tensorboard
RUN pip install --no-cache-dir wandb
RUN git clone https://github.com/supreethms1809/code_porting_models.git

# Set the entrypoint to bash
ENTRYPOINT ["/bin/bash"]