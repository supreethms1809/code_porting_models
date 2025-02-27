# Use the NVIDIA PyTorch container as the base image
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Set the working directory
WORKDIR /workspace

# Install additional Python packages
RUN pip install --no-cache-dir transformers datasets accelerate deepspeed
RUN pip install --no-cache-dir tensorboardX tensorboard
RUN pip install --no-cache-dir wandb tiktoken blobfile sentencepiece
RUN pip install --no-cache-dir -U "huggingface_hub[cli]"

ENV BASE_DIR=/workspace/code_porting_models
ENV HF_HOME=/workspace/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub

# Set the entrypoint to bash
ENTRYPOINT ["/bin/bash"]

# To build the image, use the following command:
# docker build -t code_porting_models .
# To run the container with GPU support, use the following command:
# docker run --gpus all -it --rm -v $(pwd):/workspace code_porting_models