FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ADD requirements.txt /reproducability/
ADD Dockerfile /reproducability/
RUN pip install -r /reproducability/requirements.txt
WORKDIR /workspace/
