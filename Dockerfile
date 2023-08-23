FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ADD requirements.txt /repository/
WORKDIR /repository/workspace/code/
RUN pip install -r ../../requirements.txt
