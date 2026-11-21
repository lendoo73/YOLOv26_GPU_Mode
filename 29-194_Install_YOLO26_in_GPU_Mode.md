# 29-196. Installation: Install YOLO26 in GPU Mode


## [Install PyTorch](https://pytorch.org/get-started/locally/)

Check CUDA Version
```shell
nvidia-smi
```

Install PyTorch with CUDA support
```shell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

## Install ultralytics

```shell
pip install ultralytics
```