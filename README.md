# HSID-CNN on Caffe (WSL2 + CUDA 10.2)

This repository provides the implementation of **HSID-CNN** for hyperspectral image denoising, based on the [Caffe deep learning framework](https://github.com/BVLC/caffe). The code is configured to run in **WSL2** with **CUDA 10.2** and **Python 3.10**.

---

## 📋 Requirements

- WSL2 (Ubuntu 20.04 or 22.04)
- CUDA 10.2 (installed in `/usr/local/cuda-10.2`)
- cuDNN compatible with CUDA 10.2
- Python 3.10
- OpenCV 4
- Caffe with Python layer support

---

## ⚙️ Setup

### 1. Clone this repository

```
git clone https://github.com/YOUR_USERNAME/HSID-CNN.git
cd HSID-CNN/caffe
```


### 2. Install system dependencies
```
sudo apt update
sudo apt install -y build-essential cmake libprotobuf-dev protobuf-compiler \
    libatlas-base-dev libboost-all-dev libhdf5-dev libgflags-dev libglog-dev \
    liblmdb-dev libopencv-dev python3-dev python3-pip
```


### 3. Install Python dependencies
```
pip install -r ../requirements.txt
```


### 4. Set up Makefile.config
Use the provided `Makefile.config` (edited for WSL2 + CUDA 10.2 + Python 3.10):
```
cp Makefile.config.example Makefile.config
```

Make sure the following values are set in `Makefile.config`:
```
CUDA_DIR := /usr/local/cuda-10.2
USE_CUDNN := 1
WITH_PYTHON_LAYER := 1
OPENCV_VERSION := 4
```

And set proper Python include/lib paths for Python 3.10.


### 5. Build Caffe
```
make all -j$(nproc)
make pycaffe
```

### 6. Export Python path
```
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
```


### 🚀 Run HSID-CNN
Move to the `HSID-CNN/` directory to train or test the model.