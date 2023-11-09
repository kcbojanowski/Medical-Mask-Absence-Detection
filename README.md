# Medical-Mask-Absence-Detection

[![Generic badge](https://img.shields.io/badge/version-1.0.0-green.svg)](https://shields.io/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 

![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## Overview
The Medical-Mask-Absence-Detection is a state-of-the-art real-time application designed for the detection of individuals not wearing medical masks in surveillance feeds. This solution is built using YOLOv8 and NVIDIA's DeepStream SDK, making it highly efficient for use in public safety and health monitoring.

Developed as an Engineering Thesis at AGH University by Kacper Bojanowski, this project integrates a seamless pipeline from webcam or RTSP inputs to a YOLOv8 model and outputs to a user-friendly web application.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Support](#support)
- [DeepStream SDK](#deepstream-sdk)
- [YOLOv8 and TensorRT](#yolov8-and-tensorrt)
- [License](#license)
- [Authors](#authors)

## Installation
(Instructions on how to install the application, including how to start the detection pipeline and interface with the web application, will be included soon)

## Usage
(Instructions on how to use the application, including how to start the detection pipeline and interface with the web application, will be included soon)

## Datasets
This application utilizes the following datasets for training and validation:
- [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net)
- [Face Mask Detection on Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- [Face Detection Dataset on Kaggle](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset)

## Support
For support, please open an issue in the GitHub repository or contact the author directly.

## DeepStream SDK
NVIDIA DeepStream SDK allows for the creation and deployment of scalable AI-based video analytics applications. It provides a framework for capturing, processing, and inferencing video data, optimizing the performance on NVIDIA GPUs.

## YOLOv8 and TensorRT
- **YOLOv8**: The 8th version in the YOLO series, it is a fast and accurate deep learning model for real-time object detection.
- **TensorRT**: NVIDIA TensorRT is a platform for high-performance deep learning inference that allows for the deployment of neural network models with optimized latency and throughput.

## License
This project is made available under the MIT License.

## Authors
- Kacper Bojanowski
