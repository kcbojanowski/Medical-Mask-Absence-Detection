#!/bin/bash

# Function to check the presence of a software package
check_package_installed() {
  dpkg -s "$1" &> /dev/null
  if [ $? -eq 0 ]; then
    echo "$1 is installed."
  else
    echo "$1 is NOT installed."
  fi
}

# Function to check the presence of a software version
check_version_installed() {
  version=$(dpkg -s "$1" | grep 'Version' | awk '{print $2}')
  if [[ "$version" == *"$2"* ]]; then
    echo "$1 version $2 is installed."
  else
    echo "$1 version $2 is NOT installed."
  fi
}

# Function to check CUDA version
check_cuda_installed() {
  if nvcc --version | grep -q "Cuda compilation tools, release $1"; then
    echo "CUDA version $1 is installed."
  else
    echo "CUDA version $1 is NOT installed."
  fi
}

# Function to check NVIDIA driver version
check_nvidia_driver_installed() {
  if nvidia-smi | grep -q "$1"; then
    echo "NVIDIA driver version $1 is installed."
  else
    echo "NVIDIA driver version $1 is NOT installed."
  fi
}

# Function to check TensorRT version
check_tensorrt_installed() {
  if dpkg -l | grep 'libnvinfer' | grep -q "$1"; then
    echo "TensorRT version $1 is installed."
  else
    echo "TensorRT version $1 is NOT installed."
  fi
}

# Function to check GStreamer version
check_gstreamer_installed() {
  gst_version=$(gst-launch-1.0 --version --version | grep 'GStreamer' | awk '{print $2}')
  if [[ "$gst_version" == "$1" ]]; then
    echo "GStreamer version $1 is installed."
  else
    echo "GStreamer version $1 is NOT installed."
  fi
}

check_requirements() {
# Check Ubuntu Version
ubuntu_version=$(lsb_release -rs)
if [ "$ubuntu_version" == "20.04" ]; then
  echo "Ubuntu 20.04 is installed."
else
  echo "This script requires Ubuntu 20.04 but found Ubuntu $ubuntu_version."
  exit 1
fi

# Check for GStreamer
check_gstreamer_installed "1.16.3"

# Check for NVIDIA driver
check_nvidia_driver_installed "535.129.03"

# Check for CUDA
check_cuda_installed "12.1"

# Check for TensorRT
check_tensorrt_installed "8.5.3.1"

# Check if DeepStream is installed by looking for its detect
if [ -d "/opt/nvidia/deepstream" ]; then
  echo "A DeepStream installation was found."
else
  echo "No DeepStream installation was found."
fi

# Check for the presence of required packages
echo "Checking required packages:"
packages=(
  "libssl1.1"
  "libgstreamer1.0-0"
  "gstreamer1.0-tools"
  "gstreamer1.0-plugins-good"
  "gstreamer1.0-plugins-bad"
  "gstreamer1.0-plugins-ugly"
  "gstreamer1.0-libav"
  "libgstreamer-plugins-base1.0-dev"
  "libgstrtspserver-1.0-0"
  "libjansson4"
  "libyaml-cpp-dev"
  "libjsoncpp-dev"
  "protobuf-compiler"
  "gcc"
  "make"
  "git"
  "python3"
)

for package in "${packages[@]}"; do
  check_package_installed "$package"
done

echo "Script execution completed."
