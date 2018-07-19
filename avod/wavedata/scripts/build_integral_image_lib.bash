#!/bin/bash
set -e # exit on first error

build_integral_image_lib()
{
    cd wavedata/tools/core/lib
    cmake src
    make
}

# install cmake first
sudo apt-get install cmake
build_integral_image_lib
