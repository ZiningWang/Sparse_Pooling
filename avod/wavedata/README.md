# Wavedata

[1]:[https://travis-ci.com/kujason/wavedata]
[![Build Status](https://travis-ci.com/kujason/wavedata.svg?token=q1CfB5VfAVvKxUyudP69&branch=master)][1]

The repository contains the public release of the [WAVE Laboratory](http://wavelab.uwaterloo.ca/) ([GitHub](https://github.com/wavelab)) Python library of dataset helper functions. The aim of this library is to provide functions for data input/output and processing of the different components of a dataset. The functions are currently compatible with the Kitti dataset.

## Getting Started
Implemented and tested on Ubuntu 16.04 with Python 3.5.

1. Clone repository:
```
git clone git@github.com:kujason/wavedata.git
```

2. Install Dependencies:
```
cd ~/wavedata
pip3 install -r requirements.txt
```

3. Compile C++ files:

First install cmake `sudo apt-get install cmake`.

To compile with script:
```bash
cd wavedata/scripts
build_integral_image_lib.bash
```
To manually compile:
```
cd wavedata/tools/core/lib
cmake src
make
```

## LICENSE
MIT License

Copyright (c) 2018
[Ali Harakeh](https://github.com/asharakeh),
[Jason Ku](https://github.com/kujason),
[Jungwook Lee](https://github.com/jungwook-lee),
[WAVE Laboratory](http://wavelab.uwaterloo.ca/)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
