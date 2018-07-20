


# Sparse Pooling Implemented on Aggregate View Object Detection

## Claim

The code is based on the latest (May 10 2018) avod-fpn from [Avod-FPN of Jason Ku](https://github.com/kujason/avod). This repository is aimed to show the performance different by adding Sparse Pooling to the avod for better fusion. There is only little change on the original code to incorporate the Sparse Non-homogeneous Pooling Layer. You can run both the original avod-fpn and one with Sparse Pooling in this repository using the same tutorial of avod.

If you are interested in how to efficiently fuse features from different views and sensors. This resipotory is an example of how to add the Sparse Pooling Layer to your own network. [**Fusing Bird View LIDAR Point Cloud and Front View Camera Image for Deep Object Detection**](https://arxiv.org/abs/1711.06703)

If you are interested in the detection network itself. Please refer to [Avod-FPN of Jason Ku](https://github.com/kujason/avod) and [**Joint 3D Proposal Generation and Object Detection from View Aggregation**](https://arxiv.org/abs/1712.02294)

Some result files shown in scripts/results compare the some preliminary performance on peds and cycs between the original avod and avod with Sparse Pooling (other configs are the same) on the VALIDATION dataset. (I am never able to reproduce the test performance of detection networks on KITTI.)


## Getting Started
WZN: Implemented and tested on Ubuntu 16.04 with Python 3.5, CUDA9.0, cudnn7.1 and Tensorflow 1.8.0.
WZN: The following instructions is basically the same with avod except for the 'TRAINING CONFIGURATION' (and 'GET KITTI RESULT')


1. Clone this repo
```bash
git clone git@github.com:kujason/avod.git --recurse-submodules
```
If you forget to clone the wavedata submodule:
```bash
git submodule update --init --recursive
```

2. Install Python dependencies
```bash
cd avod
pip3 install -r requirements.txt
pip3 install tensorflow-gpu==1.3.0
```

3. Add `avod (top level)` and `wavedata` to your PYTHONPATH
```bash
# For virtualenvwrapper users
add2virtualenv .
add2virtualenv wavedata
```

```bash
# For nonvirtualenv users
export PYTHONPATH=$PYTHONPATH:'/path/to/avod'
export PYTHONPATH=$PYTHONPATH:'/path/to/avod/wavedata'
```

4. Compile integral image library in wavedata
```bash
sh scripts/install/build_integral_image_lib.bash
```

5. Avod uses Protobufs to configure model and training parameters. Before the framework can be used, the protos must be compiled (from top level avod folder):
```bash
sh avod/protos/run_protoc.sh
```

Alternatively, you can run the `protoc` command directly:
```bash
protoc avod/protos/*.proto --python_out=.
```

## Training
### Dataset
To train on the [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):
- Download the data and place it in your home folder at `~/Kitti/object`
- Go [here](https://drive.google.com/open?id=1yjCwlSOfAZoPNNqoMtWfEjPCfhRfJB-Z) and download the ``planes` folder into `~/Kitti/object/training`

The folder should look something like the following:
```
Kitti
    object
        testing
        training
            calib
            image_2
            label_2
            planes
            velodyne
        train.txt
        val.txt
```

### Mini-batch Generation
The training data needs to be pre-processed to generate mini-batches for the RPN. To configure the mini-batches, you can modify `avod/configs/mb_preprocessing/rpn_[class].config`. You also need to select the *class* you want to train on. Inside the `scripts/preprocessing/gen_mini_batches.py` select the classes to process. By default it processes the *Car* and *People* classes, where the flag `process_[class]` is set to True. The People class includes both Pedestrian and Cyclists. You can also generate mini-batches for a single class such as *Pedestrian* only.

Note: This script does parallel processing with `num_[class]_children` processes for faster processing. This can also be disabled inside the script by setting `in_parallel` to `False`.

```bash
cd avod
python scripts/preprocessing/gen_mini_batches.py
```

Once this script is done, you should now have the following folders inside `avod/data`:
```
data
    label_clusters
    mini_batches
```

### Training Configuration
There are sample configuration files for training inside `avod/configs`. You can train on the example config, or modify an existing configuration. To train a new configuration, copy a config, e.g. `avod_cars_example.config`, rename this file to a unique experiment name and make sure the file name matches the `checkpoint_name: 'avod_cars_example'` entry inside your config.

WZN: 5 additional parameters are added to enable different kinds of Sparse Pooling.  
'rpn_use_sparse_pooling: True'                      Enable Sparse Pooling right before RPN  
'rpn_sparse_pooling_use_batch_norm: True'           Use Batch Norm after the Sparse Pooling right before RPN  
'rpn_sparse_pooling_conv_after_fusion: True'        Add another convolution after the Sparse Pooling right before RPN  
'rpn_sparse_pooling_after_vgg: True'                Enable Sparse Pooling right after VGG16 and before FPN (no batch norm, no conv)  
'rpn_dual_sparse_pooling_after_vgg: True'           Fuse both LIDAR to camera and camera to LIDAR  


### Run Trainer
To start training, run the following:
```bash
python avod/experiments/run_training.py --pipeline_config=avod/configs/avod_cars_example.config
```
(Optional) Training defaults to using GPU device 1, and the `train` split. You can specify using the GPU device and data split as follows:
```bash
python avod/experiments/run_training.py --pipeline_config=avod/configs/avod_cars_example.config  --device='0' --data_split='train'
```
Depending on your setup, training should take approximately 16 hours with a Titan Xp, and 20 hours with a GTX 1080. If the process was interrupted, training (or evaluation) will continue from the last saved checkpoint if it exists.

### Run Evaluator
To start evaluation, run the following:
```bash
python avod/experiments/run_evaluation.py --pipeline_config=avod/configs/avod_cars_example.config
```
(Optional) With additional options:
```bash
python avod/experiments/run_evaluation.py --pipeline_config=avod/configs/avod_cars_example.config --device='0' --data_split='val'
```

The evaluator has two main modes, you can either evaluate a single checkpoint, a list of indices of checkpoints, or repeatedly. The evaluator is designed to be run in parallel with the trainer on the same GPU, to repeatedly evaluate checkpoints. This can be configured inside the same config file (look for `eval_config` entry).

To view the TensorBoard summaries:
```bash
cd avod/data/outputs/avod_cars_example
tensorboard --logdir logs
```

Note: In addition to evaluating the loss, calculating accuracies, etc, the evaluator also runs the KITTI native evaluation code on each checkpoint. Predictions are converted to KITTI format and the AP is calculated for every checkpoint. The results are saved inside `scripts/offline_eval/results/avod_cars_example_results_0.1.txt` where `0.1` is the score threshold.

### Run Inference
To run inference on the `val` split, run the following script:
```bash
python avod/experiments/run_inference.py --checkpoint_name='avod_cars_example' --data_split='val' --ckpt_indices=120 --device='1'
```
The `ckpt_indices` here indicates the indices of the checkpoint in the list. If the `checkpoint_interval` inside your config is `1000`, to evaluate checkpoints `116000` and `120000`, the indices should be `--ckpt_indices=116 120`. You can also just set this to `-1` to evaluate the last checkpoint.

### Viewing Results
All results should be saved in `avod/data/outputs`. Here you should see `proposals_and_scores` and `final_predictions_and_scores` results. To visualize these results, you can run `demos/show_predictions_2d.py`. The script needs to be configured to your specific experiments. The `scripts/offline_eval/plot_ap.py` will plot the AP vs. step, and print the 5 highest performing checkpoints for each evaluation metric at the moderate difficulty.

## GET KITTI RESULT
To get the result for submission to KITTI, see scripts/offline_eval/save_kitti_predictions.py  
To get AP on the validation set, see scripts/offline_eval/sample_command


## LICENSE
Copyright (c) 2018 [Jason Ku](https://github.com/kujason), [Melissa Mozifian](https://github.com/melfm), [Zining Wang](https://github.com/ZiningWang)

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