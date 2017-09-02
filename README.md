# Instructions to run 

NYU weights path: /Users/harshm/Documents/GitHub/FCRN-DepthPrediction/tensorflow/weights/NYU_ResNet-UpProj.npy
Test images path: /Users/harshm/Documents/GitHub/FCRN-DepthPrediction/tensorflow/test/DSC02101.jpg
Command:
python predict.py --model_path=/Users/harshm/Documents/GitHub/FCRN-DepthPrediction/tensorflow/weights/NYU_ResNet-UpProj.npy --image_paths=/Users/harshm/Documents/GitHub/FCRN-DepthPrediction/tensorflow/test/DSC02101.jpg

Resources:

[Original repository](https://github.com/iro-cp/FCRN-DepthPrediction)
[CoreML advanced - for how to interface with coremltools](https://developer.apple.com/videos/play/wwdc2017/710/)
[Image editing with depth](https://developer.apple.com/videos/play/wwdc2017/508)
[Squeezing deep learning to mobile phones - tricks to reduce the size](https://www.slideshare.net/anirudhkoul/squeezing-deep-learning-into-mobile-phones)


TODO:
1. Convert the FCRN-DepthPrediction to keras, retrain and save the model in h5 model.
2. Convert the keras model to coreml model format using python package coremltools
3. iOS app to create blur effect from depth data
4. Or use keras.js to predict depth on desktop


# Deeper Depth Prediction with Fully Convolutional Residual Networks


## Introduction

This repository contains the CNN models trained for depth prediction from a single RGB image, as described in the paper "[Deeper Depth Prediction with Fully Convolutional Residual Networks](https://arxiv.org/abs/1606.00373)". The provided models are those that were used to obtain the results reported in the paper on the benchmark datasets NYU Depth v2 and Make3D for indoor and outdoor scenes respectively. Moreover, the provided code can be used for inference on arbitrary images. 

	
### TensorFlow
The code provided in the *tensorflow* folder requires accordingly a successful installation of the [TensorFlow](https://www.tensorflow.org/) library (any platform). 
The model's graph is constructed in ```fcrn.py``` and the corresponding weights can be downloaded using the link below. The implementation is based on [ethereon's](https://github.com/ethereon/caffe-tensorflow) Caffe-to-TensorFlow conversion tool. 
```predict.py``` provides sample code for using the network to predict the depth map of an input image.

## Models

The models are fully convolutional and use the residual learning idea also for upsampling CNN layers. Here we provide the fastest variant in which interleaving of feature maps is used for upsampling. For this reason, a custom layer `+dagnn/Combine.m` is provided.

The trained models - namely **ResNet-UpProj** in the paper - can also be downloaded here:

- NYU Depth v2: [MatConvNet model](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.zip), [TensorFlow model](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy)
- Make3D: [MatConvNet model](http://campar.in.tum.de/files/rupprecht/depthpred/Make3D_ResNet-UpProj.zip), TensorFlow model (soon)

## Citation

If you use this method in your research, please cite:

    @inproceedings{laina2016deeper,
            title={Deeper depth prediction with fully convolutional residual networks},
            author={Laina, Iro and Rupprecht, Christian and Belagiannis, Vasileios and Tombari, Federico and Navab, Nassir},
            booktitle={3D Vision (3DV), 2016 Fourth International Conference on},
            pages={239--248},
            year={2016},
            organization={IEEE}
    }

## License

Simplified BSD License

Copyright (c) 2016, Iro Laina  
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
