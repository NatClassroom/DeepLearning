# Lecture 7: Convolutional Neural Networks

This lecture covers the mechanics of convolutional neural networks (CNNs) in PyTorch.

## Topics

1. PyTorch Convolution Layer: Understanding convolution operations, kernel size, padding, stride, and other arguments
2. Max Pooling Layer: Understanding pooling operations for downsampling
3. Custom nn.Module: Creating custom neural network models using PyTorch's nn.Module
4. CNN for Classification: Building a complete CNN for image classification

## Files

- `01_convolution_layer.py`: Understanding PyTorch's Conv2d layer, including kernel size, padding, stride, and output shape calculations
- `02_max_pooling.py`: Understanding max pooling operations and their effect on feature maps
- `03_custom_nn_module.py`: Creating custom neural network models by subclassing nn.Module
- `04_cnn_classification.py`: Building and training a complete CNN for image classification

## Setup

You need to have image data in a `data/2d_data/` directory (same as Lecture 2). The code will use the cat images for demonstration.

## Running the Code

Each file can be run independently:

```bash
python 01_convolution_layer.py
python 02_max_pooling.py
python 03_custom_nn_module.py
python 04_cnn_classification.py
```

