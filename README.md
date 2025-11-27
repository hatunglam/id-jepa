
## Introduction

A key aspiration in artificial intelligence research is to build machines that not only see the world but also understand it in a manner similar to humans. Machines can now recognize objects and generate images with remarkable accuracy, yet they still struggle with one of the most natural aspects of human vision, is to understanding depth perception. Humans effortlessly infer three-dimensional structure from incomplete or ambiguous cues, grounding perception in a way that supports reasoning and decision-making. Most artificial systems, however, remain limited to flat inputs and deterministic models that cannot capture uncertainty or generalize flexibly across contexts.  

This thesis tackles this gap by introducing the Image-Depth Joint Embedding Predictive Architecture (ID-JEPA), the first JEPA model trained jointly on RGB and depth data. The aim is to teach machines not just to see, but to internalize spatial structure in a way that mirrors human perception. To push beyond deterministic prediction, the study incorporates a variational latent space into the JEPA predictor as there are still yet to exist a JEPA implementation incorporating Bayesian methods. Inspired by Variational Autoencoders, this design encodes knowledge into structured latent distributions, allowing the model to reason under uncertainty, interpolate between learned concepts, and build richer internal representations.  

The model is evaluated through three main experiments. First, this study test whether ID-JEPA can perform cross-modal prediction, reconstructing depth features from RGB input and vice versa, to show that the architecture can align heterogeneous modalities. Second, this study examine the role of the variational latent space by comparing models with and without it, revealing that latent regularization improves training stability, reduces overfitting, and produces embeddings that capture more abstract geometric structure. Third, this study transfer the learned representations to a downstream metric depth estimation task, fine-tuning ID-JEPA within a state-of-the-art DPT framework. In this setting, the variational version achieves faster convergence and more accurate depth maps, demonstrating that its embeddings are not only abstract but also practically useful. Alongside these experiments, this study propose a simple yet novel method of stacking single-channel depth maps into three channels for pretrained vision transformers, which proves effective and open a new direction for multimodal transfer learning.

These findings demonstrate that combining multi-modal JEPA with probabilistic latent variables enables more human-like representation learning for machine learning models. They also reveal an opportunity for future research: exploring the latent space itself as a window into how machines structure knowledge, offering new insights into both AI interpretability and cognitive modeling.

## What is JEPA?

## Methodology

### Dataset
The primary dataset used will be NYU Depth V2, a widely-used RGB-D dataset consisting of indoor images captured with a Microsoft Kinect camera. It contains aligned RGB and depth sequences at 640×480 resolution, captured across various room types such as bedrooms, kitchens, and living rooms. The dataset is well-suited for this study due to its diversity in scene layouts, object types, and rich depth variation.

### Data Preprocessing
#### Student Input
The student input always takes an RGB image from the NYUv2 dataset and processes it to fit the data pipeline. The data processor is initialized from the pretrained DINOv2 image processor configuration, with center cropping applied to a size of $224 \times 224$ pixels and resizing disabled. The input image is passed through this processor, which outputs the pixel values as a tensor of shape (3, 224, 224), where 3 corresponds to the RGB channels and $224 \times 224$ is the cropped image size.

#### Teacher Input
The teacher input can be initialized and processed in two different ways: either by using a pretrained DepthAnything encoder with RGB image inputs, or by using a pretrained DINOv2 encoder with depth map inputs that are stacked to match RGB channels.

For the RGB image input, a DepthAnything pretrained processor is initialized with resizing disabled. The input image is passed through this processor to obtain the pixel values, which are then center-cropped to a resolution of $224 \times 224$. The resulting tensor has shape (3, 224, 224), where 3 corresponds to the RGB channels.

For the depth map input, a pretrained DINOv2 image processor is initialized with center cropping set to $224 \times 224$ and all other options disabled, including RGB conversion, normalization, rescaling, and resizing. Before being processed, the raw depth maps require adjustment. In training mode, the depth image is read as an 8-bit grayscale image and scaled from the range [0, 255] to [0, maximum depth] in centimeters, then clipped to ensure values do not exceed the maximum depth. This produces a normalized depth map within [0, 1]. In testing mode, the depth image is read as a floating-point array and divided by 10.0 to convert from millimeters to centimeters. After scaling, the single-channel depth map of shape (1, 224, 224) is repeated across three channels to form (3, 224, 224), matching the RGB format expected by the DINOv2 processor. The adjusted tensor is then passed through the processor to obtain pixel values for the teacher input.



## Project's codebase guide:

To download the dataset:
<pre>gdown --id 1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw </pre>

To run model training visualization on tensorboard: 
<pre>tensorboard --logdir lightning_logs --port 6006 --host 0.0.0.0 </pre>
