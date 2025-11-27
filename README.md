
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

#### Data Format
After both input modalities are preprocessed, each sample is returned as a dictionary with two keys: one for the student input and the other for the teacher input. This dictionary is passed to the data module, which handles batching and supplies the inputs to the model during training and evaluation.

### Data Module
The data module is implemented using PyTorch Lightning’s modular data handling interface to create dataset preparation and dataloader construction across different training phases. The first step is to initialize separate dataset instances for training, validation, and testing phase. All datasets instances are configured using a shared set of parameters, with the input crop size of $224 \times 224$, maximum depth range of 1000, and the type of teacher model used.

Separate dataloaders are constructed for the training, validation, and testing phases. Each dataloader is built using the dataset instances initialized earlier corresponding to its respective phase. All dataloaders use a batch size of 8 and are configured to utilize four worker processes for parallel data loading. Memory pinning is enabled to improve data transfer efficiency between CPU and GPU. Shuffling is applied only to the training dataloader to ensure randomized sampling during training, while validation and test dataloaders retain the original sample order for consistent evaluation.

### Student and Teacher encoder
#### The Student Encoder
The student encoder is based on a Vision Transformer (ViT) backbone and can be initialized either with pretrained weights or from scratch. When pretrained initialization is selected, weights from a publicly available DINOv2 model are loaded. 

#### The Teacher Encoder
The teacher encoder is selected based on the chosen input modality, mainly for comparision purpose. If an RGB image is used as input, the teacher encoder is initialized with pretrained weights from the DepthAnything model, which is specifically designed to take RGB images and produce depth-related feature embeddings. 

Alternatively, if a depth map is used as input, the teacher encoder is instantiated using the same Vision Transformer (ViT)-based architecture as the student. In this case, pretrained weights from DINOv2 are loaded to ensure that the encoder can extract high-level representations from the stacked depth input, which has been preprocessed to match the format of RGB images. 

Regardless of which encoder is selected, the teacher is set to evaluation mode, and its weights are frozen throughout training. Only the student encoder is updated via backpropagation.

### ID-JEPA Model
The JEPA base module serves as the core component responsible for encoding inputs and generating masked token predictions. It takes an context-target pair as input, encodes them into patch-level embeddings, and constructs context and target representations based on a masking strategy. The sampled context embeddings and masked target tokens are then passed to the predictor module, which outputs reconstructed embeddings for the masked positions.

----------diagram--------------

For the context input, the data is first passed through a pretrained student encoder, which produces the embedding representation of the input image. In testing or inference mode, this embedding is returned directly for later uses. In training mode, only a subset of the embedding is used. Specifically, a context block representing a fraction of the full set of tokens is sampled from the encoded embedding. The number of tokens to retain is determined by sampling within a predefined ratio range relative to the total number of tokens. The context block is then constructed by randomly selecting tokens from the encoded embedding up to the sampled number of tokens.

Regarding the Target input, the data is passed through a pretrained teacher encoder, which produces a sequence of depth feature maps. From this sequence, the final feature map corresponding to the deepest layer is selected to serve as the target embedding representation. This tensor has shape $(B, T, D)$, where $B$ is the batch size, $T$ is the number of patch tokens, and $D$ is the embedding dimension.
Next, the number of tokens to be masked is randomly sampled within a predefined ratio range. Based on this number, a binary target mask of shape $(B, T)$ is generated.
Since the mask tokens themselves do not contain any positional information, a positional embedding of shape $(1, T, D)$ is obtained by interpolating from the teacher encoder.
The positional embedding corresponding to these positions is then added to the mask token to create the final target masks, with shape $(B, N, D)$, where N is the number of masked tokens per sample.
Finally, the target blocks which contains the ground-truth embeddings, are constructed by slicing the encoder output at the same masked positions. These target blocks serve as the reference for prediction during training.

The context encoding and target masks are then passed to the predictor module, which is responsible for reconstructing the masked target embeddings. The predictor outputs a tensor of shape $(B, N, D)$, corresponding to the predicted embeddings at the masked positions. These predictions are returned together with the ground-truth target blocks for use in the training objective.







## Project's codebase guide:

To download the dataset:
<pre>gdown --id 1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw </pre>

To run model training visualization on tensorboard: 
<pre>tensorboard --logdir lightning_logs --port 6006 --host 0.0.0.0 </pre>
