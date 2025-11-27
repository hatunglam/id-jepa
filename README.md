
## Introduction

A key aspiration in artificial intelligence research is to build machines that not only see the world but also understand it in a manner similar to humans. Machines can now recognize objects and generate images with remarkable accuracy, yet they still struggle with one of the most natural aspects of human vision, is to understanding depth perception. Humans effortlessly infer three-dimensional structure from incomplete or ambiguous cues, grounding perception in a way that supports reasoning and decision-making. Most artificial systems, however, remain limited to flat inputs and deterministic models that cannot capture uncertainty or generalize flexibly across contexts.  

This thesis tackles this gap by introducing the Image-Depth Joint Embedding Predictive Architecture (ID-JEPA), the first JEPA model trained jointly on RGB and depth data. The aim is to teach machines not just to see, but to internalize spatial structure in a way that mirrors human perception. To push beyond deterministic prediction, the study incorporates a variational latent space into the JEPA predictor as there are still yet to exist a JEPA implementation incorporating Bayesian methods. Inspired by Variational Autoencoders, this design encodes knowledge into structured latent distributions, allowing the model to reason under uncertainty, interpolate between learned concepts, and build richer internal representations.  

The model is evaluated through three main experiments. First, this study test whether ID-JEPA can perform cross-modal prediction, reconstructing depth features from RGB input and vice versa, to show that the architecture can align heterogeneous modalities. Second, this study examine the role of the variational latent space by comparing models with and without it, revealing that latent regularization improves training stability, reduces overfitting, and produces embeddings that capture more abstract geometric structure. Third, this study transfer the learned representations to a downstream metric depth estimation task, fine-tuning ID-JEPA within a state-of-the-art DPT framework. In this setting, the variational version achieves faster convergence and more accurate depth maps, demonstrating that its embeddings are not only abstract but also practically useful. Alongside these experiments, this study propose a simple yet novel method of stacking single-channel depth maps into three channels for pretrained vision transformers, which proves effective and open a new direction for multimodal transfer learning.

These findings demonstrate that combining multi-modal JEPA with probabilistic latent variables enables more human-like representation learning for machine learning models. They also reveal an opportunity for future research: exploring the latent space itself as a window into how machines structure knowledge, offering new insights into both AI interpretability and cognitive modeling.

## What is JEPA?

## Methodology

### Dataset
The primary dataset used will be NYU Depth V2, a widely-used RGB-D dataset consisting of indoor images captured with a Microsoft Kinect camera. It contains aligned RGB and depth sequences at 640×480 resolution, captured across various room types such as bedrooms, kitchens, and living rooms. The dataset is well-suited for this study due to its diversity in scene layouts, object types, and rich depth variation.

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

### The Predictor

The Predictor module receives a context embedding and a set of target mask tokens, and it produces predicted embeddings corresponding to the masked regions of the target input.

First, the context encoding with shape $(B, Tc, D)$ and the target mask tokens with shape $(B, Tt, D)$ are concatenated along the sequence dimension, resulting in an input of shape $(B, Tc + Tt, D)$. Here, $B$ is the batch size, $Tc$ is the number of context tokens, $Tt$ is the number of target tokens, and $D$ is the embedding dimension. This combined input is then passed through a linear projection layer that maps the embedding dimension $D$ into a latent dimension of 384, producing an output of shape $(B, Tc + Tt, 384)$. This lower dimension projection is called a "bottleneck", which is a mechanism to prevent model collapsing. This lightweight predictor will force the encoder to do all the representation learning work \cite{byol}

Next, the projected sequence is processed by a multi-head self-attention encoder consisting of 8 transformer blocks, each with 12 attention heads. The transformer output retains the shape $(B, Tc + Tt, 384)$. This is then normalized and passed through a final linear projection layer that maps the latent dimension back to the original embedding dimension $D$, producing a tensor of shape $(B, Tc + Tt, D)$.

Finally, only the token representations corresponding to the masked target positions are extracted, resulting in predictions of shape $(B, Tt, D)$. These predicted embeddings are then compared with the ground-truth target embeddings during training.


### The Variational Latent Predictor

#### The Fusion Model
The fusion module is designed to update the context embeddings based on information drawn from a variational latent representation, which receives two input tensors: a main sequence and a secondary update sequence, then returns a fused representation of the same shape as the main input. The fusion is performed using an 8-head multi-head cross attention layer.The original context embeddings acting as the main sequence is used as the attention query. The update sequence, which contains the latent-informed features, is used as both the attention key and value. After attention is applied, the resulting sequence is added to the original main input through a residual connection. This is followed by a layer normalization step to stabilize the trainings.

#### Reparameterization Trick

To make the gradient-based optimization possible, the model uses the reparameterization trick which transforms the random sampling operation in the latent space into a differentiable function. Given the mean $\mu$ and log-variance $\log \sigma^2$ predicted from the latent projection, the standard deviation is computed as 
\[
\sigma = \exp\left(0.5 \cdot \log \sigma^2\right)
\]

Instead of sampling $z$ directly from a normal distribution, the model samples a noise $\epsilon \sim \mathcal{N}(0, I)$ and constructs the latent variable as 


$\[
z = \mu + \sigma \cdot \epsilon
$\]

This step is important because it lets the model learn from the random sampling process. Normally, sampling a value from a distribution is a random operation, and randomness makes it hard for the model to learn using gradients. To fix this, the model uses a trick: instead of sampling the latent variable \( z \) directly, it samples a random noise \( \epsilon \) and combines it with the learned mean \( \mu \) and standard deviation \( \sigma \). This creates a new sample \( z = \mu + \sigma \cdot \epsilon \), which behaves like a random sample but is still connected to the model's parameters. This trick makes the sampling step smooth and predictable enough so that gradients can flow through it during training.

#### Updating the Context via Variational Inference
The student input is first passed through the student encoder to produce context embeddings of shape $(B, C, D)$, where $B$ is the batch size, $C$ is the sequence length, and $D$ is the embedding dimension. These embeddings are compressed by a linear projection layer that maps the dimension from $D$ to a latent dimension $Dz$. The projected tensor is then passed through two separate linear layers to generate the parameters of the latent distribution: the mean ($\mu$) and the log-variance ($\log \sigma^2$), both of shape $(B, C, Dz)$.The reparameterization trick is then applied with these variables to produce the latent space $Z$ with the same shape $(B, C, Dz)$. 

-------------------diagram------------------------

During training, a latent dropout mechanism is applied to encourage robustness. A binary dropout mask is sampled from a Bernoulli distribution with probability $(1 - p)$, where $p$ is the dropout rate. This mask is multiplied element-wise with the latent variable $Z$, randomly zeroing out a fraction of the latent dimensions.


#### Prediction using Updated Context
The sampled latent variable $z$, with shape $(B, C, Dz)$, is first projected back to the original embedding dimension D using a linear layer, resulting in a tensor of shape $(B, C, D)$. This projected latent output is then combined with the original context embeddings to pass through the Fusion module. The fusion process produces an fused context representation of the same shape $(B, C, D)$.

----------------diagram---------------------





## Project's codebase guide:

To download the dataset:
<pre>gdown --id 1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw </pre>

To run model training visualization on tensorboard: 
<pre>tensorboard --logdir lightning_logs --port 6006 --host 0.0.0.0 </pre>
