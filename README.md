
## Abstract:

A key aspiration in artificial intelligence research is to build machines that not only see the world but also understand it in a manner similar to humans. Machines can now recognize objects and generate images with remarkable accuracy, yet they still struggle with one of the most natural aspects of human vision, is to understanding depth perception. Humans effortlessly infer three-dimensional structure from incomplete or ambiguous cues, grounding perception in a way that supports reasoning and decision-making. Most artificial systems, however, remain limited to flat inputs and deterministic models that cannot capture uncertainty or generalize flexibly across contexts.  

This thesis tackles this gap by introducing the Image-Depth Joint Embedding Predictive Architecture (ID-JEPA), the first JEPA model trained jointly on RGB and depth data. The aim is to teach machines not just to see, but to internalize spatial structure in a way that mirrors human perception. To push beyond deterministic prediction, the study incorporates a variational latent space into the JEPA predictor as there are still yet to exist a JEPA implementation incorporating Bayesian methods. Inspired by Variational Autoencoders, this design encodes knowledge into structured latent distributions, allowing the model to reason under uncertainty, interpolate between learned concepts, and build richer internal representations.  

The model is evaluated through three main experiments. First, this study test whether ID-JEPA can perform cross-modal prediction, reconstructing depth features from RGB input and vice versa, to show that the architecture can align heterogeneous modalities. Second, this study examine the role of the variational latent space by comparing models with and without it, revealing that latent regularization improves training stability, reduces overfitting, and produces embeddings that capture more abstract geometric structure. Third, this study transfer the learned representations to a downstream metric depth estimation task, fine-tuning ID-JEPA within a state-of-the-art DPT framework. In this setting, the variational version achieves faster convergence and more accurate depth maps, demonstrating that its embeddings are not only abstract but also practically useful. Alongside these experiments, this study propose a simple yet novel method of stacking single-channel depth maps into three channels for pretrained vision transformers, which proves effective and open a new direction for multimodal transfer learning.

These findings demonstrate that combining multi-modal JEPA with probabilistic latent variables enables more human-like representation learning for machine learning models. They also reveal an opportunity for future research: exploring the latent space itself as a window into how machines structure knowledge, offering new insights into both AI interpretability and cognitive modeling.

## Methodology:

<to be added>





To download the dataset:
<pre>gdown --id 1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw </pre>


To run model training visualization on tensorboard: 
<pre>tensorboard --logdir lightning_logs --port 6006 --host 0.0.0.0 </pre>
