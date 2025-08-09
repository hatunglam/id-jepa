* Must do:
0. Finish the dataset code and figure out computing resource
1. Put patch embedding inside ViT, Image = Grayscale, Depth = as it is 
2. Put patch embedding inside ViT, Image = RGB, Depth = repeat 3 
3. (If training does not take long) Put patch embeddings outside ViT, Image = RGB, Depth as it is (T`his mean student and teacher are just ViT each with separate patch embed)
4. Train the best of 1, 2, 3 with latent variables 
* Email Yunus and Ali about the summary daily


* Extra: 
0. Use VGG/ ResNEt / EfficientNet / ConvNext instead of patch embedding
1. Implement a Decoder to take the trained ID-JEPA output embeddings to reconstruct Depth



1- RGB (3 channel) + Depth (1 channel) 
   Project Depth to 3 channel (not inside teacher, will be outside, to faciliate exact copy of teacher from student. Thus not updated using EMA).

2- Do not use depth_proj, copy teacher from student exactly as it is. To make depth 3 channel, stack depth 3 times.

3- Copy teacher exactly from student, do not stack depth but use gray scale images.

gdown --id 1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw