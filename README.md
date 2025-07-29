* Must do:
[] 0. Finish the dataset code and figure out computing resource
[] 1. Put patch embedding inside ViT, Image = Grayscale, Depth = as it is 
[] 2. Put patch embedding inside ViT, Image = RGB, Depth = repeat 3 
[] 3. (If training does not take long) Put patch embeddings outside ViT, Image = RGB, Depth as it is (This mean student and teacher are just ViT each with separate patch embed)
[] 4. Train the best of 1, 2, 3 with latent variables 
* Email Yunus and Ali about the summary daily


* Extra: 
0. Use VGG/ ResNEt / EfficientNet / ConvNext instead of patch embedding
1. Implement a Decoder to take the trained ID-JEPA output embeddings to reconstruct Depth



