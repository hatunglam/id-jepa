import torch
from model_2.variational_predictor import PredictorVAE

embed_dim = 256
hidden_size = 128
z_dim = 64
num_heads = 4
depth = 2
layer_dropout = 0.1

model = PredictorVAE(
    embed_dim=embed_dim,
    hidden_size=hidden_size,
    z_dim=z_dim,
    num_heads=num_heads,
    depth=depth,
    layer_dropout=layer_dropout
)

batch_size = 2
num_context_patches = 10
num_target_patches = 5

context_encoding = torch.randn(batch_size, num_context_patches, embed_dim)
target_masks = torch.randn(batch_size, num_target_patches, embed_dim)

# Forward pass
prediction, mu, logvar = model(context_encoding, target_masks)

print("Prediction shape:", prediction.shape)  # Expect: (B, num_target_patches, embed_dim)
print("Mu shape:", mu.shape)                  # Expect: (B, num_context_patches + num_target_patches, z_dim)
print("Logvar shape:", logvar.shape)          