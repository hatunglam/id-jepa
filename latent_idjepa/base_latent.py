import torch
import torch.nn as nn
from .predictor import Predictor


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, main, update_with):
        fused, _ = self.cross_attn(query=main, key=update_with, value=update_with)
        return self.norm(main + fused)
    
class JEPAVAriationalLatent(nn.Module):
    def __init__(self,
                 image_encoder,
                 depth_encoder,
                 decoder_depth,
                 n_heads,
                 latent_num_heads=8,
                 fusion_module = CrossAttentionFusion,
                 latent_dim=512,
                 latent_dropout_prob=0.1,
                 predictor_embed_dim=None,
                 mode="train",
                 context_ratio_range=(0.85, 0.95),
                 target_mask_range=(0.15, 0.25),
                 kl_anneal_start=0,
                 kl_anneal_end=10000,
                 kl_anneal_max=1.0,
                 **kwargs):

        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_heads = n_heads
        self.mode = mode.lower()
        self.context_ratio_range = context_ratio_range
        self.target_mask_range = target_mask_range
        self.embed_dim = image_encoder.config.hidden_size
        assert self.embed_dim is not None, "image_encoder.config.hidden_size is None!"
        print(f"[DEBUG] embed_dim: {self.embed_dim}")
        assert self.embed_dim == image_encoder.config.hidden_size
    
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)

        self.image_encoder = image_encoder
        self.depth_encoder = depth_encoder

        for p in self.depth_encoder.parameters():
            p.requires_grad = False # Freeze depth encoder

        self.predictor = Predictor(embed_dim=self.embed_dim,
                                   num_heads=self.n_heads,
                                   depth=decoder_depth,
                                   predictor_embed_dim=predictor_embed_dim)

        # Variational Latent Predictor:

        self.latent_dim = latent_dim
        self.latent_dropout_prob = latent_dropout_prob

        # Project Context to Latent space with variational inference
        self.project_to_latent = nn.Linear(self.embed_dim, self.latent_dim)
        self.to_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.to_logvar = nn.Linear(self.latent_dim, self.latent_dim)
        # Project latent back to embedding input dimension
        self.project_from_latent = nn.Linear(self.latent_dim, self.embed_dim)

        self.latent_mask_token = nn.Parameter(torch.randn(1, 1, self.latent_dim))
        nn.init.trunc_normal_(self.latent_mask_token, 0.02)

        self.fusion_module = fusion_module(dim=self.embed_dim,
                                           num_heads=latent_num_heads,
                                           )
        
        self.kl_anneal_start = kl_anneal_start
        self.kl_anneal_end = kl_anneal_end
        self.kl_anneal_max = kl_anneal_max
        # self.global_step = 0

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Convert log-variance to standard deviation
        eps = torch.randn_like(std)  # Sample epsilon ~ N(0, I)
        return mu + eps * std    

    def forward_latent(self, context_encoding):
        # Input -> Hidden
        compressed = self.project_to_latent(context_encoding)
        # Hidden -> mu, logvar
        mu = self.to_mu(compressed)
        logvar = self.to_logvar(compressed)
        z = self.reparameterize(mu, logvar)

        if self.training and self.latent_dropout_prob > 0.0:
            dropout_mask = torch.bernoulli((1 - self.latent_dropout_prob) * torch.ones_like(z))
            z = z * dropout_mask

        return z, mu, logvar
    
    def forward_base(self, image, depth):
        test_mode = self.mode == "test"

        # IMAGE: 
        # -------------------------------------------------------
        # Encode the Context input with a Pretrained VIT Image Encoder
        image_embeddings = self.image_encoder(image).last_hidden_state
        if test_mode:
            # Return only the Context Embedding when in Test mode 
            return image_embeddings

        batch, n_context_patches, embed_dim = image_embeddings.shape

        z, mu, logvar = self.forward_latent(image_embeddings)
        latent_expanded = self.project_from_latent(z)

        # Fuse latent and context
        updated_image_embeddings = self.fusion_module(main=image_embeddings,
                                                      update_with=latent_expanded)
        
        num_context_blocks = self.sample_num_blocks(
            T=n_context_patches,
            min_ratio=self.context_ratio_range[0],
            max_ratio=self.context_ratio_range[1],
            exclude_cls=True,
        )

        context_encoding = self.sample_context_blocks(
            updated_image_embeddings, num_context_blocks
        )
        # -------------------------------------------------------

        # DEPTH:
        # -------------------------------------------------------
        # Encode the Context input with a Pretrained VIT Depth
        depth_embeddings = self.depth_encoder(depth).feature_maps[-1]
        
        batch, n_target_patches, embed_dim = depth_embeddings.shape

        num_target_masks = self.sample_num_blocks(
            T=n_target_patches,
            min_ratio=self.target_mask_range[0],
            max_ratio=self.target_mask_range[1],
            exclude_cls=True,
        )

        target_mask = self.create_fixed_mask(
            batch_size=batch,
            num_tokens=n_target_patches,
            num_masked=num_target_masks,
            device=self._device
        )

        batch, n_chans, height, width = depth.shape
        pos_embeddings = (
            self.depth_encoder.embeddings.interpolate_pos_encoding(depth_embeddings,
                                                                   height,
                                                                   width
                                                                   )
        ) # (1, n_tokens, embed_dim)

        pos_embeddings = pos_embeddings.squeeze(0) # (n_tokens, embed_dim)
        
        (target_masks, target_blocks) = self.create_target_masks_and_blocks(last_hidden_state=depth_embeddings,
                                                                            pos_embeddings=pos_embeddings,
                                                                            mask_token=self.mask_token,
                                                                            mask=target_mask,
                                                                            )
        
        batch_size, num_target_blocks, embed_dim = target_blocks.shape

        predictions = self.predictor(context_encoding=context_encoding,
                                         target_masks=target_masks)
        
        return (predictions,
                target_blocks,
                mu,
                logvar)

    def sample_num_blocks(
        self, T: int, min_ratio: float, max_ratio: float, exclude_cls: bool = True
    ):
        """
        Samples a single integer value based on a ratio range.

        Parameters
        ----------
        T : int
            Total number of tokens (including CLS).
        min_ratio : float
            Minimum proportion of tokens to sample.
        max_ratio : float
            Maximum proportion of tokens to sample.
        exclude_cls : bool
        Whether to exclude the CLS token (index 0) from sampleing.

        Returns
        -------
        int
            Number of tokens to sample.
        """
        num_candidates = T - 1 if exclude_cls else T
        min_num_samples = max(1, int(min_ratio * num_candidates))
        max_num_samples = max(
            min_num_samples + 1, int(max_ratio * num_candidates)
        )  # ensure > min
        return torch.randint(
            low=min_num_samples, high=max_num_samples + 1, size=(1,)
        ).item()

    def create_fixed_mask(
        self,
        batch_size: int,
        num_tokens: int,
        num_masked: int,
        device: torch.device,
        exclude_cls: bool = True,
    ) -> torch.BoolTensor:
      """
      Creates a per-sample boolean mask with exactly `num_masked` masked positions.

      Parameters
      ----------
      batch_size : int
          Number of samples in the batch.
      num_tokens : int
          Number of total tokens per sample (e.g. 257).
      num_masked : int
          Number of tokens to mask per sample.
      device : torch.device
          Device to place the output mask on.
      exclude_cls : bool
          Whether to exclude the CLS token (index 0) from being masked.

      Returns
      -------
      torch.BoolTensor
          A tensor of shape (B, T) with exactly `num_masked` True values per row.
      """
      valid_indices = (
          list(range(1, num_tokens)) if exclude_cls else list(range(num_tokens))
      )

      mask: torch.BoolTensor = torch.zeros(
          (batch_size, num_tokens), dtype=torch.bool, device=device
      )

      for b in range(batch_size):
          selected = torch.randperm(len(valid_indices), device=device)[:num_masked]
          masked_indices = torch.tensor(valid_indices, device=device)[selected]
          mask[b, masked_indices] = True

      return mask
    
    def create_target_masks_and_blocks(
        self,
        last_hidden_state: torch.Tensor,  # (B, T, D)
        pos_embeddings: torch.Tensor,  # (T, D)
        mask_token: nn.Parameter,  # (1, 1, D)
        mask: torch.BoolTensor,  # (B, T)
    ):

        """
        Extracts target masks and target blocks from masked positions in ViT encoder output.

        For all masked positions (excluding [CLS]), returns:
          - target_masks: mask_token + pos_embedding
          - target_blocks: original hidden states at those positions

        Parameters
        ----------
        last_hidden_state : torch.Tensor
            ViT encoder output, shape (B, T, D)
        pos_embeddings : torch.Tensor
            Positional embeddings, shape (B, T, D)
        mask_token : nn.Parameter
            Learnable mask token, shape (1, 1, D)
        mask : torch.BoolTensor
            Boolean mask indicating which positions to mask, shape (B, T)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            target_masks : torch.Tensor of shape (B, N_masked, D)
                Mask token plus positional embedding at masked positions
            target_blocks : torch.Tensor of shape (B, N_masked, D)
                Ground-truth hidden states at the masked positions
        """
        B, T, D = last_hidden_state.shape

        # Ensure CLS token is not masked (typically index 0)
        mask = mask.clone()
        mask[:, 0] = False

        # Expand mask token to (B, T, D)
        mask_token_expanded = mask_token.expand(B, T, D)  # (B, T, D)

        # Compute context masks
        target_masks = []
        target_blocks = []

        for b in range(B):
            masked_indices = mask[b]  # (T,)

            target_masks.append(
                mask_token_expanded[b][masked_indices] + pos_embeddings[masked_indices]
            )  # (N_masked, D)
            target_blocks.append(last_hidden_state[b][masked_indices])  # (N_masked, D)

        return torch.stack(target_masks).cuda(), torch.stack(target_blocks).cuda()

    def sample_context_blocks(
        self,
        image_embeddings,
        num_context_blocks,
        exclude_cls=True,
    ):
        """
        Randomly samples `num_context_blocks` from depth_embeddings per sample.

        Parameters
        ----------
        image_embeddings : torch.Tensor
            Input embeddings of shape (B, T, D)
        num_context_blocks : int
            Number of blocks to sample per sample
        exclude_cls : bool
            Whether to exclude the CLS token (index 0)

        Returns
        -------
        torch.Tensor
            Context blocks of shape (B, num_context_blocks, D)
        """
        B, T, D = image_embeddings.shape
        context_blocks = []

        # Range of valid indices (excluding CLS if needed)
        valid_indices = list(range(1, T)) if exclude_cls else list(range(T))
        for b in range(B):
            selected_indices = torch.randperm(len(valid_indices))[:num_context_blocks]
            selected_indices = torch.tensor(
                valid_indices, device=image_embeddings.device
            )[selected_indices]

            blocks_b: torch.Tensor = image_embeddings[b][
                selected_indices
            ]  # shape: (num_context_blocks, D)
            context_blocks.append(blocks_b)

        return torch.stack(context_blocks).cuda()  # shape: (B, num_context_blocks, D)
    
    def get_kl_weight(self) -> float:
      
        if self.global_step < self.kl_anneal_start:
            return 0.0

        elif self.global_step > self.kl_anneal_end:
            return self.kl_anneal_max

        else:
            pct = (self.global_step - self.kl_anneal_start) / (self.kl_anneal_end - self.kl_anneal_start)

            return self.kl_anneal_max * pct
        

if __name__ == "__main__":
    crs_attn = CrossAttentionFusion(10, 2)
    main = torch.randn(1, 5, 10)
    update_with = torch.randn(1, 5, 10)

    out = crs_attn(main, update_with)
    print(out.shape) 