import torch
import torch.nn as nn
from x_transformers import Encoder
from .predictor import Predictor

class JEPA_base(nn.Module):

    def __init__(
            self,
            image_encoder,
            depth_encoder,
            decoder_depth,
            n_heads,
            predictor_embed_dim, # Latent space size of the Predictor
            post_enc_norm=False,
            mode="train",
            context_ratio_range=(0.85, 0.95),
            target_mask_range=(0.15, 0.25),
            freeze="depth",
            **kwargs
    ):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_heads = n_heads
        self.mode = mode.lower()
        self.context_ratio_range = context_ratio_range
        self.target_mask_range = target_mask_range
        self.embed_dim = image_encoder.config.hidden_size # Need edit
        assert self.embed_dim == image_encoder.config.hidden_size

        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)

        self.post_enc_norm = post_enc_norm
        self.post_enc_norm_jepa = (
            nn.LayerNorm(self.embed_dim) if self.post_enc_norm else nn.Identity()
        )
        
        self.image_encoder = image_encoder
        self.depth_encoder = depth_encoder

        if freeze == "depth":
            for p in self.depth_encoder.parameters():
                p.requires_grad = False # Freeze depth encoder
        elif freeze == "image":
            for p in self.image_encoder.parameters():
                p.requires_grad = False # Freeze image encoder


        # Initialize Predictor module
        self.predictor = Predictor(
            embed_dim=self.embed_dim,
            num_heads=self.n_heads,
            depth=decoder_depth,
            predictor_embed_dim=predictor_embed_dim
            )

    def forward_base(
            self,
            image,
            depth
    ):
        test_mode = self.mode == "test" # False if self.mode == "train"

        # IMAGE: 
        # -------------------------------------------------------
        # Encode the Context input with a Pretrained VIT Image Encoder
        image_embeddings = self.image_encoder(image).last_hidden_state
        if test_mode:
            # Return only the Context Embedding when in Test mode 
            return image_embeddings
        
        # Get Context Block
        batch, n_context_patches, embed_dim = image_embeddings.shape

        num_context_blocks: int = self.sample_num_blocks(
            T=n_context_patches,
            min_ratio=self.context_ratio_range[0],
            max_ratio=self.context_ratio_range[1],
            exclude_cls=True,
        )
        
        context_encoding = self.sample_context_blocks(
            image_embeddings, num_context_blocks
        )
        # -------------------------------------------------------

        # DEPTH:
        # -------------------------------------------------------
        # Encode the Context input with a Pretrained VIT Depth(?) Encoder
        with torch.no_grad():
            depth_embeddings = self.depth_encoder(depth).last_hidden_state

        batch, n_target_patches, embed_dim = depth_embeddings.shape

        num_target_masks = self.sample_num_blocks(
            T=n_target_patches,
            min_ratio=self.target_mask_range[0],
            max_ratio=self.target_mask_range[1],
            exclude_cls=True,
        )

        # Create empty target masks
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
        # -------------------------------------------------------

        predictions = self.predictor(context_encoding=context_encoding,
                                    target_masks=target_masks
                                    )
        
        return (predictions, target_blocks)
    
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

        print("last hidden: ", last_hidden_state.shape, "  expect: ", (B, T, D ))
        print("mask: ", mask.shape,  "  expect: ", (B, T))
        print("mask token: ", mask_token.shape,  "  expect: ", (1, 1, D ))
        print("mask token exp: ", mask_token_expanded.shape,  "  expect: ", (B, T, D ))
        print("pos emb: ", pos_embeddings.shape, "  expect: ", (T, D ))

        # Compute context masks
        target_masks = []
        target_blocks = []

        for b in range(B):
            masked_indices = mask[b]  # (T,)

            target_masks.append(
                mask_token_expanded[b][masked_indices] + pos_embeddings[masked_indices]
            )  # (N_masked, D)
            target_blocks.append(last_hidden_state[b][masked_indices])  # (N_masked, D)

        return torch.stack(target_masks), torch.stack(target_blocks)

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
        depth_embeddings : torch.Tensor
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

        return torch.stack(context_blocks)  # shape: (B, num_context_blocks, D)









