"""
Morph-Tag Transformer for grammar-only tradition classification.

A small transformer over per-token morphological tag sequences that discovers
sequential grammatical patterns (clause-opening sequences, participial chains,
agreement patterns) unreachable by hand-counted statistics. Fully BERT-free:
sees only PROIEL morph codes + dependency relation labels.

Architecture (~358K params):
    Per token: [62d morph one-hot | 16d deprel embedding] = 78d
        -> Linear(78, 128) projection
        -> + Learned positional encoding
        -> TransformerEncoder (2 layers, 4 heads, d_model=128, pre-norm, GELU)
        -> Mean pool (masked)
        -> Concat with sentence_grammar_profile (391d)
        -> MLP(519 -> 128 -> 64 -> 2) classifier head
"""

import torch
import torch.nn as nn
import math


class MorphTagTransformer(nn.Module):
    """Transformer over per-token morphological tag sequences for tradition classification."""

    def __init__(
        self,
        morph_dim: int = 62,
        deprel_vocab_size: int = 21,
        deprel_embed_dim: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        sentence_profile_dim: int = 391,
        num_classes: int = 2,
    ):
        super().__init__()

        self.d_model = d_model

        # Deprel embedding: vocab_size entries + 1 padding token
        self.deprel_embedding = nn.Embedding(
            deprel_vocab_size + 1, deprel_embed_dim, padding_idx=deprel_vocab_size
        )

        # Project concatenated [morph_one_hot | deprel_embed] to d_model
        self.input_projection = nn.Linear(morph_dim + deprel_embed_dim, d_model)

        # Learned positional encoding
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Pre-transformer layer norm and dropout
        self.embed_layer_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Classifier head: pooled transformer output + sentence grammar profile -> classes
        self.classifier = nn.Sequential(
            nn.Linear(d_model + sentence_profile_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        morph_seq: torch.Tensor,           # [batch, seq_len, morph_dim]
        deprel_seq: torch.Tensor,           # [batch, seq_len] (long indices)
        morph_mask: torch.Tensor,           # [batch, seq_len] (bool: True=valid)
        sentence_grammar_profile: torch.Tensor,  # [batch, profile_dim]
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            morph_seq: Per-position one-hot morph codes [batch, seq_len, 62]
            deprel_seq: Deprel label indices [batch, seq_len]
            morph_mask: Boolean mask, True for valid (non-pad) tokens [batch, seq_len]
            sentence_grammar_profile: Sentence-level grammar features [batch, 391]

        Returns:
            Logits [batch, num_classes]
        """
        batch_size, seq_len = morph_seq.shape[:2]

        # Embed deprels and concatenate with morph one-hot
        deprel_emb = self.deprel_embedding(deprel_seq)  # [B, S, 16]
        token_input = torch.cat([morph_seq, deprel_emb], dim=-1)  # [B, S, 78]

        # Project to d_model
        x = self.input_projection(token_input)  # [B, S, d_model]

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, S]
        x = x + self.position_embedding(positions)

        # Layer norm + dropout
        x = self.embed_layer_norm(x)
        x = self.embed_dropout(x)

        # Transformer expects src_key_padding_mask where True = IGNORE
        padding_mask = ~morph_mask  # invert: True=pad -> ignore
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Mean pool over valid tokens
        # Expand mask for broadcasting: [B, S, 1]
        valid_mask = morph_mask.unsqueeze(-1).float()
        token_count = valid_mask.sum(dim=1).clamp(min=1)  # [B, 1]
        pooled = (x * valid_mask).sum(dim=1) / token_count  # [B, d_model]

        # Concat with sentence grammar profile and classify
        combined = torch.cat([pooled, sentence_grammar_profile], dim=-1)
        logits = self.classifier(combined)

        return logits
