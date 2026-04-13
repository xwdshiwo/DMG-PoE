"""
DMG: Dynamic Modality Graph
Graph-level information propagation across modality nodes (up to 7 nodes).
Missing modality nodes receive information but their outgoing influence is attenuated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class ModalityGraphAttention(nn.Module):
    """
    Modality graph attention layer.
    Attention-weighted information propagation across modality nodes.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,           # (batch, num_modalities, embed_dim)
        mask: Optional[torch.Tensor] = None,  # (batch, num_modalities) 1=available
    ) -> torch.Tensor:
        """
        Args:
            x: modality embeddings
            mask: modality availability mask; missing modalities have reduced outgoing weight
        Returns:
            refined: propagated embeddings
        """
        batch_size, num_mod, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, num_mod, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_mod, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_mod, self.num_heads, self.head_dim).transpose(1, 2)
        # shape: (batch, num_heads, num_mod, head_dim)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # (batch, num_heads, num_mod, num_mod)
        
        # Soft mask: missing modalities as keys have reduced attention weight
        if mask is not None:
            # mask: (batch, num_mod) -> (batch, 1, 1, num_mod)
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn * (0.1 + 0.9 * key_mask)
        
        attn = F.softmax(attn, dim=-1)
        self._attn_weights = attn.detach()
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        # (batch, num_heads, num_mod, head_dim)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, num_mod, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class DMGLayer(nn.Module):
    """Single DMG layer: Attention + FFN + Residual."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = ModalityGraphAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class DynamicModalityGraph(nn.Module):
    """
    Dynamic Modality Graph network.
    Treats each modality embedding as a graph node and propagates information via GNN.
    Missing modality nodes can receive information from available ones.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Learnable modality positional encodings (up to 7 modalities)
        self.modality_pe = nn.Parameter(torch.randn(7, embed_dim) * 0.02)
        
        self.layers = nn.ModuleList([
            DMGLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.out_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],  # {name: (batch, embed_dim)}
        modality_mask: Optional[torch.Tensor] = None,   # (batch, num_modalities)
        modality_order: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            modality_embeddings: per-modality embedding dict
            modality_mask: modality availability mask
            modality_order: modality names in order (must align with mask columns)
        
        Returns:
            refined_embeddings: propagated embedding dict
        """
        if modality_order is None:
            modality_order = list(modality_embeddings.keys())
        
        num_modalities = len(modality_order)
        batch_size = next(iter(modality_embeddings.values())).shape[0]
        device = next(iter(modality_embeddings.values())).device
        
        x = torch.stack([modality_embeddings[name] for name in modality_order], dim=1)
        x = x + self.modality_pe[:num_modalities].unsqueeze(0)
        
        for layer in self.layers:
            x = layer(x, modality_mask)
        
        x = self.out_norm(x)
        
        refined = {}
        for i, name in enumerate(modality_order):
            refined[name] = x[:, i, :]

        return refined

    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        Returns attention weights from each layer for visualization.
        Each tensor has shape (batch, num_heads, num_mod, num_mod).
        """
        attn_weights = []
        for layer in self.layers:
            if hasattr(layer.attention, '_attn_weights'):
                attn_weights.append(layer.attention._attn_weights)
        return attn_weights
