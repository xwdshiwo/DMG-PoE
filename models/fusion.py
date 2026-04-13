"""
PoE Fusion: Product of Experts
Uncertainty-weighted fusion using precision τ = 1/σ².
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class PoEFusion(nn.Module):
    """
    Product of Experts fusion.
    
    Under the diagonal Gaussian assumption, the closed-form PoE solution is:
    - Fused mean:     z = Σ(τ_m * e_m) / Σ(τ_m)
    - Fused variance: Var(z) = 1 / Σ(τ_m)
    
    where τ_m = 1/σ²_m is the precision of modality m.
    """
    
    def __init__(
        self,
        embed_dim: int,
        min_var: float = 1e-4,
        max_var: float = 100.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.min_var = min_var
        self.max_var = max_var
    
    def forward(
        self,
        means: Dict[str, torch.Tensor],      # {name: (batch, embed_dim)}
        logvars: Dict[str, torch.Tensor],    # {name: (batch, embed_dim)}
        modality_mask: Optional[torch.Tensor] = None,  # (batch, num_modalities)
        modality_order: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            means: per-modality means
            logvars: per-modality log-variances
            modality_mask: availability mask (1=available, 0=missing)
            modality_order: modality names in order (must align with mask columns)
        
        Returns:
            fused_mean: (batch, embed_dim)
            fused_var:  (batch, embed_dim)
        """
        if modality_order is None:
            modality_order = list(means.keys())
        
        batch_size = next(iter(means.values())).shape[0]
        device = next(iter(means.values())).device
        
        precision_sum = torch.zeros(batch_size, self.embed_dim, device=device)
        weighted_mean_sum = torch.zeros(batch_size, self.embed_dim, device=device)
        
        for i, name in enumerate(modality_order):
            if name not in means:
                continue
            
            mean = means[name]
            logvar = logvars[name]
            
            var = torch.exp(logvar).clamp(self.min_var, self.max_var)
            precision = 1.0 / var  # τ = 1/σ²
            
            if modality_mask is not None:
                mask = modality_mask[:, i:i+1]  # (batch, 1)
                precision = precision * mask
            
            precision_sum += precision
            weighted_mean_sum += precision * mean
        
        precision_sum = precision_sum.clamp(min=1e-8)
        
        fused_mean = weighted_mean_sum / precision_sum
        fused_var = 1.0 / precision_sum
        
        return fused_mean, fused_var


class MoEGatedFusion(nn.Module):
    """
    MoE-style gated fusion (ablation baseline).
    Uses a learned gate rather than explicit uncertainty weighting.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_modalities: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1),
        )
    
    def forward(
        self,
        means: Dict[str, torch.Tensor],
        modality_order: list,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            means: per-modality means
            modality_order: modality names in order
            modality_mask: availability mask
        
        Returns:
            fused: (batch, embed_dim)
        """
        embeddings = torch.stack([means[name] for name in modality_order], dim=1)
        # (batch, num_modalities, embed_dim)
        
        batch_size = embeddings.shape[0]
        
        flat = embeddings.view(batch_size, -1)
        weights = self.gate(flat)  # (batch, num_modalities)
        
        if modality_mask is not None:
            weights = weights * modality_mask
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        weights = weights.unsqueeze(-1)  # (batch, num_modalities, 1)
        fused = (embeddings * weights).sum(dim=1)  # (batch, embed_dim)
        
        return fused


class AverageFusion(nn.Module):
    """
    Simple equal-weight average fusion (ablation baseline).
    No uncertainty awareness; directly averages modality embeddings.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        means: Dict[str, torch.Tensor],
        logvars: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
        modality_order: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if modality_order is None:
            modality_order = list(means.keys())

        batch_size = next(iter(means.values())).shape[0]
        device = next(iter(means.values())).device

        sum_embed = torch.zeros(batch_size, self.embed_dim, device=device)
        count = torch.zeros(batch_size, 1, device=device)

        for i, name in enumerate(modality_order):
            if name not in means:
                continue
            embed = means[name]
            if modality_mask is not None:
                mask = modality_mask[:, i:i+1]
                embed = embed * mask
                count += mask
            else:
                sum_embed += embed
                count += 1
                continue
            sum_embed += embed

        count = count.clamp(min=1)
        fused_mean = sum_embed / count
        fused_var = torch.ones_like(fused_mean)

        return fused_mean, fused_var


class HybridPoEMoEFusion(nn.Module):
    """
    Hybrid fusion: PoE uncertainty weighting + learnable modality importance.

    Each modality m has a learnable scalar weight w_m.  A larger w_m increases
    the effective precision of modality m, amplifying its contribution to the
    fused representation.  Specifically, the effective logvar is shifted by
    -log(sigmoid(w_m)) before PoE aggregation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_modalities: int,
        use_learned_weight: bool = True,
    ):
        super().__init__()
        
        self.poe = PoEFusion(embed_dim)
        self.use_learned_weight = use_learned_weight
        
        if use_learned_weight:
            self.modality_weights = nn.Parameter(torch.ones(num_modalities))
    
    def forward(
        self,
        means: Dict[str, torch.Tensor],
        logvars: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
        modality_order: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if modality_order is None:
            modality_order = list(means.keys())
        
        if self.use_learned_weight:
            adjusted_logvars = {}
            for i, name in enumerate(modality_order):
                if name in logvars:
                    weight = torch.sigmoid(self.modality_weights[i])
                    adjusted_logvars[name] = logvars[name] - torch.log(weight + 1e-8)
                else:
                    adjusted_logvars[name] = logvars[name]
            logvars = adjusted_logvars
        
        return self.poe(means, logvars, modality_mask, modality_order)
