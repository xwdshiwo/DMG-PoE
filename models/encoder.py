"""
Modality encoders.
Each modality uses an independent MLP encoder that outputs mean and log-variance
for uncertainty estimation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class ModalityEncoder(nn.Module):
    """Single-modality encoder: MLP outputs mean and log-variance."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.3,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.mean_head = nn.Linear(in_dim, output_dim)
        self.logvar_head = nn.Linear(in_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            mean: (batch, output_dim)
            logvar: (batch, output_dim)
        """
        h = self.backbone(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar


class MultiModalityEncoder(nn.Module):
    """Multi-modality encoder: creates an independent encoder per modality."""
    
    def __init__(
        self,
        modality_dims: Dict[str, int],  # {modality_name: input_dim}
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.3,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.modality_names = list(modality_dims.keys())
        self.output_dim = output_dim
        
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(
                input_dim=dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                num_layers=num_layers,
            )
            for name, dim in modality_dims.items()
        })
        
        # Learnable mask token for missing modalities
        self.mask_tokens = nn.ParameterDict({
            name: nn.Parameter(torch.randn(output_dim))
            for name in modality_dims.keys()
        })
        
        # Default high log-variance for missing modalities (log(10) ≈ 2.3)
        self.default_logvar = 2.3
    
    def forward(
        self,
        features: torch.Tensor,
        modality_indices: Dict[str, Tuple[int, int]],
        modality_mask: Optional[torch.Tensor] = None,  # (batch, num_modalities)
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            features: (batch, total_features)
            modality_indices: {modality_name: (start_idx, end_idx)}
            modality_mask: (batch, num_modalities), 1=available, 0=missing
        
        Returns:
            means: {modality_name: (batch, output_dim)}
            logvars: {modality_name: (batch, output_dim)}
        """
        batch_size = features.shape[0]
        device = features.device
        
        means = {}
        logvars = {}
        
        for i, name in enumerate(self.modality_names):
            if name not in modality_indices:
                continue
                
            start, end = modality_indices[name]
            x = features[:, start:end]
            
            mean, logvar = self.encoders[name](x)
            
            if modality_mask is not None:
                mask = modality_mask[:, i:i+1]  # (batch, 1)
                mask_token = self.mask_tokens[name].unsqueeze(0).expand(batch_size, -1)
                mean = mean * mask + mask_token * (1 - mask)
                default_lv = torch.full_like(logvar, self.default_logvar)
                logvar = logvar * mask + default_lv * (1 - mask)
            
            means[name] = mean
            logvars[name] = logvar
        
        return means, logvars


class SimpleConcatFusion(nn.Module):
    """Simple concatenation fusion (baseline)."""
    
    def __init__(
        self,
        modality_names: list,
        embed_dim: int,
        output_dim: int,
    ):
        super().__init__()
        
        self.modality_names = modality_names
        total_dim = len(modality_names) * embed_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
    
    def forward(
        self,
        means: Dict[str, torch.Tensor],
        logvars: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            means: {modality_name: (batch, embed_dim)}
        Returns:
            fused: (batch, output_dim)
        """
        embeddings = [means[name] for name in self.modality_names if name in means]
        concat = torch.cat(embeddings, dim=-1)
        return self.fusion(concat)
