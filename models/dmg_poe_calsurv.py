"""
DMG-PoE-CalSurv: complete model.
Dynamic Modality Graph + uncertainty-aware PoE fusion + calibration-consistent survival prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .encoder import MultiModalityEncoder
from .dmg import DynamicModalityGraph
from .fusion import PoEFusion, HybridPoEMoEFusion, MoEGatedFusion, AverageFusion
from .surv_head import DiscreteHazardHead, NLLSurvivalLoss, CIndexMetric


class DMGPoECalSurv(nn.Module):
    """
    DMG-PoE-CalSurv complete model.
    
    Pipeline:
    1. MultiModalityEncoder: per-modality MLP encoder, outputs mean + logvar
    2. DynamicModalityGraph: modality-level GNN for complementary information propagation
    3. PoEFusion: uncertainty-weighted fusion
    4. DiscreteHazardHead: discrete-time hazard survival prediction
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 128,
        embed_dim: int = 64,
        num_time_bins: int = 20,
        dmg_layers: int = 2,
        dmg_heads: int = 4,
        dropout: float = 0.3,
        use_dmg: bool = True,
        use_poe: bool = True,
        fusion_type: str = 'hybrid_poe',
        imputation_type: str = 'dmg',
    ):
        """
        Args:
            fusion_type: 'hybrid_poe' | 'poe' | 'moe' | 'average' | 'concat'
            imputation_type: 'dmg' | 'zero' | 'mean'
        """
        super().__init__()

        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(modality_dims)
        self.embed_dim = embed_dim
        self.use_dmg = use_dmg
        self.use_poe = use_poe
        self.fusion_type = fusion_type
        self.imputation_type = imputation_type

        self.encoder = MultiModalityEncoder(
            modality_dims=modality_dims,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        if use_dmg:
            self.dmg = DynamicModalityGraph(
                embed_dim=embed_dim,
                num_layers=dmg_layers,
                num_heads=dmg_heads,
                ffn_dim=hidden_dim,
                dropout=dropout,
            )

        if fusion_type == 'hybrid_poe':
            self.fusion = HybridPoEMoEFusion(
                embed_dim=embed_dim,
                num_modalities=self.num_modalities,
            )
        elif fusion_type == 'poe':
            self.fusion = PoEFusion(embed_dim=embed_dim)
        elif fusion_type == 'moe':
            self.fusion_moe = MoEGatedFusion(
                embed_dim=embed_dim,
                num_modalities=self.num_modalities,
            )
        elif fusion_type == 'average':
            self.fusion = AverageFusion(embed_dim=embed_dim)
        else:
            # concat
            self.fusion_linear = nn.Sequential(
                nn.Linear(embed_dim * self.num_modalities, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        if fusion_type == 'concat':
            fusion_out_dim = hidden_dim
        elif fusion_type == 'moe':
            fusion_out_dim = embed_dim
        else:
            fusion_out_dim = embed_dim
        self.surv_head = DiscreteHazardHead(
            input_dim=fusion_out_dim,
            num_time_bins=num_time_bins,
            hidden_dim=hidden_dim // 2,
        )

        self.num_time_bins = num_time_bins
    
    def forward(
        self,
        features: torch.Tensor,
        modality_indices: Dict[str, Tuple[int, int]],
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (batch, total_features)
            modality_indices: {modality_name: (start_idx, end_idx)}
            modality_mask: (batch, num_modalities), 1=available, 0=missing
        
        Returns:
            dict with hazards, survival, risk_score, means, logvars, fused_var
        """
        means, logvars = self.encoder(features, modality_indices, modality_mask)

        if self.use_dmg:
            refined_means = self.dmg(means, modality_mask, self.modality_names)
        elif self.imputation_type == 'zero':
            refined_means = {}
            for i, name in enumerate(self.modality_names):
                if modality_mask is not None:
                    mask = modality_mask[:, i:i+1]
                    refined_means[name] = means[name] * mask
                else:
                    refined_means[name] = means[name]
        elif self.imputation_type == 'mean':
            refined_means = {}
            if modality_mask is not None:
                stacked = torch.stack([means[n] for n in self.modality_names], dim=1)
                mask_expanded = modality_mask.unsqueeze(-1)
                mean_embed = (stacked * mask_expanded).sum(dim=1, keepdim=True) / mask_expanded.sum(dim=1, keepdim=True).clamp(min=1)
                for i, name in enumerate(self.modality_names):
                    mask = modality_mask[:, i:i+1]
                    refined_means[name] = means[name] * mask + mean_embed.squeeze(1) * (1 - mask)
            else:
                refined_means = means
        else:
            refined_means = means

        fused_var = None
        if self.fusion_type == 'moe':
            fused = self.fusion_moe(refined_means, self.modality_names, modality_mask)
        elif self.fusion_type == 'concat':
            embeddings = torch.cat(
                [refined_means[name] for name in self.modality_names], dim=-1
            )
            fused = self.fusion_linear(embeddings)
        elif self.fusion_type in ('hybrid_poe', 'poe', 'average'):
            fused, fused_var = self.fusion(
                refined_means, logvars, modality_mask, self.modality_names
            )
        else:
            fused, fused_var = self.fusion(
                refined_means, logvars, modality_mask, self.modality_names
            )

        hazards = self.surv_head(fused)
        survival = self.surv_head.compute_survival(hazards)
        risk_score = -survival[:, -1]

        return {
            'hazards': hazards,
            'survival': survival,
            'risk_score': risk_score,
            'fused': fused,
            'fused_var': fused_var,
            'means': means,
            'logvars': logvars,
            'refined_means': refined_means if self.use_dmg else None,
        }


class CalibrationLoss(nn.Module):
    """Auxiliary Brier calibration and subset consistency losses."""
    
    def __init__(self, num_time_bins: int = 20):
        super().__init__()
        self.num_time_bins = num_time_bins
    
    def compute_brier_calibration(
        self,
        survival: torch.Tensor,  # (batch, num_time_bins)
        events: torch.Tensor,    # (batch,)
        times: torch.Tensor,     # (batch,)
        max_time: float,
    ) -> torch.Tensor:
        """
        Discretized Brier calibration regularizer over observed patient-time
        statuses.

        At time t, patients followed beyond t are known to be event-free and
        patients with an observed event by t are known to have experienced the
        event. Patients censored at or before t are excluded because their
        status at t is unknown. Reported evaluation IBS is computed separately
        with IPCW in metrics.py.
        """
        device = survival.device

        time_points = torch.linspace(0, 1, self.num_time_bins, device=device) * max_time

        brier_scores = []
        for t_idx in range(self.num_time_bins):
            t = time_points[t_idx]
            s_t = survival[:, t_idx]
            target = (times > t).to(dtype=s_t.dtype)
            known_status = (times > t) | ((events > 0.5) & (times <= t))

            if known_status.any():
                bs = (s_t[known_status] - target[known_status]) ** 2
                brier_scores.append(bs.mean())

        if not brier_scores:
            return survival.sum() * 0.0

        return torch.stack(brier_scores).mean()
    
    def compute_subset_consistency(
        self,
        model: nn.Module,
        features: torch.Tensor,
        modality_indices: Dict[str, Tuple[int, int]],
        num_masks: int = 2,
    ) -> torch.Tensor:
        """
        Subset consistency loss: predictions under different random modality subsets
        should remain close.
        """
        batch_size = features.shape[0]
        device = features.device
        num_modalities = len(modality_indices)
        
        masks = []
        outputs_list = []
        
        for _ in range(num_masks):
            mask = torch.rand(batch_size, num_modalities, device=device) > 0.3
            mask = mask.float()
            mask[:, 0] = 1.0
            masks.append(mask)
            
            with torch.no_grad():
                out = model(features, modality_indices, mask)
            outputs_list.append(out['survival'])
        
        consistency_loss = 0.0
        for i in range(num_masks):
            for j in range(i + 1, num_masks):
                diff = (outputs_list[i] - outputs_list[j]) ** 2
                consistency_loss += diff.mean()
        
        num_pairs = num_masks * (num_masks - 1) / 2
        return consistency_loss / max(num_pairs, 1)


class DMGPoECalSurvLoss(nn.Module):
    """
    Full loss: NLL + lambda_bc * L_BC + lambda_sc * L_SC.

    Brier calibration warmup (optional):
        The L_BC weight can be ramped up linearly over a warmup period so the model
        first learns basic risk structure before calibration is enforced.
        To enable, uncomment the body of update_lambda_bc and call it each epoch.

        Example:
        -----------------------------------------------------------------------
        # loss_fn = DMGPoECalSurvLoss(..., lambda_bc=0.1)
        # for epoch in range(total_epochs):
        #     loss_fn.update_lambda_bc(epoch, warmup_epochs=20)
        #     train_one_epoch(...)
        -----------------------------------------------------------------------
    """
    
    def __init__(
        self,
        num_time_bins: int = 20,
        lambda_bc: float = 0.1,
        lambda_sc: float = 0.1,
        use_calibration: bool = True,
    ):
        super().__init__()
        
        self.nll_loss = NLLSurvivalLoss(num_time_bins)
        self.cal_loss = CalibrationLoss(num_time_bins) if use_calibration else None
        self.lambda_bc = lambda_bc
        self._lambda_bc_base = lambda_bc
        self.lambda_sc = lambda_sc
        self.use_calibration = use_calibration

    def update_lambda_bc(self, epoch: int, warmup_epochs: int = 20) -> None:
        """
        Linear warmup for the Brier calibration weight (disabled by default).
        Uncomment the block below and call once per epoch to enable.
        """
        # if epoch < warmup_epochs:
        #     self.lambda_bc = self._lambda_bc_base * (epoch / warmup_epochs)
        # else:
        #     self.lambda_bc = self._lambda_bc_base
        pass
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        events: torch.Tensor,
        times: torch.Tensor,
        max_time: Optional[float] = None,
        model: Optional[nn.Module] = None,
        features: Optional[torch.Tensor] = None,
        modality_indices: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        if max_time is None:
            max_time = times.max().item()
        
        nll = self.nll_loss(outputs['hazards'], events, times, max_time)
        
        losses = {'nll': nll, 'total': nll}
        
        if self.use_calibration and self.cal_loss is not None:
            if self.lambda_bc > 0:
                bc = self.cal_loss.compute_brier_calibration(
                    outputs['survival'], events, times, max_time
                )
                losses['bc'] = bc
                losses['total'] = losses['total'] + self.lambda_bc * bc

            if self.lambda_sc > 0 and model is not None and features is not None and modality_indices is not None:
                sc = self.cal_loss.compute_subset_consistency(
                    model, features, modality_indices
                )
                losses['sc'] = sc
                losses['total'] = losses['total'] + self.lambda_sc * sc

        return losses


def create_dmg_poe_model(
    data: Dict,
    hidden_dim: int = 128,
    embed_dim: int = 64,
    num_time_bins: int = 20,
    use_dmg: bool = True,
    use_poe: bool = True,
    dmg_layers: int = 2,
    dmg_heads: int = 4,
    dropout: float = 0.3,
    fusion_type: str = 'hybrid_poe',
    imputation_type: str = 'dmg',
) -> DMGPoECalSurv:
    """Create model from a data dict produced by SurvBoardLoader."""
    modality_indices = data['modality_indices']
    modality_dims = {
        name: end - start
        for name, (start, end) in modality_indices.items()
    }

    return DMGPoECalSurv(
        modality_dims=modality_dims,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_time_bins=num_time_bins,
        dmg_layers=dmg_layers,
        dmg_heads=dmg_heads,
        dropout=dropout,
        use_dmg=use_dmg,
        use_poe=use_poe,
        fusion_type=fusion_type,
        imputation_type=imputation_type,
    )
