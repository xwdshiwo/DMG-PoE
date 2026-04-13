"""
Survival analysis head (discrete hazard).
Outputs per-interval hazard rates from which the full survival curve is derived.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class DiscreteHazardHead(nn.Module):
    """
    Discrete hazard survival head.
    Outputs hazard probability for each time interval; survival function S(t) is
    computed as the cumulative product of (1 - h_k).
    """
    
    def __init__(
        self,
        input_dim: int,
        num_time_bins: int = 20,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.num_time_bins = num_time_bins
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_time_bins),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) fused representation
        Returns:
            hazards: (batch, num_time_bins) per-interval hazard probabilities
        """
        logits = self.head(x)
        hazards = torch.sigmoid(logits)
        return hazards
    
    def compute_survival(self, hazards: torch.Tensor) -> torch.Tensor:
        """
        S(t) = prod_{i=0}^{t-1} (1 - h_i)
        
        Args:
            hazards: (batch, num_time_bins)
        Returns:
            survival: (batch, num_time_bins)
        """
        survival = torch.cumprod(1 - hazards, dim=-1)
        return survival
    
    def compute_pmf(self, hazards: torch.Tensor) -> torch.Tensor:
        """
        Event probability mass function: f(t) = h_t * S(t-1)
        
        Args:
            hazards: (batch, num_time_bins)
        Returns:
            pmf: (batch, num_time_bins)
        """
        survival = self.compute_survival(hazards)
        survival_prev = F.pad(survival[:, :-1], (1, 0), value=1.0)
        pmf = hazards * survival_prev
        return pmf


class NLLSurvivalLoss(nn.Module):
    """Negative log-likelihood loss for discrete-time survival analysis."""
    
    def __init__(self, num_time_bins: int = 20):
        super().__init__()
        self.num_time_bins = num_time_bins
    
    def discretize_time(
        self, 
        times: torch.Tensor, 
        max_time: float,
    ) -> torch.Tensor:
        """Map continuous times to discrete bin indices."""
        bins = (times / max_time * (self.num_time_bins - 1)).clamp(0, self.num_time_bins - 1)
        return bins.long()
    
    def forward(
        self,
        hazards: torch.Tensor,  # (batch, num_time_bins)
        events: torch.Tensor,   # (batch,) 0=censored, 1=event
        times: torch.Tensor,    # (batch,)
        max_time: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            hazards: predicted hazard probabilities
            events: event indicator (1=event occurred, 0=censored)
            times: observed times
            max_time: maximum time for discretisation
        Returns:
            loss: scalar
        """
        if max_time is None:
            max_time = times.max().item()
        
        batch_size = hazards.shape[0]
        device = hazards.device
        
        time_bins = self.discretize_time(times, max_time)
        
        survival = torch.cumprod(1 - hazards, dim=-1)
        survival_prev = F.pad(survival[:, :-1], (1, 0), value=1.0)
        
        eps = 1e-7
        loss = 0.0
        
        for i in range(batch_size):
            t = time_bins[i]
            e = events[i]
            
            if e == 1:
                log_h = torch.log(hazards[i, t] + eps)
                log_s_prev = torch.log(survival_prev[i, t] + eps)
                loss -= (log_h + log_s_prev)
            else:
                log_s = torch.log(survival[i, t] + eps)
                loss -= log_s
        
        return loss / batch_size


class CIndexMetric:
    """Harrell's concordance index."""
    
    @staticmethod
    def compute(
        risk_scores: torch.Tensor,  # (N,) higher score = higher risk
        events: torch.Tensor,       # (N,)
        times: torch.Tensor,        # (N,)
    ) -> float:
        risk = risk_scores.detach().cpu().numpy()
        event = events.detach().cpu().numpy()
        time = times.detach().cpu().numpy()
        
        n = len(risk)
        concordant = 0
        permissible = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if time[i] == time[j]:
                    continue
                
                if event[i] == 1 and time[i] < time[j]:
                    permissible += 1
                    if risk[i] > risk[j]:
                        concordant += 1
                    elif risk[i] == risk[j]:
                        concordant += 0.5
                elif event[j] == 1 and time[j] < time[i]:
                    permissible += 1
                    if risk[j] > risk[i]:
                        concordant += 1
                    elif risk[i] == risk[j]:
                        concordant += 0.5
        
        if permissible == 0:
            return 0.5
        
        return concordant / permissible
