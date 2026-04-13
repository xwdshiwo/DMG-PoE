"""
Survival analysis evaluation metrics aligned with the SurvBoard benchmark.
Includes: Antolini's C-index, IBS, D-Calibration.
"""

import numpy as np
from typing import Tuple, Optional
import warnings


def antolini_c_index(
    risk_scores: np.ndarray,
    events: np.ndarray, 
    times: np.ndarray
) -> float:
    """
    Antolini's time-dependent C-index.
    
    Args:
        risk_scores: predicted risk scores (higher = higher risk)
        events: event indicator (1=event, 0=censored)
        times: observed times
    
    Returns:
        C-index (0.5=random, 1.0=perfect)
    """
    n = len(risk_scores)
    concordant = 0
    permissible = 0
    
    for i in range(n):
        for j in range(n):
            if times[i] < times[j] and events[i] == 1:
                permissible += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5
    
    if permissible == 0:
        return 0.5
    
    return concordant / permissible


def kaplan_meier_estimator(
    events: np.ndarray,
    times: np.ndarray,
    time_points: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kaplan-Meier survival function estimator.
    
    Returns:
        unique_times: unique time points
        survival_probs: corresponding survival probabilities
    """
    unique_times = np.sort(np.unique(times))
    n_at_risk = len(times)
    survival_prob = 1.0
    survival_probs = []
    
    for t in unique_times:
        n_events = np.sum((times == t) & (events == 1))
        n_censored = np.sum((times == t) & (events == 0))
        
        if n_at_risk > 0:
            survival_prob *= (1 - n_events / n_at_risk)
        
        survival_probs.append(survival_prob)
        n_at_risk -= (n_events + n_censored)
    
    return unique_times, np.array(survival_probs)


def integrated_brier_score(
    survival_probs: np.ndarray,  # shape: (n_samples, n_time_points)
    time_points: np.ndarray,     # shape: (n_time_points,)
    events: np.ndarray,          # shape: (n_samples,)
    times: np.ndarray,           # shape: (n_samples,)
    max_time: Optional[float] = None
) -> float:
    """
    Integrated Brier Score (IBS) with IPCW weighting.
    
    Args:
        survival_probs: predicted survival probability matrix S(t|x)
        time_points: evaluation time points
        events: event indicator
        times: observed times
        max_time: maximum evaluation time (defaults to max observed time)
    
    Returns:
        IBS value (lower is better)
    """
    n_samples = len(events)
    
    if max_time is None:
        max_time = times.max()
    
    km_times, km_surv = kaplan_meier_estimator(1 - events, times)
    
    def get_censoring_prob(t):
        idx = np.searchsorted(km_times, t, side='right') - 1
        if idx < 0:
            return 1.0
        return max(km_surv[idx], 1e-8)
    
    brier_scores = []
    valid_time_points = time_points[time_points <= max_time]
    
    for t_idx, t in enumerate(valid_time_points):
        bs_t = 0.0
        n_valid = 0
        
        for i in range(n_samples):
            if times[i] <= t and events[i] == 1:
                actual = 0
                weight = 1.0 / get_censoring_prob(times[i])
            elif times[i] > t:
                actual = 1
                weight = 1.0 / get_censoring_prob(t)
            else:
                continue
            
            pred_surv = survival_probs[i, t_idx] if t_idx < survival_probs.shape[1] else survival_probs[i, -1]
            
            bs_t += weight * (pred_surv - actual) ** 2
            n_valid += 1
        
        if n_valid > 0:
            brier_scores.append(bs_t / n_valid)
    
    if len(brier_scores) < 2:
        return 0.0
    
    valid_times = valid_time_points[:len(brier_scores)]
    ibs = np.trapz(brier_scores, valid_times) / (valid_times[-1] - valid_times[0])
    
    return ibs


def d_calibration(
    survival_probs: np.ndarray,  # shape: (n_samples, n_time_points) 
    time_points: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float]:
    """
    D-Calibration test for survival probability calibration.
    
    Args:
        survival_probs: predicted survival probability matrix
        time_points: evaluation time points
        events: event indicator
        times: observed times
        n_bins: number of bins for the chi-squared test
    
    Returns:
        d_cal: p-value (higher is better; >0.05 indicates good calibration)
        chi2: chi-squared statistic
    """
    from scipy import stats
    
    n_samples = len(events)
    
    mid_idx = len(time_points) // 2
    mid_time = time_points[mid_idx]
    
    pred_probs = survival_probs[:, mid_idx]
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(pred_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    observed = np.zeros(n_bins)
    expected = np.zeros(n_bins)
    
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        
        bin_events = events[mask]
        bin_times = times[mask]
        bin_probs = pred_probs[mask]
        
        expected[b] = bin_probs.sum()
        observed[b] = np.sum(bin_times > mid_time)
    
    valid = (expected > 0) & (observed >= 0)
    if valid.sum() < 2:
        return 1.0, 0.0
    
    observed = observed[valid]
    expected = expected[valid]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi2 = np.sum((observed - expected) ** 2 / np.maximum(expected, 1e-8))
        df = len(observed) - 1
        p_value = 1 - stats.chi2.cdf(chi2, df) if df > 0 else 1.0
    
    return p_value, chi2


class SurvivalMetrics:
    """Wrapper for computing all survival analysis metrics."""
    
    def __init__(self, time_points: np.ndarray):
        """
        Args:
            time_points: evaluation time points for discretising the survival function
        """
        self.time_points = time_points
    
    def compute_all(
        self,
        hazards: np.ndarray,      # shape: (n_samples, n_time_bins)
        events: np.ndarray,
        times: np.ndarray,
        risk_scores: Optional[np.ndarray] = None
    ) -> dict:
        """
        Compute all metrics.
        
        Args:
            hazards: predicted discrete hazard rates
            events: event indicator
            times: observed times
            risk_scores: risk scores (computed from hazards if None)
        
        Returns:
            dict with c_index, ibs, d_cal, d_cal_chi2, survival_probs
        """
        survival_probs = self._hazards_to_survival(hazards)
        
        if risk_scores is None:
            risk_scores = hazards.sum(axis=1)
        
        c_index = antolini_c_index(risk_scores, events, times)
        
        ibs = integrated_brier_score(
            survival_probs, self.time_points, events, times
        )
        
        d_cal, chi2 = d_calibration(
            survival_probs, self.time_points, events, times
        )
        
        return {
            'c_index': c_index,
            'ibs': ibs,
            'd_cal': d_cal,
            'd_cal_chi2': chi2,
            'survival_probs': survival_probs
        }
    
    def _hazards_to_survival(self, hazards: np.ndarray) -> np.ndarray:
        """Convert discrete hazard rates to survival probabilities."""
        survival = np.cumprod(1 - hazards, axis=1)
        return survival


def compute_survival_metrics(
    hazards: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    time_points: np.ndarray,
    risk_scores: Optional[np.ndarray] = None
) -> dict:
    """
    Convenience function to compute all survival metrics in one call.
    
    Returns:
        {
            'c_index': float,  # Antolini's C (higher is better)
            'ibs': float,      # Integrated Brier Score (lower is better)
            'd_cal': float,    # D-calibration p-value (higher is better; >0.05 = well-calibrated)
        }
    """
    metrics = SurvivalMetrics(time_points)
    results = metrics.compute_all(hazards, events, times, risk_scores)
    
    return {
        'c_index': results['c_index'],
        'ibs': results['ibs'],
        'd_cal': results['d_cal']
    }
