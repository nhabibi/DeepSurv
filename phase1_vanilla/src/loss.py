"""
Cox Proportional Hazards loss function.
"""

import torch
import torch.nn as nn


# ============================================================================
# Cox Proportional Hazards Loss
# ============================================================================

class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards loss.
    Negative log partial likelihood.
    
    Args:
        method: 'breslow' or 'efron' for tied event times
    """
    
    def __init__(self, method: str = 'efron'):
        super(CoxPHLoss, self).__init__()
        if method not in ['breslow', 'efron']:
            raise ValueError("method must be 'breslow' or 'efron'")
        self.method = method
    
    # ------------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------------
    
    def forward(
        self,
        risk_scores: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log partial likelihood.
        
        Args:
            risk_scores: Predicted risk scores [batch_size]
            times: Survival times [batch_size]
            events: Event indicators [batch_size]
        
        Returns:
            Loss value
        """
        # --------------------------------------------------------------------
        # Sort by descending time
        # --------------------------------------------------------------------
        sorted_indices = torch.argsort(times, descending=True)
        risk_scores = risk_scores[sorted_indices]
        times = times[sorted_indices]
        events = events[sorted_indices]
        
        if self.method == 'breslow':
            return self._breslow_loss(risk_scores, events)
        else:
            return self._efron_loss(risk_scores, events, times)
    
    # ------------------------------------------------------------------------
    # Breslow Method
    # ------------------------------------------------------------------------
    
    def _breslow_loss(self, log_risk: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """Breslow method for handling tied event times."""
        num_events = torch.sum(events)
        if num_events == 0:
            return torch.tensor(0.0)
        
        # Compute risk and cumulative sum (from right to left)
        risk = torch.exp(log_risk)
        cumsum_risk = torch.cumsum(risk.flip(0), dim=0).flip(0)
        
        # Log partial likelihood
        log_likelihood = log_risk - torch.log(cumsum_risk + 1e-7)
        
        # Negative average log likelihood for events only
        loss = -torch.sum(log_likelihood * events) / num_events
        return loss
    
    # ------------------------------------------------------------------------
    # Efron Method
    # ------------------------------------------------------------------------
    
    def _efron_loss(self, log_risk: torch.Tensor, events: torch.Tensor, 
                    times: torch.Tensor) -> torch.Tensor:
        """Efron method for handling tied event times (more accurate)."""
        risk = torch.exp(log_risk)
        cumsum_risk = torch.cumsum(risk.flip(0), dim=0).flip(0)
        unique_times = torch.unique(times[events == 1])
        
        total_loss = 0.0
        total_events = 0
        
        # Process each unique event time
        for event_time in unique_times:
            # Find events at this time
            is_event_at_time = (times == event_time) & (events == 1)
            if not is_event_at_time.any():
                continue
            
            # Get event indices and counts
            event_idx = torch.where(is_event_at_time)[0]
            n_events = len(event_idx)
            
            # Sum of log risks for events at this time
            log_risk_sum = torch.sum(log_risk[is_event_at_time])
            
            # Efron approximation for tied events
            risk_at_time = torch.sum(risk[is_event_at_time])
            risk_set = cumsum_risk[event_idx[0]]
            
            for j in range(n_events):
                correction = (j / n_events) * risk_at_time
                total_loss += torch.log(risk_set - correction + 1e-7)
            
            total_loss -= log_risk_sum
            total_events += n_events
        
        return total_loss / total_events if total_events > 0 else torch.tensor(0.0)
