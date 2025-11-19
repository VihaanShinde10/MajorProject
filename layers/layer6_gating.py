import torch
import torch.nn as nn
import numpy as np
from typing import Dict

class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class GatingController:
    def __init__(self, model_path: str = None):
        self.model = GatingNetwork()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def compute_alpha(self, 
                      text_confidence: float,
                      token_count: int,
                      is_generic_text: bool,
                      recurrence_confidence: float,
                      cluster_density: float,
                      user_txn_count: int,
                      semantic_consensus: float = 0.0,
                      embedding_norm: float = 1.0) -> float:
        """
        Compute gating weight α (text vs behavior) with improved heuristics.
        Returns: α ∈ [0.15, 0.85]
        
        Higher α = Trust text more
        Lower α = Trust behavior more
        """
        # Cold-start override - force text reliance when little history
        if user_txn_count < 15:
            return max(0.7, text_confidence)
        
        # Use improved heuristic-based alpha until network is trained
        alpha = self._heuristic_alpha(
            text_confidence=text_confidence,
            token_count=token_count,
            is_generic_text=is_generic_text,
            recurrence_confidence=recurrence_confidence,
            cluster_density=cluster_density,
            user_txn_count=user_txn_count
        )
        
        # Clip to range
        alpha = max(0.15, min(0.85, alpha))
        
        return alpha
    
    def _heuristic_alpha(self,
                         text_confidence: float,
                         token_count: int,
                         is_generic_text: bool,
                         recurrence_confidence: float,
                         cluster_density: float,
                         user_txn_count: int) -> float:
        """
        Improved heuristic for computing alpha.
        
        Strategy:
        1. Start with base alpha from text confidence
        2. Boost if text is very clear
        3. Reduce if behavior patterns are very strong
        4. Adjust based on experience (more transactions = more trust in behavior)
        """
        
        # Base alpha from text confidence
        base_alpha = text_confidence
        
        # Scenario 1: Very clear text (high confidence, many tokens, not generic)
        if text_confidence > 0.85 and token_count >= 3 and not is_generic_text:
            return 0.80  # Trust text heavily
        
        # Scenario 2: Clear text but not perfect
        if text_confidence > 0.75 and not is_generic_text:
            return 0.70  # Still favor text
        
        # Scenario 3: Strong recurring behavior pattern (subscription-like)
        if recurrence_confidence > 0.85 and cluster_density > 0.75:
            # Very strong behavior signal
            if is_generic_text or text_confidence < 0.6:
                return 0.25  # Trust behavior heavily
            else:
                return 0.40  # Moderate behavior preference
        
        # Scenario 4: Good behavior pattern but decent text
        if recurrence_confidence > 0.70 and cluster_density > 0.60:
            if text_confidence < 0.70:
                return 0.35  # Favor behavior
            else:
                return 0.50  # Balanced
        
        # Scenario 5: Generic text (force behavior if available)
        if is_generic_text:
            if cluster_density > 0.50 or recurrence_confidence > 0.50:
                return 0.30  # Trust behavior
            else:
                return 0.55  # Slightly favor text (no good alternative)
        
        # Scenario 6: Balanced - adjust based on experience
        if user_txn_count > 50:
            # More experience = slightly favor behavior
            behavior_boost = min(0.15, (user_txn_count - 50) / 500)
            adjustment = -behavior_boost * (recurrence_confidence + cluster_density) / 2
        else:
            # Less experience = slightly favor text
            adjustment = 0.05
        
        # Default: weighted combination
        alpha = (base_alpha * 0.6) + (1 - (recurrence_confidence + cluster_density) / 2) * 0.4 + adjustment
        
        return alpha
    
    def fuse_confidence(self, 
                       text_confidence: float,
                       behavior_confidence: float,
                       alpha: float) -> float:
        """
        Fuse text and behavior confidence using gating weight.
        final_confidence = α × text_confidence + (1-α) × behavior_confidence
        """
        return alpha * text_confidence + (1 - alpha) * behavior_confidence
