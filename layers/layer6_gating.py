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
        Compute gating weight α (text vs behavior).
        Returns: α ∈ [0.15, 0.85]
        """
        # Cold-start override
        if user_txn_count < 15:
            return max(0.7, text_confidence)
        
        # Prepare input features
        features = np.array([
            text_confidence,
            token_count / 10.0,  # Normalize
            1.0 if is_generic_text else 0.0,
            recurrence_confidence,
            cluster_density,
            min(user_txn_count / 100.0, 1.0),  # Normalize
            semantic_consensus,
            embedding_norm
        ], dtype=np.float32)
        
        # Forward pass
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            alpha = self.model(x).item()
        
        # Clip to range
        alpha = max(0.15, min(0.85, alpha))
        
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

