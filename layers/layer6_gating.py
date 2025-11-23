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
    def __init__(self, model_path: str = None, use_trained_model: bool = True):
        """
        Initialize gating controller.
        
        Args:
            model_path: Path to trained model weights
            use_trained_model: If True, use trained neural network; 
                             If False, use heuristic-based gating
        """
        self.model = GatingNetwork()
        self.use_trained_model = use_trained_model
        self.model_loaded = False
        
        if model_path and use_trained_model:
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                self.model_loaded = True
                print(f"✅ Loaded trained gating model from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load gating model: {e}")
                print("   Falling back to heuristic-based gating")
                self.use_trained_model = False
    
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
        
        Higher α = Trust text more
        Lower α = Trust behavior more
        
        Uses trained neural network if available, otherwise falls back to heuristics.
        """
        # Cold-start override - force text reliance when little history
        if user_txn_count < 15:
            return max(0.7, text_confidence)
        
        # Use trained model if available
        if self.use_trained_model and self.model_loaded:
            alpha = self._neural_alpha(
                text_confidence=text_confidence,
                token_count=token_count,
                is_generic_text=is_generic_text,
                recurrence_confidence=recurrence_confidence,
                cluster_density=cluster_density,
                user_txn_count=user_txn_count,
                semantic_consensus=semantic_consensus,
                embedding_norm=embedding_norm
            )
        else:
            # Fallback to heuristic-based alpha
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
    
    def _neural_alpha(self,
                     text_confidence: float,
                     token_count: int,
                     is_generic_text: bool,
                     recurrence_confidence: float,
                     cluster_density: float,
                     user_txn_count: int,
                     semantic_consensus: float,
                     embedding_norm: float) -> float:
        """
        Compute alpha using trained neural network.
        """
        # Prepare features
        features = torch.FloatTensor([
            text_confidence,
            recurrence_confidence,  # Swapped to match training order
            token_count / 10.0,  # Normalize
            1.0 if is_generic_text else 0.0,
            recurrence_confidence,
            cluster_density,
            min(user_txn_count / 100.0, 1.0),  # Normalize
            semantic_consensus
        ]).unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        with torch.no_grad():
            alpha = self.model(features).item()
        
        return alpha
    
    def _heuristic_alpha(self,
                         text_confidence: float,
                         token_count: int,
                         is_generic_text: bool,
                         recurrence_confidence: float,
                         cluster_density: float,
                         user_txn_count: int) -> float:
        """
        Improved heuristic for computing alpha - FAVORS TEXT/SENTIMENT MORE.
        
        Strategy:
        1. Prioritize text analysis for depth and semantic understanding
        2. Boost if text is clear (even moderately clear)
        3. Only reduce for VERY strong behavior patterns
        4. Text gets higher weight across all scenarios
        
        PHILOSOPHY: Text contains rich semantic information that captures intent,
        while behavior is useful but secondary. We want transactions to explore
        semantic layers deeply before falling back to behavioral patterns.
        """
        
        # REBALANCED: Give behavioral patterns more weight
        # Goal: Allow both text and behavior to contribute meaningfully
        base_alpha = min(0.75, text_confidence + 0.05)  # Reduced text boost from 0.10 to 0.05
        
        # Scenario 1: Very clear text (high confidence, many tokens, not generic)
        if text_confidence > 0.85 and token_count >= 4 and not is_generic_text:
            return 0.75  # Reduced from 0.85 - Allow behavior input
        
        # Scenario 2: Clear text but not perfect
        if text_confidence > 0.75 and not is_generic_text:
            return 0.65  # Reduced from 0.78 - More balanced
        
        # Scenario 3: Moderate text confidence with good context
        if text_confidence > 0.65 and token_count >= 4:
            return 0.60  # Reduced from 0.70 - Balanced
        
        # Scenario 4: Strong recurring behavior pattern - TRUST BEHAVIOR MORE
        if recurrence_confidence > 0.80 and cluster_density > 0.70:
            # Very strong behavior signal
            if is_generic_text or text_confidence < 0.55:
                return 0.30  # Reduced from 0.35 - Trust behavior heavily
            else:
                return 0.45  # Reduced from 0.50 - Behavior gets more weight
        
        # Scenario 5: Good behavior pattern - MORE BALANCED
        if recurrence_confidence > 0.65 and cluster_density > 0.55:
            if text_confidence < 0.65:
                return 0.40  # Reduced from 0.45 - Behavior-leaning
            else:
                return 0.55  # Reduced from 0.60 - More balanced
        
        # Scenario 6: Generic text - TRUST BEHAVIOR MORE
        if is_generic_text:
            if cluster_density > 0.50 or recurrence_confidence > 0.50:
                return 0.35  # Reduced from 0.40 - Behavior dominant
            else:
                return 0.50  # Reduced from 0.60 - Balanced
        
        # Scenario 7: Balanced - ALLOW BEHAVIOR MORE INFLUENCE
        if user_txn_count > 50:
            # More experience = favor behavior MORE
            behavior_boost = min(0.15, (user_txn_count - 50) / 400)  # Increased impact
            adjustment = -behavior_boost * (recurrence_confidence + cluster_density) / 2
        else:
            # Less experience = still somewhat balanced
            adjustment = 0.03  # Reduced from 0.08
        
        # Default: weighted combination with BALANCED weights
        alpha = (base_alpha * 0.55) + (1 - (recurrence_confidence + cluster_density) / 2) * 0.45 + adjustment
        
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
