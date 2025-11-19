import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class ClassificationResult:
    category: str
    confidence: float
    reason: str
    provenance: Dict
    layer_used: str
    should_prompt: bool

class FinalClassifier:
    def __init__(self):
        self.confidence_thresholds = {
            'auto_label': 0.75,
            'probable': 0.50,
            'request_feedback': 0.0
        }
        self.categories = [
            'Food & Dining',
            'Commute/Transport',
            'Shopping',
            'Bills & Utilities',
            'Entertainment',
            'Healthcare',
            'Education',
            'Investments',
            'Salary/Income',
            'Transfers',
            'Subscriptions',
            'Others/Uncategorized'
        ]
    
    def classify(self, 
                 rule_result: Tuple[Optional[str], float, str],
                 semantic_result: Tuple[Optional[str], float, Dict],
                 behavioral_result: Tuple[Optional[str], float, Dict],
                 gating_alpha: float,
                 text_confidence: float,
                 behavior_confidence: float,
                 zeroshot_result: Tuple[Optional[str], float, Dict] = (None, 0.0, {})) -> ClassificationResult:
        """
        Final classification combining all layers.
        """
        # Layer 0: Rule-based (highest priority)
        if rule_result[0] is not None:
            return ClassificationResult(
                category=rule_result[0],
                confidence=rule_result[1],
                reason=rule_result[2],
                provenance={'layer': 'L0_rules', 'alpha': None},
                layer_used='L0: Rule-Based',
                should_prompt=False
            )
        
        # Layer 3: Semantic search
        semantic_category, semantic_conf, semantic_prov = semantic_result
        
        # Layer 5: Behavioral clustering
        behavioral_category, behavioral_conf, behavioral_prov = behavioral_result
        
        # Fuse confidences using gating
        if semantic_category and behavioral_category:
            # Both available - use gating
            final_conf = gating_alpha * semantic_conf + (1 - gating_alpha) * behavioral_conf
            
            # Choose category based on confidence
            if gating_alpha > 0.5:
                final_category = semantic_category
                reason = f"Gated fusion (α={gating_alpha:.2f}, favoring text): {semantic_prov.get('reason', '')}"
                layer = 'L3: Semantic (gated)'
            else:
                final_category = behavioral_category
                reason = f"Gated fusion (α={gating_alpha:.2f}, favoring behavior): {behavioral_prov.get('reason', '')}"
                layer = 'L5: Behavioral (gated)'
            
            provenance = {
                'layer': 'L6_gated_fusion',
                'alpha': gating_alpha,
                'text_confidence': text_confidence,
                'behavior_confidence': behavior_confidence,
                'semantic': semantic_prov,
                'behavioral': behavioral_prov
            }
        
        elif semantic_category:
            final_category = semantic_category
            final_conf = semantic_conf
            reason = semantic_prov.get('reason', 'Semantic match')
            layer = 'L3: Semantic Search'
            provenance = {'layer': 'L3_semantic', 'details': semantic_prov}
        
        elif behavioral_category:
            final_category = behavioral_category
            final_conf = behavioral_conf
            reason = behavioral_prov.get('reason', 'Behavioral pattern')
            layer = 'L5: Behavioral Clustering'
            provenance = {'layer': 'L5_behavioral', 'details': behavioral_prov}
        
        else:
            # Layer 8: Zero-shot fallback (if provided)
            zeroshot_category, zeroshot_conf, zeroshot_prov = zeroshot_result
            
            if zeroshot_category and zeroshot_conf >= 0.60:
                final_category = zeroshot_category
                final_conf = zeroshot_conf * 0.85  # Discount slightly
                reason = zeroshot_prov.get('reason', 'Zero-shot classification')
                layer = 'L8: Zero-Shot (BART-MNLI)'
                provenance = {'layer': 'L8_zeroshot', 'details': zeroshot_prov}
            else:
                # No classification possible
                final_category = 'Others/Uncategorized'
                final_conf = 0.0
                reason = 'No layer could classify transaction (including zero-shot)'
                layer = 'None'
                provenance = {'layer': 'none', 'reason': 'failed_all_layers', 'zeroshot_attempted': zeroshot_conf > 0}
        
        # Determine if user prompt needed
        should_prompt = final_conf < self.confidence_thresholds['auto_label']
        
        return ClassificationResult(
            category=final_category,
            confidence=final_conf,
            reason=reason,
            provenance=provenance,
            layer_used=layer,
            should_prompt=should_prompt
        )
    
    def to_dict(self, result: ClassificationResult) -> Dict:
        return {
            'category': result.category,
            'confidence': result.confidence,
            'reason': result.reason,
            'layer_used': result.layer_used,
            'should_prompt': result.should_prompt,
            'provenance': result.provenance
        }

