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
        # ADJUSTED: Lower auto-label threshold to allow more layer exploration
        # Goal: Let transactions travel through more layers before "settling"
        # This ensures deeper semantic and behavioral analysis
        self.confidence_thresholds = {
            'auto_label': 0.60,  # Lowered from 0.70 - more exploration
            'probable': 0.45,    # Lowered from 0.50 - wider "uncertain" band
            'request_feedback': 0.0
        }
        # Fixed categories - ONLY these are allowed
        self.fixed_categories = [
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
        REDESIGNED: Always use gating when multiple layers have results.
        Layer 0 is now just another input, not a hard override.
        """
        # Extract all layer results
        rule_category, rule_conf, rule_reason = rule_result
        semantic_category, semantic_conf, semantic_prov = semantic_result
        behavioral_category, behavioral_conf, behavioral_prov = behavioral_result
        
        # Count how many layers have results
        available_layers = []
        if rule_category: available_layers.append('rule')
        if semantic_category: available_layers.append('semantic')
        if behavioral_category: available_layers.append('behavioral')
        
        # STRATEGY 1: Use gating when we have semantic AND behavioral (preferred)
        # This ensures maximum layer participation
        if semantic_category and behavioral_category:
            # Both semantic and behavioral available - use gating
            final_conf = gating_alpha * semantic_conf + (1 - gating_alpha) * behavioral_conf
            
            # If rule also matched, blend it in with lower weight
            if rule_category:
                # Give rule result 20% weight, gated result 80% weight
                final_conf = 0.20 * rule_conf + 0.80 * final_conf
                
                # Choose category based on highest individual confidence
                candidates = [
                    (semantic_category, semantic_conf),
                    (behavioral_category, behavioral_conf),
                    (rule_category, rule_conf)
                ]
                final_category = max(candidates, key=lambda x: x[1])[0]
                
                reason = f"Multi-layer fusion (Rule+Semantic+Behavioral, α={gating_alpha:.2f})"
                layer = 'L6: Gated Fusion (3 layers)'
            else:
                # Choose category based on gating weight
                if gating_alpha > 0.5:
                    final_category = semantic_category
                    reason = f"Gated fusion (α={gating_alpha:.2f}, favoring text): {semantic_prov.get('reason', '')}"
                    layer = 'L6: Gated Fusion (Semantic)'
                else:
                    final_category = behavioral_category
                    reason = f"Gated fusion (α={gating_alpha:.2f}, favoring behavior): {behavioral_prov.get('reason', '')}"
                    layer = 'L6: Gated Fusion (Behavioral)'
            
            provenance = {
                'layer': 'L6_gated_fusion',
                'alpha': gating_alpha,
                'text_confidence': text_confidence,
                'behavior_confidence': behavior_confidence,
                'rule_confidence': rule_conf if rule_category else 0.0,
                'semantic': semantic_prov,
                'behavioral': behavioral_prov,
                'rule': {'category': rule_category, 'reason': rule_reason} if rule_category else None
            }
        
        # STRATEGY 2: Rule + Semantic (no behavioral)
        elif rule_category and semantic_category:
            # Blend rule and semantic
            final_conf = 0.30 * rule_conf + 0.70 * semantic_conf
            final_category = semantic_category if semantic_conf > rule_conf else rule_category
            reason = f"Rule+Semantic fusion: {rule_reason} + {semantic_prov.get('reason', '')}"
            layer = 'L3: Semantic + L0: Rule'
            provenance = {
                'layer': 'L0_L3_fusion',
                'rule': {'category': rule_category, 'confidence': rule_conf, 'reason': rule_reason},
                'semantic': semantic_prov
            }
        
        # STRATEGY 3: Rule + Behavioral (no semantic)
        elif rule_category and behavioral_category:
            # Blend rule and behavioral
            final_conf = 0.30 * rule_conf + 0.70 * behavioral_conf
            final_category = behavioral_category if behavioral_conf > rule_conf else rule_category
            reason = f"Rule+Behavioral fusion: {rule_reason} + {behavioral_prov.get('reason', '')}"
            layer = 'L5: Behavioral + L0: Rule'
            provenance = {
                'layer': 'L0_L5_fusion',
                'rule': {'category': rule_category, 'confidence': rule_conf, 'reason': rule_reason},
                'behavioral': behavioral_prov
            }
        
        # STRATEGY 4: Only one layer has results (fallback to single layer)
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
        
        elif rule_category:
            final_category = rule_category
            final_conf = rule_conf
            reason = rule_reason
            layer = 'L0: Rule-Based'
            provenance = {'layer': 'L0_rules', 'reason': rule_reason}
        
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
        
        # Validate final category
        final_category = self._validate_category(final_category)
        
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
    
    def _validate_category(self, category: str) -> str:
        """Ensure category is in fixed list."""
        if category in self.fixed_categories:
            return category
        
        # Try to map similar categories
        category_lower = category.lower()
        for fixed_cat in self.fixed_categories:
            if category_lower in fixed_cat.lower() or fixed_cat.lower() in category_lower:
                return fixed_cat
        
        # Default to Others
        return 'Others/Uncategorized'
    
    def to_dict(self, result: ClassificationResult) -> Dict:
        return {
            'category': result.category,
            'confidence': result.confidence,
            'reason': result.reason,
            'layer_used': result.layer_used,
            'should_prompt': result.should_prompt,
            'provenance': result.provenance
        }

