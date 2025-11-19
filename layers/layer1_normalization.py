import re
from typing import Dict, Tuple
from rapidfuzz import fuzz

class TextNormalizer:
    def __init__(self):
        self.noise_patterns = [
            r'upi', r'neft', r'imps', r'ref', r'txn', r'auth', 
            r'payment', r'\d{6,}'
        ]
        self.canonical_aliases = {
            'swiggy': ['swiggy', 'swigy', 'swgy'],
            'zomato': ['zomato', 'zmato', 'zomto'],
            'uber': ['uber', 'ubr'],
            'ola': ['ola', 'olacabs'],
            'rapido': ['rapido', 'rpdo'],
            'metro': ['metro', 'metrorail', 'dmrc'],
            'netflix': ['netflix', 'ntflx'],
            'spotify': ['spotify', 'sptfy'],
            'amazon': ['amazon', 'amzn', 'amazonpay'],
            'flipkart': ['flipkart', 'flpkrt', 'fkrt']
        }
        self.category_map = {
            'swiggy': 'Food & Dining',
            'zomato': 'Food & Dining',
            'uber': 'Commute/Transport',
            'ola': 'Commute/Transport',
            'rapido': 'Commute/Transport',
            'metro': 'Commute/Transport',
            'netflix': 'Subscriptions',
            'spotify': 'Subscriptions',
            'amazon': 'Shopping',
            'flipkart': 'Shopping'
        }
    
    def normalize(self, text: str) -> Tuple[str, Dict]:
        """
        Normalize merchant/description text.
        Returns: (normalized_text, metadata)
        """
        if not text or pd.isna(text):
            return '', {'original_length': 0, 'vpa_extracted': None}
        
        original = text
        text = text.lower().strip()
        
        # Extract VPA if present
        vpa = self._extract_vpa(text)
        
        # Remove noise patterns
        for pattern in self.noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Keep only alphanumeric and @ . characters
        text = re.sub(r'[^a-z0-9@\.]', ' ', text)
        
        # Keep tokens > 2 chars
        tokens = [t for t in text.split() if len(t) > 2]
        text = ' '.join(tokens)
        
        # Fuzzy match against canonical aliases
        canonical, confidence = self._match_canonical(text)
        
        metadata = {
            'original_length': len(original),
            'vpa_extracted': vpa,
            'canonical_match': canonical,
            'canonical_confidence': confidence,
            'token_count': len(tokens)
        }
        
        return text if not canonical else canonical, metadata
    
    def _extract_vpa(self, text: str) -> str:
        vpa_pattern = r'\w+@[\w\.]+'
        match = re.search(vpa_pattern, text)
        return match.group(0) if match else None
    
    def _match_canonical(self, text: str) -> Tuple[str, float]:
        best_match = None
        best_score = 0
        
        for canonical, aliases in self.canonical_aliases.items():
            for alias in aliases:
                score = fuzz.partial_ratio(text, alias)
                if score > best_score:
                    best_score = score
                    best_match = canonical
        
        if best_score >= 90:
            return best_match, best_score / 100.0
        elif best_score >= 75:
            return best_match, best_score / 100.0
        return None, 0.0

import pandas as pd

