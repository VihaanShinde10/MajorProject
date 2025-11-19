import re
from typing import Dict, Tuple, Optional
from rapidfuzz import fuzz
import pandas as pd

class TextNormalizer:
    def __init__(self):
        self.noise_patterns = [
            r'upi', r'neft', r'imps', r'ref', r'txn', r'auth', 
            r'payment', r'paytm', r'\d{6,}'
        ]
        self.canonical_aliases = {
            'swiggy': ['swiggy', 'swigy', 'swgy'],
            'zomato': ['zomato', 'zmato', 'zomto'],
            'uber': ['uber', 'ubr'],
            'ola': ['ola', 'olacabs'],
            'rapido': ['rapido', 'rpdo'],
            'metro': ['metro', 'metrorail', 'dmrc'],
            'netflix': ['netflix', 'ntflx', 'netflixupi'],
            'spotify': ['spotify', 'sptfy'],
            'amazon': ['amazon', 'amzn', 'amazonpay'],
            'flipkart': ['flipkart', 'flpkrt', 'fkrt'],
            'bhart': ['bhartia', 'bhart'],
            'imagicaa': ['imagicaa', 'imagica'],
            'vinayak': ['vinayak'],
            'anushka': ['anushka', 'anushkashe'],
            'shubham': ['shubham', 'shubhamraj'],
            'venkates': ['venkates', 'venkatesha'],
            'harshba': ['harshba', 'harshbala'],
            'alphavi': ['alphavi', 'alpharane'],
            'mayabha': ['mayabha', 'mayabhanu'],
            'indianr': ['indianr', 'indian railways'],
            'snowcre': ['snowcre', 'snowcreat'],
            'julfikar': ['julfikar', 'julfika'],
            'bikaner': ['bikaner', 'bikanervala']
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
            'flipkart': 'Shopping',
            'imagicaa': 'Entertainment',
            'indianr': 'Commute/Transport',
            'bikaner': 'Food & Dining'
        }
    
    def normalize(self, text: str, recipient_name: str = None, upi_id: str = None, 
                  note: str = None) -> Tuple[str, Dict]:
        """
        Normalize merchant/description text with enhanced UPI-based data.
        Returns: (normalized_text, metadata)
        
        Args:
            text: Primary description/merchant field
            recipient_name: UPI recipient name (e.g., "JULFIKAR", "VINAYAK")
            upi_id: UPI ID (e.g., "paytmqr1jc", "vinayakpbh")
            note: Transaction note (e.g., "baker", "UPI", "Payme")
        """
        if not text or pd.isna(text):
            text = ''
        
        # Collect all available text sources
        original_text = text
        recipient = str(recipient_name).lower() if recipient_name and not pd.isna(recipient_name) else ''
        upi = str(upi_id).lower() if upi_id and not pd.isna(upi_id) else ''
        note_text = str(note).lower() if note and not pd.isna(note) else ''
        
        # Combine all sources for richer context
        combined_text = f"{text} {recipient} {note_text}".strip()
        text = combined_text.lower().strip()
        
        # Extract VPA/UPI ID
        vpa = self._extract_vpa(text) or upi
        
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
        
        # Enhanced: Check recipient name directly
        if not canonical and recipient:
            recipient_canonical, recipient_conf = self._match_canonical(recipient)
            if recipient_canonical and recipient_conf > confidence:
                canonical = recipient_canonical
                confidence = recipient_conf
        
        # Enhanced: Check note field
        if not canonical and note_text:
            note_canonical, note_conf = self._match_canonical(note_text)
            if note_canonical and note_conf > confidence:
                canonical = note_canonical
                confidence = note_conf
        
        metadata = {
            'original_length': len(original_text),
            'vpa_extracted': vpa,
            'canonical_match': canonical,
            'canonical_confidence': confidence,
            'token_count': len(tokens),
            'recipient_name': recipient_name,
            'upi_id': upi_id,
            'note': note_text,
            'has_recipient': bool(recipient),
            'has_upi_id': bool(upi),
            'has_note': bool(note_text)
        }
        
        return text if not canonical else canonical, metadata
    
    def _extract_vpa(self, text: str) -> Optional[str]:
        """Extract UPI VPA (Virtual Payment Address)."""
        vpa_pattern = r'\w+@[\w\.]+'
        match = re.search(vpa_pattern, text)
        return match.group(0) if match else None
    
    def _match_canonical(self, text: str) -> Tuple[Optional[str], float]:
        """Match text against canonical merchant names."""
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
