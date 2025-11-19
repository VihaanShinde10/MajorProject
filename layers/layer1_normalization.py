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
            # 'bhart': ['bhartia', 'bhart'],
            'imagicaa': ['imagicaa', 'imagica'],
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
        ROBUST: Prioritizes recipient_name, upi_id, and note fields FIRST.
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
        
        # Clean empty values
        recipient = recipient if recipient not in ['', 'nan', 'none'] else ''
        upi = upi if upi not in ['', 'nan', 'none'] else ''
        note_text = note_text if note_text not in ['', 'nan', 'none'] else ''
        
        # ROBUST STRATEGY: Check UPI fields FIRST (they're more reliable)
        canonical = None
        confidence = 0.0
        matched_source = ''
        
        # Priority 1: Check RECIPIENT_NAME first (most reliable)
        if recipient:
            recipient_canonical, recipient_conf = self._match_canonical(recipient)
            if recipient_canonical and recipient_conf > confidence:
                canonical = recipient_canonical
                confidence = recipient_conf
                matched_source = 'recipient_name'
        
        # Priority 2: Check UPI_ID (contains merchant identifiers)
        if upi and confidence < 0.90:  # Only check if we don't have strong match
            upi_canonical, upi_conf = self._match_canonical(upi)
            if upi_canonical and upi_conf > confidence:
                canonical = upi_canonical
                confidence = upi_conf
                matched_source = 'upi_id'
        
        # Priority 3: Check NOTE field
        if note_text and confidence < 0.90:
            note_canonical, note_conf = self._match_canonical(note_text)
            if note_canonical and note_conf > confidence:
                canonical = note_canonical
                confidence = note_conf
                matched_source = 'note'
        
        # Priority 4: Check combined text (fallback)
        if not canonical or confidence < 0.75:
            # Combine sources with UPI fields FIRST
            combined_text = f"{recipient} {upi} {note_text} {text}".strip()
            text_normalized = combined_text.lower().strip()
            
            # Extract VPA/UPI ID
            vpa = self._extract_vpa(text_normalized) or upi
            
            # Remove noise patterns
            for pattern in self.noise_patterns:
                text_normalized = re.sub(pattern, '', text_normalized, flags=re.IGNORECASE)
            
            # Keep only alphanumeric and @ . characters
            text_normalized = re.sub(r'[^a-z0-9@\.]', ' ', text_normalized)
            
            # Keep tokens > 2 chars
            tokens = [t for t in text_normalized.split() if len(t) > 2]
            text_normalized = ' '.join(tokens)
            
            # Fuzzy match against canonical aliases
            combined_canonical, combined_conf = self._match_canonical(text_normalized)
            if combined_canonical and combined_conf > confidence:
                canonical = combined_canonical
                confidence = combined_conf
                matched_source = 'combined'
            
            text = text_normalized
        else:
            # Use the canonical match directly
            text = canonical
            vpa = upi
        
        metadata = {
            'original_length': len(original_text),
            'vpa_extracted': vpa,
            'canonical_match': canonical,
            'canonical_confidence': confidence,
            'matched_source': matched_source,  # NEW: Track which field matched
            'token_count': len(text.split()) if not canonical else len(canonical.split()),
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
