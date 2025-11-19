import hashlib
import json
from typing import Optional, Dict, Tuple

class TransactionCache:
    """
    Deterministic caching system to ensure IDENTICAL transactions ALWAYS get the SAME label.
    
    Problem: Same transaction was getting different labels on different runs.
    Solution: Hash transaction key fields and cache the result.
    """
    
    def __init__(self):
        self.cache = {}  # hash -> (category, confidence, reason, layer_used)
        self.hit_count = 0
        self.miss_count = 0
    
    def _compute_hash(self, 
                     recipient_name: str = '', 
                     upi_id: str = '', 
                     note: str = '',
                     description: str = '',
                     amount: float = 0.0,
                     txn_type: str = '') -> str:
        """
        Compute deterministic hash from transaction key fields.
        CRITICAL: Same input ALWAYS produces same hash.
        """
        # Normalize all inputs to handle None/NaN
        recipient = str(recipient_name).lower().strip() if recipient_name else ''
        upi = str(upi_id).lower().strip() if upi_id else ''
        note_text = str(note).lower().strip() if note else ''
        desc = str(description).lower().strip() if description else ''
        amt = round(float(amount), 2) if amount else 0.0
        typ = str(txn_type).lower().strip() if txn_type else ''
        
        # Create a deterministic key from all fields
        # Order matters - same order always
        key_parts = [
            f"recipient:{recipient}",
            f"upi:{upi}",
            f"note:{note_text}",
            f"desc:{desc}",
            f"amount:{amt}",
            f"type:{typ}"
        ]
        
        key_string = "|".join(key_parts)
        
        # Compute SHA256 hash for deterministic unique identifier
        hash_object = hashlib.sha256(key_string.encode('utf-8'))
        return hash_object.hexdigest()
    
    def get(self, 
            recipient_name: str = '', 
            upi_id: str = '', 
            note: str = '',
            description: str = '',
            amount: float = 0.0,
            txn_type: str = '') -> Optional[Tuple[str, float, str, str]]:
        """
        Get cached result for this transaction.
        Returns: (category, confidence, reason, layer_used) or None
        """
        txn_hash = self._compute_hash(recipient_name, upi_id, note, description, amount, txn_type)
        
        if txn_hash in self.cache:
            self.hit_count += 1
            return self.cache[txn_hash]
        
        self.miss_count += 1
        return None
    
    def set(self,
            category: str,
            confidence: float,
            reason: str,
            layer_used: str,
            recipient_name: str = '', 
            upi_id: str = '', 
            note: str = '',
            description: str = '',
            amount: float = 0.0,
            txn_type: str = ''):
        """
        Cache the result for this transaction.
        ENSURES: Same transaction will ALWAYS return this result.
        """
        txn_hash = self._compute_hash(recipient_name, upi_id, note, description, amount, txn_type)
        
        self.cache[txn_hash] = (category, confidence, reason, layer_used)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': f"{hit_rate:.1f}%"
        }
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0

