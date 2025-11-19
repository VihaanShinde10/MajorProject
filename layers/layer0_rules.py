import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import json
import os

class RuleBasedDetector:
    def __init__(self, corpus_path: str = None):
        """Initialize with Mumbai merchant corpus."""
        # Fixed categories - these are the ONLY allowed categories
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
        
        # Load merchant corpus
        if corpus_path is None:
            corpus_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mumbai_merchants_corpus.json')
        
        self.merchant_corpus = {}
        if os.path.exists(corpus_path):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                self.merchant_corpus = json.load(f)
        
        # Build quick lookup dictionary
        self.keyword_to_category = {}
        self._build_keyword_map()
    
    def _build_keyword_map(self):
        """Build reverse lookup: keyword -> category."""
        for category, data in self.merchant_corpus.items():
            if category not in self.fixed_categories:
                continue  # Skip invalid categories
                
            # Add all subcategories
            for subcategory, items in data.items():
                for item in items:
                    # Store as lowercase for matching
                    self.keyword_to_category[item.lower()] = category
    
    def detect(self, txn: pd.Series, user_history: pd.DataFrame) -> Tuple[Optional[str], float, str]:
        """
        Detect transaction category using deterministic rules.
        Returns: (category, confidence, reason)
        """
        amount = txn['amount']
        description = str(txn.get('description', '')).lower()
        merchant = str(txn.get('merchant', '')).lower()
        txn_type = str(txn.get('type', '')).lower()
        date = pd.to_datetime(txn['date'])
        
        # Enhanced: Include UPI fields for better matching
        recipient_name = str(txn.get('Recipient_Name', txn.get('recipient_name', ''))).lower()
        upi_id = str(txn.get('UPI_ID', txn.get('upi_id', ''))).lower()
        note = str(txn.get('Note', txn.get('note', ''))).lower()
        
        # Combine all text sources
        combined_text = f"{description} {merchant} {recipient_name} {upi_id} {note}"
        
        # Priority 1: Check merchant corpus (most reliable)
        corpus_match = self._match_corpus(combined_text)
        if corpus_match:
            category, confidence, reason = corpus_match
            return category, confidence, reason
        
        # Priority 2: Salary Detection (very specific rules)
        if txn_type == 'credit':
            salary_match = self._is_salary(amount, date, user_history)
            if salary_match:
                return 'Salary/Income', 1.0, 'Rule: Credit, monthly, high amount, month start'
        
        # Priority 3: Investment Detection (SIP patterns)
        if txn_type == 'debit':
            sip_match = self._is_sip(amount, description, merchant, user_history)
            if sip_match:
                return 'Investments', 1.0, 'Rule: Debit, monthly, SIP keywords, recurring'
        
        # Priority 4: Subscription Detection (recurring + keywords)
        subscription_match = self._is_subscription(combined_text, amount, user_history)
        if subscription_match:
            return 'Subscriptions', 1.0, 'Rule: Recurring pattern, subscription keywords'
        
        # Priority 5: Transfer Detection
        transfer_match = self._is_transfer(combined_text, txn_type)
        if transfer_match:
            return 'Transfers', 0.95, 'Rule: Transfer keywords detected'
        
        # No rule matched
        return None, 0.0, 'No rule matched'
    
    def _match_corpus(self, text: str) -> Optional[Tuple[str, float, str]]:
        """Match against Mumbai merchant corpus."""
        text = text.lower()
        
        # Check for exact or partial matches
        matched_items = []
        for keyword, category in self.keyword_to_category.items():
            if keyword in text:
                # Calculate match quality
                match_length = len(keyword)
                text_length = len(text)
                match_ratio = match_length / text_length
                
                matched_items.append((category, keyword, match_ratio))
        
        if not matched_items:
            return None
        
        # Sort by match ratio (best match first)
        matched_items.sort(key=lambda x: x[2], reverse=True)
        
        best_category, best_keyword, best_ratio = matched_items[0]
        
        # High confidence for good matches
        if best_ratio > 0.3 or len(best_keyword) > 8:
            confidence = min(0.98, 0.85 + best_ratio * 0.3)
            return best_category, confidence, f'Corpus match: "{best_keyword}"'
        
        # Medium confidence for partial matches
        if best_ratio > 0.15:
            confidence = 0.75
            return best_category, confidence, f'Partial corpus match: "{best_keyword}"'
        
        return None
    
    def _is_salary(self, amount: float, date: datetime, history: pd.DataFrame) -> bool:
        """Detect salary with strict rules."""
        # Minimum salary threshold for Mumbai
        if amount < 15000:
            return False
        
        # Check if it's in first 5 days or last 3 days of month
        if not (date.day <= 5 or date.day >= 28):
            return False
        
        # Check if amount is in top 5% of credits
        if len(history) > 0:
            credit_history = history[history['type'] == 'credit']
            if len(credit_history) > 0:
                top_5_pct = credit_history['amount'].quantile(0.95)
                if amount < top_5_pct * 0.8:  # Allow some variance
                    return False
        
        return True
    
    def _is_sip(self, amount: float, description: str, merchant: str, history: pd.DataFrame) -> bool:
        """Detect SIP/mutual fund investments."""
        # Typical SIP range
        if not (100 <= amount <= 100000):
            return False
        
        combined = f"{description} {merchant}".lower()
        
        # Check for investment keywords
        investment_keywords = ['mutual', 'sip', 'fund', 'investment', 'amc', 'mf']
        if not any(kw in combined for kw in investment_keywords):
            return False
        
        return True
    
    def _is_subscription(self, text: str, amount: float, history: pd.DataFrame) -> bool:
        """Detect subscriptions."""
        subscription_keywords = [
            'subscription', 'membership', 'premium', 'annual', 'monthly',
            'netflix', 'spotify', 'prime', 'hotstar', 'zee5'
        ]
        
        if any(kw in text for kw in subscription_keywords):
            return True
        
        # Check for small recurring amounts (typical subscriptions)
        if 50 <= amount <= 2000:
            # Check if this amount recurs
            if len(history) > 0:
                similar_amounts = history[
                    (history['amount'] >= amount * 0.95) & 
                    (history['amount'] <= amount * 1.05)
                ]
                if len(similar_amounts) >= 2:
                    return True
        
        return False
    
    def _is_transfer(self, text: str, txn_type: str) -> bool:
        """Detect fund transfers."""
        transfer_keywords = [
            'transfer', 'neft', 'rtgs', 'imps', 'upi', 'self', 
            'own account', 'wallet', 'paytm', 'phonepe', 'gpay',
            'fund transfer', 'inter account'
        ]
        
        return any(kw in text for kw in transfer_keywords)
    
    def validate_category(self, category: str) -> str:
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
