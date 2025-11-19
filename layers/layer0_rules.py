import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
        
        # KNOWN SUBSCRIPTION SERVICES (definitive list)
        self.known_subscriptions = {
            # Streaming services
            'netflix', 'netflixupi', 'prime', 'amazon prime', 'hotstar', 'disney',
            'zee5', 'sonyliv', 'voot', 'altbalaji', 'mx player', 'jio cinema',
            'apple tv', 'youtube premium', 'spotify', 'gaana', 'jio saavn',
            'amazon music', 'youtube music','airtel',
            # Cloud & software
            'google one', 'icloud', 'microsoft 365', 'office 365', 'dropbox',
            'adobe', 'canva', 'grammarly',
            # News & magazines
            'times', 'hindu', 'mint', 'economic times', 'kindle unlimited',
            # Fitness
            'cult.fit', 'cultfit', 'healthifyme', 'fitbit', 'strava',
            # Other subscriptions
            'linkedin premium', 'medium', 'quora'
        }
    
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
        
        # Priority 2: Salary Detection (ONLY if super obvious)
        # Disabled for now - let semantic layer learn salary patterns
        # if txn_type == 'credit':
        #     salary_match = self._is_salary(amount, date, user_history)
        #     if salary_match:
        #         return 'Salary/Income', 1.0, 'Rule: Credit, monthly, high amount, month start'
        
        # Priority 3: Investment Detection (ONLY explicit SIP keywords)
        if txn_type == 'debit' and amount >= 500:
            # Only if EXPLICIT investment keywords present
            explicit_investment_keywords = ['sip', 'mutual fund', 'zerodha', 'groww', 'upstox', 'systematic']
            if any(kw in combined_text for kw in explicit_investment_keywords):
                return 'Investments', 0.95, 'Rule: Explicit SIP/investment keywords'
        
        # Priority 4: Subscription Detection - DISABLED
        # Let semantic layer learn subscription patterns from context
        # subscription_match = self._is_subscription_strict(
        #     combined_text, recipient_name, upi_id, note, amount, date, user_history
        # )
        # if subscription_match:
        #     return 'Subscriptions', 1.0, 'Rule: Confirmed subscription service'
        
        # Priority 5: Transfer Detection (ONLY explicit keywords)
        # Person-to-person transfers with explicit keywords
        explicit_transfer_keywords = ['neft', 'rtgs', 'imps', 'self transfer', 'own account']
        if any(kw in combined_text for kw in explicit_transfer_keywords):
            return 'Transfers', 0.90, 'Rule: Explicit transfer keywords'
        
        # No rule matched
        return None, 0.0, 'No rule matched'
    
    def _match_corpus(self, text: str) -> Optional[Tuple[str, float, str]]:
        """
        Match against Mumbai merchant corpus.
        ULTRA-MINIMAL: Only catch the TOP 5-10 most obvious brands.
        Let semantic layers (L3/L5) handle everything else.
        
        Goal: L0 should handle < 5% of transactions (not 30%+)
        """
        text = text.lower()
        
        # ULTRA-RESTRICTED: Only these exact brands get L0 classification
        # Everything else goes to semantic/clustering layers
        ultra_obvious_brands = {
            'netflix': 'Subscriptions',
            'netflixupi': 'Subscriptions',
            'spotify': 'Subscriptions',
            'amazon prime': 'Subscriptions',
            'hotstar': 'Subscriptions',
            'swiggy': 'Food & Dining',
            'zomato': 'Food & Dining',
            'uber': 'Commute/Transport',
            'ola': 'Commute/Transport',
            'olacabs': 'Commute/Transport'
        }
        
        # Check for EXACT matches only
        text_clean = text.strip()
        
        # Strategy 1: Exact match (highest confidence)
        if text_clean in ultra_obvious_brands:
            return ultra_obvious_brands[text_clean], 0.99, f'Exact L0 match: "{text_clean}"'
        
        # Strategy 2: Dominant substring (>70% of text, min 6 chars)
        for brand, category in ultra_obvious_brands.items():
            if brand in text_clean and len(brand) >= 6:
                match_ratio = len(brand) / len(text_clean)
                if match_ratio > 0.7:  # Brand is >70% of text
                    return category, 0.90, f'Dominant L0 match: "{brand}"'
        
        # EVERYTHING ELSE → Pass to L3/L5 for semantic analysis
        # This includes:
        # - All local merchants (Julfikar, Bikaner, etc.)
        # - All shops/stores
        # - All bills & utilities
        # - All entertainment venues
        # - All other subscriptions
        # Let the semantic layer learn these patterns!
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
    
    def _is_sip(self, amount: float, combined_text: str, history: pd.DataFrame) -> bool:
        """Detect SIP/mutual fund investments."""
        # Typical SIP range
        if not (100 <= amount <= 100000):
            return False
        
        # Check for investment keywords
        investment_keywords = ['mutual', 'sip', 'fund', 'investment', 'amc', 'mf', 'zerodha', 'groww', 'upstox']
        if not any(kw in combined_text for kw in investment_keywords):
            return False
        
        return True
    
    def _is_subscription_strict(self, combined_text: str, recipient_name: str, 
                               upi_id: str, note: str, amount: float, 
                               date: datetime, history: pd.DataFrame) -> bool:
        """
        STRICT subscription detection with multiple checks.
        All three criteria must be met for confirmation.
        """
        
        # Criterion 1: Is it a KNOWN subscription service?
        is_known_service = False
        for service in self.known_subscriptions:
            if service in combined_text or service in recipient_name or service in upi_id:
                is_known_service = True
                break
        
        # If NOT a known service, apply STRICT checks
        if not is_known_service:
            # Check for explicit subscription keywords
            explicit_keywords = ['subscription', 'membership', 'premium', 'renewal']
            has_explicit_keyword = any(kw in combined_text or kw in note for kw in explicit_keywords)
            
            if not has_explicit_keyword:
                # NOT a known service AND no explicit keywords → NOT a subscription
                return False
        
        # Criterion 2: Subscription amount pattern (₹50 - ₹3000 typical range)
        # Outside this range needs more evidence
        if amount < 50 or amount > 3000:
            if not is_known_service:
                # Large amounts need to be known services
                return False
        
        # Criterion 3: Check recurrence pattern (for confirmation)
        # Must be monthly recurring with same merchant
        if len(history) >= 2:
            # Look for similar transactions (same merchant, similar amount)
            similar_txns = history[
                (history['merchant'] == recipient_name) &
                (history['amount'] >= amount * 0.9) &
                (history['amount'] <= amount * 1.1)
            ].copy()
            
            if len(similar_txns) >= 2:
                # Check if they're monthly (25-35 days apart)
                similar_txns['date'] = pd.to_datetime(similar_txns['date'])
                similar_txns = similar_txns.sort_values('date')
                gaps = similar_txns['date'].diff().dt.days.dropna()
                
                if len(gaps) > 0:
                    avg_gap = gaps.mean()
                    # Monthly pattern (25-35 days)
                    if 25 <= avg_gap <= 35:
                        return True  # Confirmed recurring
        
        # If it's a KNOWN service, return True even without history
        # (First payment from Netflix/Spotify etc is still a subscription)
        if is_known_service:
            return True
        
        # Otherwise, not enough evidence
        return False
    
    def _is_transfer(self, combined_text: str, recipient_name: str, 
                    amount: float, txn_type: str) -> bool:
        """Detect fund transfers - person-to-person payments."""
        # Check for transfer keywords
        transfer_keywords = [
            'transfer', 'neft', 'rtgs', 'imps', 'upi', 'self', 
            'own account', 'wallet', 'fund transfer', 'inter account',
            'sent', 'payme', 'payment to'
        ]
        
        has_transfer_keyword = any(kw in combined_text for kw in transfer_keywords)
        
        # Check if recipient looks like a person name (not a business)
        # Person names are typically 5-15 characters, not in known_subscriptions
        recipient_looks_like_person = (
            len(recipient_name) >= 5 and 
            len(recipient_name) <= 15 and
            recipient_name not in self.known_subscriptions and
            not any(business in recipient_name for business in ['pvr', 'inox', 'swiggy', 'zomato', 'uber'])
        )
        
        # Check if it's a phone number (UPI via phone)
        is_phone_number = recipient_name.isdigit() and len(recipient_name) == 10
        
        # Transfer if:
        # - Has transfer keyword OR
        # - Recipient looks like person name OR
        # - Is phone number transfer
        return has_transfer_keyword or recipient_looks_like_person or is_phone_number
    
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
