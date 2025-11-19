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
        ROBUST: Prioritizes recipient_name, upi_id, and note fields.
        ENHANCED: Uses transaction type hints from Description field patterns.
        Returns: (category, confidence, reason)
        """
        amount = txn['amount']
        description = str(txn.get('description', '')).lower()
        merchant = str(txn.get('merchant', '')).lower()
        txn_type = str(txn.get('type', '')).lower()
        date = pd.to_datetime(txn['date'])
        
        # PRIORITY FIELDS: Extract UPI-specific data
        recipient_name = str(txn.get('Recipient_Name', txn.get('recipient_name', ''))).lower()
        upi_id = str(txn.get('UPI_ID', txn.get('upi_id', ''))).lower()
        note = str(txn.get('Note', txn.get('note', ''))).lower()
        
        # Clean empty values
        recipient_name = recipient_name if recipient_name not in ['', 'nan', 'none'] else ''
        upi_id = upi_id if upi_id not in ['', 'nan', 'none'] else ''
        note = note if note not in ['', 'nan', 'none'] else ''
        
        # ENHANCED: Check for transaction type hints from Description patterns
        # Patterns like "SALARY-NEFT", "AUTO-DEBIT", "ATM-CASH-WDL"
        txn_type_hint = self._detect_transaction_type_from_description(description)
        
        # REDUCED DOMINANCE: Only use type hints for very specific cases
        # Most transactions should flow through semantic/behavioral layers
        # Only catch: Salary (very clear), ATM withdrawals (definitive)
        if txn_type_hint in ['Salary/Income', 'Transfers']:
            if txn_type_hint == 'Salary/Income' and txn_type == 'credit' and amount >= 20000:
                # Very high salary threshold to avoid false positives
                return txn_type_hint, 0.96, f'Rule: High-value salary credit'
            elif txn_type_hint == 'Transfers' and 'ATM-CASH-WDL' in description.upper():
                # ATM withdrawals are definitive transfers
                return 'Transfers', 0.98, f'Rule: ATM cash withdrawal'
        # All other hints are ignored - let deeper layers analyze
        
        # Priority 1: Check RECIPIENT_NAME first (most reliable for UPI)
        if recipient_name:
            recipient_match = self._match_corpus(recipient_name)
            if recipient_match:
                category, confidence, reason = recipient_match
                return category, confidence + 0.02, f"{reason} [from RECIPIENT_NAME]"
        
        # Priority 2: Check UPI_ID (often contains merchant identifier)
        if upi_id:
            upi_match = self._match_corpus(upi_id)
            if upi_match:
                category, confidence, reason = upi_match
                return category, confidence + 0.01, f"{reason} [from UPI_ID]"
        
        # Priority 3: Check NOTE field (provides context)
        if note:
            note_match = self._match_corpus(note)
            if note_match:
                category, confidence, reason = note_match
                return category, confidence, f"{reason} [from NOTE]"
        
        # Priority 4: Check combined text (fallback)
        combined_text = f"{recipient_name} {upi_id} {note} {description} {merchant}".strip()
        corpus_match = self._match_corpus(combined_text)
        if corpus_match:
            category, confidence, reason = corpus_match
            return category, confidence, reason
        
        # Priority 5: Salary Detection (very specific rules)
        if txn_type == 'credit':
            salary_match = self._is_salary(amount, date, user_history)
            if salary_match:
                return 'Salary/Income', 1.0, 'Rule: Credit, monthly, high amount, month start'
        
        # Priority 6: Investment Detection (SIP patterns)
        if txn_type == 'debit':
            sip_match = self._is_sip(amount, combined_text, user_history)
            if sip_match:
                return 'Investments', 1.0, 'Rule: Debit, monthly, SIP keywords, recurring'
        
        # Priority 7: STRICT Subscription Detection (checks recipient + note)
        subscription_match = self._is_subscription_strict(
            combined_text, recipient_name, upi_id, note, amount, date, user_history
        )
        if subscription_match:
            return 'Subscriptions', 1.0, 'Rule: Confirmed subscription service'
        
        # Priority 8: Transfer Detection (checks recipient + note)
        transfer_match = self._is_transfer(combined_text, recipient_name, amount, txn_type)
        if transfer_match:
            return 'Transfers', 0.95, 'Rule: Transfer keywords or person name detected'
        
        # No rule matched
        return None, 0.0, 'No rule matched'
    
    def _match_corpus(self, text: str) -> Optional[Tuple[str, float, str]]:
        """
        Match against Mumbai merchant corpus.
        ULTRA-PRECISE + COMPREHENSIVE: 
        - Word-boundary checking to prevent false matches
        - MULTI-WORD ANALYSIS: Check EACH word in text individually
        - Ensures EVERY word is considered for matching
        """
        text = text.lower().strip()
        
        # Strategy 1: Check the complete text
        matched_items = []
        for keyword, category in self.keyword_to_category.items():
            # Exact match (highest priority)
            if keyword == text:
                matched_items.append((category, keyword, 1.0, True, True, True, 'exact'))
                continue
            
            # Substring with word boundaries
            if keyword in text:
                # Find the keyword position
                keyword_start = text.find(keyword)
                keyword_end = keyword_start + len(keyword)
                
                # Check if it's at word boundaries (not part of another word)
                is_word_start = keyword_start == 0 or not text[keyword_start-1].isalnum()
                is_word_end = keyword_end == len(text) or not text[keyword_end].isalnum()
                
                # Only consider it a match if it's a complete word
                if is_word_start and is_word_end:
                    match_length = len(keyword)
                    text_length = len(text)
                    match_ratio = match_length / text_length
                    
                    is_exact_match = (keyword == text)
                    is_dominant = (match_ratio > 0.5)
                    
                    matched_items.append((category, keyword, match_ratio, is_exact_match, is_dominant, True, 'substring'))
        
        # Strategy 2: MULTI-WORD ANALYSIS - Check EACH word individually
        # This ensures we don't miss keywords in multi-word transactions
        # e.g., "payment to swiggy" → catches "swiggy"
        words = text.split()
        for word in words:
            word_clean = word.strip()
            if len(word_clean) < 3:  # Skip very short words
                continue
            
            # Check if this word matches any keyword
            for keyword, category in self.keyword_to_category.items():
                # Exact word match
                if keyword == word_clean:
                    # Word match gets slightly lower confidence than full text match
                    matched_items.append((category, keyword, 0.8, False, True, True, 'word_match'))
                    continue
                
                # Word contains keyword with word boundaries
                if keyword in word_clean and len(keyword) >= 4:
                    keyword_start = word_clean.find(keyword)
                    keyword_end = keyword_start + len(keyword)
                    
                    is_word_start = keyword_start == 0 or not word_clean[keyword_start-1].isalnum()
                    is_word_end = keyword_end == len(word_clean) or not word_clean[keyword_end].isalnum()
                    
                    if is_word_start and is_word_end:
                        match_ratio = len(keyword) / len(word_clean)
                        if match_ratio > 0.5:  # Keyword is significant part of word
                            matched_items.append((category, keyword, match_ratio * 0.7, False, True, True, 'word_partial'))
        
        if not matched_items:
            return None
        
        # Sort by match quality (exact > word_boundary > dominant > ratio > method)
        matched_items.sort(key=lambda x: (x[3], x[5], x[4], x[2]), reverse=True)
        
        best_category, best_keyword, best_ratio, is_exact, is_dominant, is_word_boundary, match_method = matched_items[0]
        
        # MINIMALIST LAYER 0: Only catch EXACT, UNAMBIGUOUS matches
        # Goal: Let 90%+ of transactions flow through semantic/behavioral layers
        # Layer 0 should be a safety net for obvious cases ONLY
        
        # Rule 1: EXACT match with well-known brands ONLY
        top_brands_only = [
            'netflix', 'amazon', 'flipkart', 'swiggy', 'zomato', 'uber', 'ola',
            'spotify', 'hotstar', 'paytm', 'phonepe', 'googlepay', 'mcdonalds', 'kfc'
        ]
        if is_exact and best_keyword in top_brands_only and len(best_keyword) >= 5:
            confidence = 0.95  # High but not perfect - allow layers to refine
            return best_category, confidence, f'Exact brand match: "{best_keyword}"'
        
        # Rule 2: ONLY for very dominant matches (>70% of text) with long keywords
        if is_word_boundary and is_dominant and best_ratio > 0.70 and len(best_keyword) >= 8:
            confidence = 0.88  # Lower confidence - encourage layer participation
            return best_category, confidence, f'Dominant match: "{best_keyword}"'
        
        # REJECT everything else - semantic and behavioral layers will handle
        # This ensures comprehensive layer participation
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
        ULTRA-STRICT subscription detection.
        FIXED: Prevent transfers from being classified as subscriptions.
        """
        
        # FIRST: Check if it's clearly a TRANSFER (highest priority)
        # If it looks like a transfer, immediately reject subscription classification
        transfer_indicators = [
            'transfer', 'neft', 'rtgs', 'imps', 'sent', 'payme', 'payment to',
            'fund transfer', 'self', 'own account', 'wallet'
        ]
        
        # Check if it's a phone number (10 digits) - clear transfer
        if recipient_name.isdigit() and len(recipient_name) == 10:
            return False  # Phone number UPI = Transfer, NOT subscription
        
        # Check for transfer keywords
        if any(kw in combined_text.lower() or kw in note.lower() for kw in transfer_indicators):
            return False  # Has transfer keywords = NOT a subscription
        
        # Check if recipient looks like a person name (5-20 chars, no business keywords)
        business_keywords = ['pvr', 'inox', 'netflix', 'spotify', 'hotstar', 'prime', 
                            'swiggy', 'zomato', 'uber', 'ola', 'flipkart', 'amazon']
        recipient_is_person = (
            5 <= len(recipient_name) <= 20 and
            not any(biz in recipient_name.lower() for biz in business_keywords) and
            not recipient_name.lower().endswith('upi')
        )
        if recipient_is_person:
            return False  # Person name = Transfer, NOT subscription
        
        # NOW check if it's a KNOWN subscription service
        is_known_service = False
        for service in self.known_subscriptions:
            if service in combined_text or service in recipient_name or service in upi_id:
                is_known_service = True
                break
        
        # If NOT a known service, apply ULTRA-STRICT checks
        if not is_known_service:
            # Need MULTIPLE explicit subscription keywords (not just one)
            explicit_keywords = ['subscription', 'membership', 'premium plan', 'renewal', 'recurring']
            keyword_count = sum(1 for kw in explicit_keywords if kw in combined_text or kw in note)
            
            if keyword_count < 2:  # Need at least 2 explicit keywords
                return False  # NOT enough evidence for subscription
        
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
        """
        Detect fund transfers - person-to-person payments.
        ENHANCED: More comprehensive transfer detection.
        """
        # Strategy 1: Explicit transfer keywords
        transfer_keywords = [
            'transfer', 'neft', 'rtgs', 'imps', 'sent', 'payme', 'payment to',
            'fund transfer', 'self', 'own account', 'wallet', 'inter account',
            'send money', 'money sent', 'paid to', 'p2p', 'person to person'
        ]
        has_transfer_keyword = any(kw in combined_text.lower() for kw in transfer_keywords)
        
        # Strategy 2: Phone number (UPI via phone) - STRONG indicator
        is_phone_number = recipient_name.isdigit() and len(recipient_name) == 10
        
        # Strategy 3: Check if recipient looks like a person name (not a business)
        business_keywords = [
            'pvr', 'inox', 'swiggy', 'zomato', 'uber', 'ola', 'rapido',
            'netflix', 'spotify', 'hotstar', 'prime', 'amazon', 'flipkart',
            'paytm', 'phonepe', 'googlepay', 'bhim', 'upi',
            'mall', 'store', 'shop', 'mart', 'cafe', 'restaurant'
        ]
        
        recipient_looks_like_person = (
            5 <= len(recipient_name) <= 20 and  # Person name length
            not any(biz in recipient_name.lower() for biz in business_keywords) and
            not recipient_name.lower().endswith('upi') and
            not recipient_name.lower().endswith('pay') and
            recipient_name not in self.known_subscriptions
        )
        
        # Strategy 4: Common person name patterns (Indian names)
        common_name_parts = [
            'kumar', 'sharma', 'singh', 'verma', 'gupta', 'patel', 'shah',
            'rajesh', 'suresh', 'ramesh', 'mahesh', 'dinesh', 'vijay',
            'amit', 'ankit', 'rohit', 'priya', 'pooja', 'neha', 'ravi'
        ]
        has_common_name = any(name in recipient_name.lower() for name in common_name_parts)
        
        # Transfer if ANY of these conditions are met:
        return (
            has_transfer_keyword or 
            is_phone_number or 
            recipient_looks_like_person or 
            has_common_name
        )
    
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
    
    def _detect_transaction_type_from_description(self, description: str) -> Optional[str]:
        """
        Detect transaction type from Description field patterns.
        Examples: "SALARY-NEFT", "AUTO-DEBIT", "ATM-CASH-WDL"
        """
        if not description:
            return None
        
        desc_upper = description.upper()
        
        # Salary patterns
        if 'SALARY' in desc_upper or 'INB-CREDIT' in desc_upper:
            return 'Salary/Income'
        
        # Subscription patterns
        if 'AUTO-DEBIT' in desc_upper or desc_upper.startswith('SUBS'):
            return 'Subscriptions'
        
        # Bills patterns
        if 'BILL-PAY' in desc_upper or 'RECHARGE' in desc_upper:
            return 'Bills & Utilities'
        
        # Transfer patterns
        if 'ATM-CASH-WDL' in desc_upper or 'NEFT-TRF' in desc_upper or 'IMPS-TO' in desc_upper:
            return 'Transfers'
        if 'MANUAL-REVIEW' in desc_upper or 'PENDING-REVIEW' in desc_upper:
            return 'Transfers'
        
        # Transport patterns
        if 'TICKET' in desc_upper or 'IRCTC' in desc_upper or desc_upper.startswith('RIDE'):
            return 'Commute/Transport'
        
        # Entertainment patterns
        if desc_upper.startswith('ENT-'):
            return 'Entertainment'
        
        return None
    
    def _validate_type_hint(self, hint_category: str, amount: float, txn_type: str, context: str) -> bool:
        """
        Validate that the transaction type hint makes sense given the context.
        """
        # Salary: should be credit, reasonable amount
        if hint_category == 'Salary/Income':
            return txn_type == 'credit' and amount >= 10000
        
        # Subscriptions: typically small to medium debits
        if hint_category == 'Subscriptions':
            return txn_type == 'debit' and 50 <= amount <= 5000
        
        # Bills: medium range debits
        if hint_category == 'Bills & Utilities':
            return txn_type == 'debit' and 50 <= amount <= 10000
        
        # Transfers: any amount, check for person names
        if hint_category == 'Transfers':
            # ATM withdrawals are always transfers
            return True
        
        # Transport: typically small to medium
        if hint_category == 'Commute/Transport':
            return txn_type == 'debit' and 50 <= amount <= 10000
        
        # Entertainment: medium range
        if hint_category == 'Entertainment':
            return txn_type == 'debit' and 100 <= amount <= 5000
        
        return True  # Default: allow hint
