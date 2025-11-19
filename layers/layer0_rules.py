import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple

class RuleBasedDetector:
    def __init__(self):
        self.categories = {
            'salary': ['salary', 'wages', 'payroll', 'income'],
            'sip': ['mutual', 'sip', 'fund', 'investment'],
            'subscription': ['netflix', 'spotify', 'prime', 'subscription'],
            'utility': ['electricity', 'water', 'gas', 'broadband', 'bill']
        }
    
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
        
        combined_text = f"{description} {merchant}"
        
        # Rule 1: Salary Detection
        if txn_type == 'credit':
            if self._is_salary(amount, date, user_history):
                return 'Salary/Income', 1.0, 'Rule: Credit, monthly, high amount, month start'
        
        # Rule 2: SIP Detection
        if txn_type == 'debit':
            if self._is_sip(amount, description, merchant, user_history):
                return 'Investments', 1.0, 'Rule: Debit, monthly, SIP keywords, recurring'
        
        # Rule 3: Subscription Detection
        if self._is_subscription(combined_text, amount, user_history):
            return 'Subscriptions', 1.0, 'Rule: Recurring pattern, subscription keywords'
        
        # Rule 4: Utility Bills
        if self._is_utility(combined_text, txn_type):
            return 'Bills & Utilities', 1.0, 'Rule: Utility keywords, debit'
        
        return None, 0.0, 'No rule matched'
    
    def _is_salary(self, amount: float, date: datetime, history: pd.DataFrame) -> bool:
        if amount < 20000:
            return False
        if date.day > 5:
            return False
        if len(history) > 0:
            top_5_pct = history[history['type'] == 'credit']['amount'].quantile(0.95)
            if amount < top_5_pct:
                return False
        return True
    
    def _is_sip(self, amount: float, description: str, merchant: str, history: pd.DataFrame) -> bool:
        if not (500 <= amount <= 5000):
            return False
        combined = f"{description} {merchant}".lower()
        if not any(kw in combined for kw in self.categories['sip']):
            return False
        return True
    
    def _is_subscription(self, text: str, amount: float, history: pd.DataFrame) -> bool:
        if any(kw in text for kw in self.categories['subscription']):
            return True
        return False
    
    def _is_utility(self, text: str, txn_type: str) -> bool:
        if txn_type != 'debit':
            return False
        return any(kw in text for kw in self.categories['utility'])

