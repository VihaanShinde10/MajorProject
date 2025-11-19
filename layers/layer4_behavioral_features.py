import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict

class BehavioralFeatureExtractor:
    def __init__(self):
        self.feature_names = []
    
    def extract(self, txn: pd.Series, user_history: pd.DataFrame) -> Dict[str, float]:
        """
        Extract ENHANCED behavioral features for better discrimination.
        Features designed to capture spending patterns, merchant behavior, temporal patterns.
        """
        features = {}
        
        # Amount features - Enhanced with more granularity
        amount = txn['amount']
        features['amount_log'] = np.log1p(amount)
        features['amount_sqrt'] = np.sqrt(amount)  # Alternative scaling
        features['amount_raw'] = min(amount / 10000, 10.0)  # Capped normalized amount
        
        # Amount range indicators (helps categorize by typical transaction sizes)
        features['is_micro'] = 1 if amount < 100 else 0  # Small purchases
        features['is_small'] = 1 if 100 <= amount < 500 else 0
        features['is_medium'] = 1 if 500 <= amount < 2000 else 0
        features['is_large'] = 1 if 2000 <= amount < 10000 else 0
        features['is_xlarge'] = 1 if amount >= 10000 else 0  # Major purchases/transfers
        
        if len(user_history) > 0:
            features['amount_percentile'] = self._percentile(amount, user_history['amount'])
            features['amount_zscore'] = self._zscore(amount, user_history['amount'])
            
            # Deviation from user's typical spending
            features['amount_deviation'] = abs(amount - user_history['amount'].median()) / (user_history['amount'].std() + 1)
        else:
            features['amount_percentile'] = 0.5
            features['amount_zscore'] = 0.0
            features['amount_deviation'] = 0.0
        
        # Temporal features - Enhanced for better pattern detection
        date = pd.to_datetime(txn['date'])
        hour = date.hour if hasattr(date, 'hour') else 12
        features['hour'] = hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)  # Cyclical encoding
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        features['day_of_week'] = date.dayofweek
        features['day_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)  # Cyclical
        features['day_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)
        
        features['day_of_month'] = date.day
        features['is_weekday'] = 1 if date.dayofweek < 5 else 0
        features['is_weekend'] = 1 if date.dayofweek >= 5 else 0
        
        # Time-of-day patterns (better for category discrimination)
        features['is_early_morning'] = 1 if 5 <= hour < 9 else 0  # Commute, breakfast
        features['is_work_hours'] = 1 if 9 <= hour < 17 else 0  # Work-related
        features['is_evening'] = 1 if 17 <= hour < 22 else 0  # Dining, entertainment
        features['is_late_night'] = 1 if hour >= 22 or hour < 5 else 0  # Subscriptions, unusual
        features['is_commute_window'] = 1 if (7 <= hour <= 10) or (17 <= hour <= 20) else 0
        
        # Month patterns
        features['is_month_start'] = 1 if date.day <= 5 else 0  # Salary, bills
        features['is_month_mid'] = 1 if 10 <= date.day <= 20 else 0
        features['is_month_end'] = 1 if date.day >= 25 else 0
        
        # Recurrence features
        if len(user_history) > 0:
            recurrence_info = self._detect_recurrence(txn, user_history)
            features.update(recurrence_info)
        else:
            features['is_periodic'] = 0
            features['recurrence_confidence'] = 0.0
            features['freq_count_30d'] = 0
            features['days_since_last'] = 999
        
        # Merchant features - ENHANCED for better behavior patterns
        recipient_name = str(txn.get('Recipient_Name', txn.get('recipient_name', '')))
        merchant = str(txn.get('merchant', ''))
        
        # Use recipient_name if available and valid, else fall back to merchant
        merchant_id = recipient_name if (recipient_name and recipient_name not in ['', 'nan', 'none', 'None']) else merchant
        
        if len(user_history) > 0 and merchant_id:
            # Check both recipient_name and merchant columns for matching
            if 'Recipient_Name' in user_history.columns or 'recipient_name' in user_history.columns:
                recipient_col = 'Recipient_Name' if 'Recipient_Name' in user_history.columns else 'recipient_name'
                merchant_txns = user_history[
                    (user_history[recipient_col] == merchant_id) | 
                    (user_history['merchant'] == merchant_id)
                ]
            else:
                merchant_txns = user_history[user_history['merchant'] == merchant_id]
            
            features['merchant_frequency'] = min(len(merchant_txns) / 10, 5.0)  # Normalized
            features['is_new_merchant'] = 0 if len(merchant_txns) > 0 else 1
            features['is_regular_merchant'] = 1 if len(merchant_txns) >= 3 else 0  # Regular vendor
            features['is_frequent_merchant'] = 1 if len(merchant_txns) >= 10 else 0
            
            # Merchant amount consistency (helps identify subscription vs variable spending)
            if len(merchant_txns) > 1:
                merchant_amounts = merchant_txns['amount'].values
                features['merchant_amount_std'] = np.std(merchant_amounts) / (np.mean(merchant_amounts) + 1)
                features['merchant_amount_consistency'] = 1 / (1 + features['merchant_amount_std'])
            else:
                features['merchant_amount_std'] = 1.0
                features['merchant_amount_consistency'] = 0.5
        else:
            features['merchant_frequency'] = 0.0
            features['is_new_merchant'] = 1
            features['is_regular_merchant'] = 0
            features['is_frequent_merchant'] = 0
            features['merchant_amount_std'] = 1.0
            features['merchant_amount_consistency'] = 0.5
        
        # Transaction type indicators (debit vs credit patterns)
        txn_type = str(txn.get('type', '')).lower()
        features['is_debit'] = 1 if 'debit' in txn_type or txn_type == 'dr' else 0
        features['is_credit'] = 1 if 'credit' in txn_type or txn_type == 'cr' else 0
        
        # Rolling statistics (7d, 30d)
        if len(user_history) > 0:
            rolling_stats = self._rolling_stats(txn, user_history)
            features.update(rolling_stats)
        else:
            features['avg_amount_7d'] = amount
            features['std_amount_7d'] = 0.0
            features['txn_count_7d'] = 0
        
        self.feature_names = list(features.keys())
        return features
    
    def _percentile(self, value: float, series: pd.Series) -> float:
        if len(series) == 0:
            return 0.5
        result = (series < value).sum() / len(series)
        return result if not np.isnan(result) else 0.5
    
    def _zscore(self, value: float, series: pd.Series) -> float:
        if len(series) == 0:
            return 0.0
        mean = series.mean()
        std = series.std()
        # Handle NaN values
        if np.isnan(mean) or np.isnan(std) or std == 0:
            return 0.0
        result = (value - mean) / std
        return result if not np.isnan(result) and not np.isinf(result) else 0.0
    
    def _detect_recurrence(self, txn: pd.Series, history: pd.DataFrame) -> Dict:
        """
        Detect recurring transactions.
        ROBUST: Uses recipient_name (UPI) for better merchant matching.
        """
        # Priority: Use recipient_name if available
        recipient_name = str(txn.get('Recipient_Name', txn.get('recipient_name', '')))
        merchant = str(txn.get('merchant', ''))
        
        # Use recipient_name if available and valid
        merchant_id = recipient_name if (recipient_name and recipient_name not in ['', 'nan', 'none', 'None']) else merchant
        
        amount = txn['amount']
        date = pd.to_datetime(txn['date'])
        
        # Find similar transactions (same merchant_id, similar amount)
        # Match using recipient_name or merchant
        if 'Recipient_Name' in history.columns or 'recipient_name' in history.columns:
            recipient_col = 'Recipient_Name' if 'Recipient_Name' in history.columns else 'recipient_name'
            merchant_mask = (history[recipient_col] == merchant_id) | (history['merchant'] == merchant_id)
        else:
            merchant_mask = history['merchant'] == merchant_id
        
        # Protect against division by zero
        if amount == 0:
            amount_tolerance = 0.1  # Absolute tolerance for zero amounts
            similar = history[
                merchant_mask & 
                (np.abs(history['amount'] - amount) <= amount_tolerance)
            ].copy()
        else:
            similar = history[
                merchant_mask & 
                (np.abs(history['amount'] - amount) / amount < 0.1)
            ].copy()
        
        if len(similar) < 2:
            return {
                'is_periodic': 0,
                'recurrence_confidence': 0.0,
                'freq_count_30d': 0,
                'days_since_last': 999
            }
        
        # Calculate gaps
        similar['date'] = pd.to_datetime(similar['date'])
        similar = similar.sort_values('date')
        gaps = similar['date'].diff().dt.days.dropna()
        
        if len(gaps) < 2:
            return {
                'is_periodic': 0,
                'recurrence_confidence': 0.0,
                'freq_count_30d': len(similar),
                'days_since_last': (date - similar['date'].iloc[-1]).days if len(similar) > 0 else 999
            }
        
        gap_mean = gaps.mean()
        gap_std = gaps.std()
        
        # Handle NaN values from mean/std
        if np.isnan(gap_mean):
            gap_mean = 999
        if np.isnan(gap_std):
            gap_std = 999
        
        # Check if periodic (monthly pattern)
        is_monthly = 25 <= gap_mean <= 35 and gap_std <= 7
        is_periodic = 1 if is_monthly and len(similar) >= 3 else 0
        
        if gap_mean > 0 and not np.isnan(gap_std):
            confidence = 1 - (gap_std / gap_mean)
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.0
        
        # Final NaN check
        if np.isnan(confidence) or np.isinf(confidence):
            confidence = 0.0
        
        count_30d = len(similar[similar['date'] >= (date - timedelta(days=30))])
        days_since = (date - similar['date'].iloc[-1]).days if len(similar) > 0 else 999
        
        return {
            'is_periodic': is_periodic,
            'recurrence_confidence': confidence,
            'freq_count_30d': count_30d,
            'days_since_last': days_since
        }
    
    def _rolling_stats(self, txn: pd.Series, history: pd.DataFrame) -> Dict:
        date = pd.to_datetime(txn['date'])
        
        # 7-day window
        window_7d = history[pd.to_datetime(history['date']) >= (date - timedelta(days=7))]
        
        if len(window_7d) > 0:
            avg_7d = window_7d['amount'].mean()
            std_7d = window_7d['amount'].std()
            count_7d = len(window_7d)
            
            # Handle NaN values
            if np.isnan(avg_7d):
                avg_7d = txn['amount']
            if np.isnan(std_7d):
                std_7d = 0.0
        else:
            avg_7d = txn['amount']
            std_7d = 0.0
            count_7d = 0
        
        return {
            'avg_amount_7d': avg_7d,
            'std_amount_7d': std_7d if not np.isnan(std_7d) else 0.0,
            'txn_count_7d': count_7d
        }

