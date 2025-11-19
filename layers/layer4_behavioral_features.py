import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict

class BehavioralFeatureExtractor:
    def __init__(self):
        self.feature_names = []
    
    def extract(self, txn: pd.Series, user_history: pd.DataFrame) -> Dict[str, float]:
        """Extract behavioral features for a transaction."""
        features = {}
        
        # Amount features
        amount = txn['amount']
        features['amount_log'] = np.log1p(amount)
        
        if len(user_history) > 0:
            features['amount_percentile'] = self._percentile(amount, user_history['amount'])
            features['amount_zscore'] = self._zscore(amount, user_history['amount'])
        else:
            features['amount_percentile'] = 0.5
            features['amount_zscore'] = 0.0
        
        # Temporal features
        date = pd.to_datetime(txn['date'])
        features['hour'] = date.hour if hasattr(date, 'hour') else 12
        features['day_of_week'] = date.dayofweek
        features['day_of_month'] = date.day
        features['is_weekday'] = 1 if date.dayofweek < 5 else 0
        features['is_commute_window'] = 1 if (7 <= features['hour'] <= 10) or (17 <= features['hour'] <= 20) else 0
        features['is_month_start'] = 1 if date.day <= 5 else 0
        
        # Recurrence features
        if len(user_history) > 0:
            recurrence_info = self._detect_recurrence(txn, user_history)
            features.update(recurrence_info)
        else:
            features['is_periodic'] = 0
            features['recurrence_confidence'] = 0.0
            features['freq_count_30d'] = 0
            features['days_since_last'] = 999
        
        # Merchant features
        merchant = str(txn.get('merchant', ''))
        if len(user_history) > 0 and merchant:
            merchant_txns = user_history[user_history['merchant'] == merchant]
            features['merchant_frequency'] = len(merchant_txns)
            features['is_new_merchant'] = 0 if len(merchant_txns) > 0 else 1
        else:
            features['merchant_frequency'] = 0
            features['is_new_merchant'] = 1
        
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
        merchant = str(txn.get('merchant', ''))
        amount = txn['amount']
        date = pd.to_datetime(txn['date'])
        
        # Find similar transactions (same merchant, similar amount)
        # Protect against division by zero
        if amount == 0:
            amount_tolerance = 0.1  # Absolute tolerance for zero amounts
            similar = history[
                (history['merchant'] == merchant) & 
                (np.abs(history['amount'] - amount) <= amount_tolerance)
            ].copy()
        else:
            similar = history[
                (history['merchant'] == merchant) & 
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

