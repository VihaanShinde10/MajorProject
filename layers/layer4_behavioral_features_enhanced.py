"""
Enhanced Behavioral Feature Extraction with Advanced Patterns.

Adds sophisticated features for better transaction discrimination:
- Velocity features (spending rate changes)
- Sequence patterns (transaction chains)
- Network features (merchant relationships)
- Anomaly detection features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from collections import Counter, defaultdict
from scipy import stats

class EnhancedBehavioralFeatureExtractor:
    """
    Enhanced behavioral feature extractor with advanced pattern detection.
    Builds on base features with velocity, sequence, and network analysis.
    """
    
    def __init__(self):
        self.feature_names = []
        self.merchant_graph = defaultdict(set)  # Merchant co-occurrence network
    
    def extract(self, txn: pd.Series, user_history: pd.DataFrame) -> Dict[str, float]:
        """
        Extract comprehensive behavioral features.
        Combines base features with advanced patterns.
        """
        features = {}
        
        # Base features (from original extractor)
        features.update(self._extract_base_features(txn, user_history))
        
        # Advanced features
        if len(user_history) > 10:  # Need sufficient history
            features.update(self._extract_velocity_features(txn, user_history))
            features.update(self._extract_sequence_features(txn, user_history))
            features.update(self._extract_network_features(txn, user_history))
            features.update(self._extract_anomaly_features(txn, user_history))
            features.update(self._extract_category_hints(txn, user_history))
        else:
            # Default values for cold start
            features.update(self._default_advanced_features())
        
        return features
    
    def _extract_base_features(self, txn: pd.Series, user_history: pd.DataFrame) -> Dict[str, float]:
        """Extract base behavioral features."""
        features = {}
        
        amount = txn['amount']
        date = pd.to_datetime(txn['date'])
        
        # Amount features
        features['amount_log'] = np.log1p(amount)
        features['amount_sqrt'] = np.sqrt(amount)
        features['amount_raw'] = min(amount / 10000, 10.0)
        
        # Amount range indicators
        features['is_micro'] = 1 if amount < 100 else 0
        features['is_small'] = 1 if 100 <= amount < 500 else 0
        features['is_medium'] = 1 if 500 <= amount < 2000 else 0
        features['is_large'] = 1 if 2000 <= amount < 10000 else 0
        features['is_xlarge'] = 1 if amount >= 10000 else 0
        
        if len(user_history) > 0:
            features['amount_percentile'] = self._percentile(amount, user_history['amount'])
            features['amount_zscore'] = self._zscore(amount, user_history['amount'])
        else:
            features['amount_percentile'] = 0.5
            features['amount_zscore'] = 0.0
        
        # Temporal features
        hour = date.hour if hasattr(date, 'hour') else 12
        features['hour'] = hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['day_of_week'] = date.dayofweek
        features['day_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
        features['day_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)
        
        # Time windows
        features['is_early_morning'] = 1 if 5 <= hour < 9 else 0
        features['is_work_hours'] = 1 if 9 <= hour < 17 else 0
        features['is_evening'] = 1 if 17 <= hour < 22 else 0
        features['is_late_night'] = 1 if hour >= 22 or hour < 5 else 0
        features['is_commute_window'] = 1 if (7 <= hour <= 10) or (17 <= hour <= 20) else 0
        
        # Month patterns
        features['is_month_start'] = 1 if date.day <= 5 else 0
        features['is_month_end'] = 1 if date.day >= 25 else 0
        
        return features
    
    def _extract_velocity_features(self, txn: pd.Series, user_history: pd.DataFrame) -> Dict[str, float]:
        """
        Extract velocity features (rate of change in spending).
        Captures acceleration/deceleration in spending patterns.
        """
        features = {}
        
        date = pd.to_datetime(txn['date'])
        amount = txn['amount']
        
        # Recent spending velocity (last 7 days vs previous 7 days)
        recent_7d = user_history[
            pd.to_datetime(user_history['date']) >= date - timedelta(days=7)
        ]
        prev_7d = user_history[
            (pd.to_datetime(user_history['date']) >= date - timedelta(days=14)) &
            (pd.to_datetime(user_history['date']) < date - timedelta(days=7))
        ]
        
        recent_spend = recent_7d['amount'].sum() if len(recent_7d) > 0 else 0
        prev_spend = prev_7d['amount'].sum() if len(prev_7d) > 0 else 0
        
        # Velocity (rate of change)
        if prev_spend > 0:
            features['spending_velocity_7d'] = (recent_spend - prev_spend) / prev_spend
        else:
            features['spending_velocity_7d'] = 0.0
        
        # Transaction frequency velocity
        recent_count = len(recent_7d)
        prev_count = len(prev_7d)
        if prev_count > 0:
            features['txn_frequency_velocity'] = (recent_count - prev_count) / prev_count
        else:
            features['txn_frequency_velocity'] = 0.0
        
        # Spending acceleration (30d window)
        features['spending_acceleration'] = self._compute_acceleration(user_history, date)
        
        # Burst detection (sudden increase in frequency)
        features['is_spending_burst'] = 1 if recent_count > prev_count * 2 else 0
        
        # Trend features
        features['spending_trend_7d'] = 1 if recent_spend > prev_spend else -1
        features['spending_trend_30d'] = self._compute_trend(user_history, date, days=30)
        
        return features
    
    def _extract_sequence_features(self, txn: pd.Series, user_history: pd.DataFrame) -> Dict[str, float]:
        """
        Extract sequence features (transaction chains and patterns).
        Captures sequential dependencies between transactions.
        """
        features = {}
        
        date = pd.to_datetime(txn['date'])
        amount = txn['amount']
        merchant = str(txn.get('merchant', ''))
        
        # Recent transaction sequence (last 5 transactions)
        recent_txns = user_history[
            pd.to_datetime(user_history['date']) < date
        ].tail(5)
        
        if len(recent_txns) > 0:
            # Amount sequence patterns
            recent_amounts = recent_txns['amount'].values
            features['amount_seq_mean'] = np.mean(recent_amounts)
            features['amount_seq_std'] = np.std(recent_amounts) if len(recent_amounts) > 1 else 0
            features['amount_seq_trend'] = self._sequence_trend(recent_amounts)
            
            # Time gap patterns
            recent_dates = pd.to_datetime(recent_txns['date'])
            time_gaps = recent_dates.diff().dt.total_seconds() / 3600  # Hours
            time_gaps = time_gaps.dropna()
            
            if len(time_gaps) > 0:
                features['avg_time_gap_hours'] = time_gaps.mean()
                features['time_gap_regularity'] = 1.0 / (time_gaps.std() + 1)  # Higher = more regular
            else:
                features['avg_time_gap_hours'] = 24.0
                features['time_gap_regularity'] = 0.0
            
            # Merchant sequence patterns
            recent_merchants = recent_txns['merchant'].values
            features['merchant_repeat_rate'] = len([m for m in recent_merchants if m == merchant]) / len(recent_merchants)
            features['merchant_diversity'] = len(set(recent_merchants)) / len(recent_merchants)
            
            # Transaction type sequence
            if 'type' in recent_txns.columns:
                recent_types = recent_txns['type'].values
                features['consecutive_debits'] = self._count_consecutive(recent_types, txn.get('type', 'debit'))
        else:
            features.update({
                'amount_seq_mean': amount,
                'amount_seq_std': 0.0,
                'amount_seq_trend': 0.0,
                'avg_time_gap_hours': 24.0,
                'time_gap_regularity': 0.0,
                'merchant_repeat_rate': 0.0,
                'merchant_diversity': 1.0,
                'consecutive_debits': 1
            })
        
        return features
    
    def _extract_network_features(self, txn: pd.Series, user_history: pd.DataFrame) -> Dict[str, float]:
        """
        Extract network features (merchant relationships).
        Captures co-occurrence patterns between merchants.
        """
        features = {}
        
        merchant = str(txn.get('merchant', ''))
        date = pd.to_datetime(txn['date'])
        
        # Build merchant co-occurrence network
        recent_window = user_history[
            pd.to_datetime(user_history['date']) >= date - timedelta(days=30)
        ]
        
        if len(recent_window) > 1:
            merchants = recent_window['merchant'].values
            
            # Update co-occurrence graph
            for i in range(len(merchants) - 1):
                self.merchant_graph[merchants[i]].add(merchants[i+1])
                self.merchant_graph[merchants[i+1]].add(merchants[i])
            
            # Network centrality (how connected is this merchant)
            if merchant in self.merchant_graph:
                features['merchant_degree'] = len(self.merchant_graph[merchant])
                features['merchant_centrality'] = min(features['merchant_degree'] / 10, 1.0)
            else:
                features['merchant_degree'] = 0
                features['merchant_centrality'] = 0.0
            
            # Co-occurrence with recent merchants
            recent_merchants = set(merchants[-5:])
            if merchant in self.merchant_graph:
                common_neighbors = len(recent_merchants & self.merchant_graph[merchant])
                features['merchant_cooccurrence'] = common_neighbors / len(recent_merchants)
            else:
                features['merchant_cooccurrence'] = 0.0
        else:
            features['merchant_degree'] = 0
            features['merchant_centrality'] = 0.0
            features['merchant_cooccurrence'] = 0.0
        
        return features
    
    def _extract_anomaly_features(self, txn: pd.Series, user_history: pd.DataFrame) -> Dict[str, float]:
        """
        Extract anomaly detection features.
        Identifies unusual transactions.
        """
        features = {}
        
        amount = txn['amount']
        date = pd.to_datetime(txn['date'])
        hour = date.hour if hasattr(date, 'hour') else 12
        
        # Amount anomaly (using IQR method)
        if len(user_history) > 10:
            q1 = user_history['amount'].quantile(0.25)
            q3 = user_history['amount'].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            features['is_amount_outlier'] = 1 if (amount < lower_bound or amount > upper_bound) else 0
            features['amount_outlier_score'] = max(
                (lower_bound - amount) / (iqr + 1),
                (amount - upper_bound) / (iqr + 1),
                0
            )
        else:
            features['is_amount_outlier'] = 0
            features['amount_outlier_score'] = 0.0
        
        # Time anomaly (unusual hour for this user)
        if len(user_history) > 0:
            hour_counts = Counter(pd.to_datetime(user_history['date']).dt.hour)
            total_txns = sum(hour_counts.values())
            hour_freq = hour_counts.get(hour, 0) / total_txns
            
            features['hour_frequency'] = hour_freq
            features['is_unusual_hour'] = 1 if hour_freq < 0.05 else 0
        else:
            features['hour_frequency'] = 0.1
            features['is_unusual_hour'] = 0
        
        # Merchant novelty
        merchant = str(txn.get('merchant', ''))
        if len(user_history) > 0:
            merchant_counts = Counter(user_history['merchant'])
            features['merchant_novelty'] = 1.0 / (merchant_counts.get(merchant, 0) + 1)
        else:
            features['merchant_novelty'] = 1.0
        
        return features
    
    def _extract_category_hints(self, txn: pd.Series, user_history: pd.DataFrame) -> Dict[str, float]:
        """
        Extract category hint features based on patterns.
        Soft signals that suggest likely categories.
        """
        features = {}
        
        amount = txn['amount']
        date = pd.to_datetime(txn['date'])
        hour = date.hour if hasattr(date, 'hour') else 12
        day = date.dayofweek
        
        # Food & Dining hints
        features['hint_food'] = (
            (100 <= amount <= 2000) * 0.3 +  # Typical food amount
            (11 <= hour <= 14 or 18 <= hour <= 22) * 0.3 +  # Meal times
            (day >= 5) * 0.2  # Weekend dining
        )
        
        # Commute hints
        features['hint_commute'] = (
            (50 <= amount <= 500) * 0.3 +  # Typical commute amount
            ((7 <= hour <= 10) or (17 <= hour <= 20)) * 0.4 +  # Commute times
            (day < 5) * 0.2  # Weekday commute
        )
        
        # Subscription hints
        features['hint_subscription'] = (
            (50 <= amount <= 1500) * 0.3 +  # Typical subscription
            (date.day <= 5 or date.day >= 25) * 0.3 +  # Month start/end
            self._is_recurring(txn, user_history) * 0.4
        )
        
        # Bills hints
        features['hint_bills'] = (
            (500 <= amount <= 5000) * 0.3 +  # Typical bill amount
            (date.day <= 10 or date.day >= 25) * 0.4 +  # Bill payment dates
            self._is_recurring(txn, user_history) * 0.3
        )
        
        # Salary hints
        features['hint_salary'] = (
            (amount >= 15000) * 0.4 +  # High amount
            (date.day <= 5 or date.day >= 28) * 0.4 +  # Salary dates
            (txn.get('type', '') == 'credit') * 0.2
        )
        
        return features
    
    def _default_advanced_features(self) -> Dict[str, float]:
        """Default values for advanced features (cold start)."""
        return {
            # Velocity
            'spending_velocity_7d': 0.0,
            'txn_frequency_velocity': 0.0,
            'spending_acceleration': 0.0,
            'is_spending_burst': 0,
            'spending_trend_7d': 0,
            'spending_trend_30d': 0.0,
            # Sequence
            'amount_seq_mean': 0.0,
            'amount_seq_std': 0.0,
            'amount_seq_trend': 0.0,
            'avg_time_gap_hours': 24.0,
            'time_gap_regularity': 0.0,
            'merchant_repeat_rate': 0.0,
            'merchant_diversity': 1.0,
            'consecutive_debits': 1,
            # Network
            'merchant_degree': 0,
            'merchant_centrality': 0.0,
            'merchant_cooccurrence': 0.0,
            # Anomaly
            'is_amount_outlier': 0,
            'amount_outlier_score': 0.0,
            'hour_frequency': 0.1,
            'is_unusual_hour': 0,
            'merchant_novelty': 1.0,
            # Category hints
            'hint_food': 0.0,
            'hint_commute': 0.0,
            'hint_subscription': 0.0,
            'hint_bills': 0.0,
            'hint_salary': 0.0
        }
    
    # Helper methods
    def _percentile(self, value: float, series: pd.Series) -> float:
        """Compute percentile of value in series."""
        return (series < value).sum() / len(series) if len(series) > 0 else 0.5
    
    def _zscore(self, value: float, series: pd.Series) -> float:
        """Compute z-score of value in series."""
        mean = series.mean()
        std = series.std()
        return (value - mean) / std if std > 0 else 0.0
    
    def _compute_acceleration(self, history: pd.DataFrame, current_date: datetime) -> float:
        """Compute spending acceleration (second derivative)."""
        # Get spending in 3 windows: 0-10d, 10-20d, 20-30d
        windows = []
        for i in range(3):
            start = current_date - timedelta(days=(i+1)*10)
            end = current_date - timedelta(days=i*10)
            window = history[
                (pd.to_datetime(history['date']) >= start) &
                (pd.to_datetime(history['date']) < end)
            ]
            windows.append(window['amount'].sum() if len(window) > 0 else 0)
        
        # Compute acceleration
        if windows[1] > 0:
            velocity1 = (windows[0] - windows[1]) / windows[1]
            velocity2 = (windows[1] - windows[2]) / windows[1]
            return velocity1 - velocity2
        return 0.0
    
    def _compute_trend(self, history: pd.DataFrame, current_date: datetime, days: int = 30) -> float:
        """Compute spending trend using linear regression."""
        window = history[
            pd.to_datetime(history['date']) >= current_date - timedelta(days=days)
        ]
        
        if len(window) < 3:
            return 0.0
        
        # Fit linear trend
        dates = pd.to_datetime(window['date'])
        x = (dates - dates.min()).dt.days.values
        y = window['amount'].values
        
        if len(x) > 1:
            slope, _, _, _, _ = stats.linregress(x, y)
            return np.tanh(slope / 100)  # Normalize to [-1, 1]
        return 0.0
    
    def _sequence_trend(self, values: np.ndarray) -> float:
        """Compute trend in sequence."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return np.tanh(slope / 100)
    
    def _count_consecutive(self, sequence: np.ndarray, value: str) -> int:
        """Count consecutive occurrences of value from end."""
        count = 0
        for item in reversed(sequence):
            if item == value:
                count += 1
            else:
                break
        return count
    
    def _is_recurring(self, txn: pd.Series, history: pd.DataFrame) -> float:
        """Check if transaction is recurring (0-1 score)."""
        merchant = str(txn.get('merchant', ''))
        amount = txn['amount']
        
        if len(history) < 2:
            return 0.0
        
        similar = history[
            (history['merchant'] == merchant) &
            (history['amount'] >= amount * 0.9) &
            (history['amount'] <= amount * 1.1)
        ]
        
        if len(similar) < 2:
            return 0.0
        
        # Check regularity
        dates = pd.to_datetime(similar['date']).sort_values()
        gaps = dates.diff().dt.days.dropna()
        
        if len(gaps) > 0:
            avg_gap = gaps.mean()
            std_gap = gaps.std()
            
            # Monthly recurring (25-35 days)
            if 25 <= avg_gap <= 35 and std_gap < 5:
                return 1.0
            # Weekly recurring (6-8 days)
            elif 6 <= avg_gap <= 8 and std_gap < 2:
                return 0.9
            # Some regularity
            elif std_gap < avg_gap * 0.3:
                return 0.5
        
        return 0.0

