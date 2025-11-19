import torch
from transformers import pipeline
from typing import Tuple, Optional, Dict, List
import pandas as pd

class ZeroShotClassifier:
    """
    Layer 8: Zero-shot classification using BART-MNLI.
    Fallback layer when semantic and behavioral methods fail.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize zero-shot classifier.
        
        Args:
            model_name: HuggingFace model for zero-shot classification
                       Default: facebook/bart-large-mnli
        """
        self.model_name = model_name
        self.classifier = None
        self.categories = [
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
            'Subscriptions'
        ]
        
        # Category hypothesis templates
        self.hypothesis_templates = {
            'Food & Dining': 'This transaction is for food, dining, restaurants, or groceries',
            'Commute/Transport': 'This transaction is for transportation, commute, taxi, or fuel',
            'Shopping': 'This transaction is for shopping, retail, or online purchases',
            'Bills & Utilities': 'This transaction is for bills, utilities, electricity, water, or internet',
            'Entertainment': 'This transaction is for entertainment, movies, events, or recreation',
            'Healthcare': 'This transaction is for healthcare, medical, pharmacy, or hospital',
            'Education': 'This transaction is for education, courses, books, or tuition',
            'Investments': 'This transaction is for investments, mutual funds, stocks, or SIP',
            'Salary/Income': 'This transaction is salary, income, or payment received',
            'Transfers': 'This transaction is a money transfer, payment to person, or P2P transfer',
            'Subscriptions': 'This transaction is a subscription service like Netflix, Spotify, or gym'
        }
    
    def _load_model(self):
        """Lazy load the model (only when needed)."""
        if self.classifier is None:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1  # Use CPU (-1), change to 0 for GPU
            )
    
    def classify(self, 
                description: str, 
                merchant: str = '', 
                amount: float = None,
                txn_type: str = '',
                recipient_name: str = None,
                upi_id: str = None,
                note: str = None) -> Tuple[Optional[str], float, Dict]:
        """
        Zero-shot classification of transaction.
        ROBUST: Prioritizes recipient_name, upi_id, and note fields.
        
        Args:
            description: Transaction description
            merchant: Merchant name
            amount: Transaction amount (optional, for context)
            txn_type: Transaction type (credit/debit)
            recipient_name: UPI recipient name (PRIORITY)
            upi_id: UPI ID (PRIORITY)
            note: Transaction note (PRIORITY)
            
        Returns:
            (category, confidence, provenance)
        """
        if not any([description, merchant, recipient_name, upi_id, note]):
            return None, 0.0, {'reason': 'No text to classify'}
        
        # Load model if not loaded
        self._load_model()
        
        # Construct premise (what we're classifying) - ROBUST with UPI fields
        premise = self._construct_premise(description, merchant, amount, txn_type, 
                                         recipient_name, upi_id, note)
        
        try:
            # Run zero-shot classification
            result = self.classifier(
                premise,
                candidate_labels=self.categories,
                hypothesis_template="This transaction is about {}."
            )
            
            # Get top prediction
            top_category = result['labels'][0]
            top_score = result['scores'][0]
            
            # Get top 3 for provenance
            top3 = list(zip(result['labels'][:3], result['scores'][:3]))
            
            # Confidence threshold
            if top_score >= 0.85:
                return top_category, top_score, {
                    'method': 'zero_shot_high_confidence',
                    'premise': premise,
                    'top_3': top3,
                    'reason': f'Zero-shot classification with high confidence ({top_score:.2%})'
                }
            elif top_score >= 0.60:
                return top_category, top_score, {
                    'method': 'zero_shot_moderate_confidence',
                    'premise': premise,
                    'top_3': top3,
                    'reason': f'Zero-shot classification with moderate confidence ({top_score:.2%})'
                }
            else:
                return None, top_score, {
                    'method': 'zero_shot_low_confidence',
                    'premise': premise,
                    'top_3': top3,
                    'reason': f'Zero-shot confidence too low ({top_score:.2%})'
                }
        
        except Exception as e:
            return None, 0.0, {
                'method': 'zero_shot_error',
                'reason': f'Zero-shot classification failed: {str(e)}'
            }
    
    def classify_with_nli(self, 
                         description: str,
                         merchant: str = '',
                         amount: float = None,
                         txn_type: str = '',
                         recipient_name: str = None,
                         upi_id: str = None,
                         note: str = None) -> Tuple[Optional[str], float, Dict]:
        """
        Alternative: Use NLI (Natural Language Inference) approach.
        Tests entailment between transaction and each category hypothesis.
        ROBUST: Prioritizes recipient_name, upi_id, and note fields.
        """
        if not any([description, merchant, recipient_name, upi_id, note]):
            return None, 0.0, {'reason': 'No text to classify'}
        
        self._load_model()
        
        # Construct premise - ROBUST with UPI fields
        premise = self._construct_premise(description, merchant, amount, txn_type,
                                         recipient_name, upi_id, note)
        
        try:
            # Test entailment for each category
            scores = {}
            
            for category, hypothesis in self.hypothesis_templates.items():
                result = self.classifier(
                    premise,
                    candidate_labels=[category],
                    hypothesis_template=hypothesis
                )
                scores[category] = result['scores'][0]
            
            # Get best match
            best_category = max(scores, key=scores.get)
            best_score = scores[best_category]
            
            # Sort for top 3
            top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if best_score >= 0.85:
                return best_category, best_score, {
                    'method': 'nli_entailment',
                    'premise': premise,
                    'top_3': top3,
                    'reason': f'NLI entailment with high confidence ({best_score:.2%})'
                }
            else:
                return None, best_score, {
                    'method': 'nli_low_confidence',
                    'premise': premise,
                    'top_3': top3,
                    'reason': f'NLI confidence too low ({best_score:.2%})'
                }
        
        except Exception as e:
            return None, 0.0, {
                'method': 'nli_error',
                'reason': f'NLI classification failed: {str(e)}'
            }
    
    def _construct_premise(self, 
                          description: str,
                          merchant: str,
                          amount: float,
                          txn_type: str,
                          recipient_name: str = None,
                          upi_id: str = None,
                          note: str = None) -> str:
        """
        Construct premise text from transaction details.
        ROBUST: Prioritizes recipient_name, upi_id, and note fields FIRST.
        """
        parts = []
        
        # Priority 1: RECIPIENT_NAME (most informative for UPI)
        if recipient_name and recipient_name not in ['', 'nan', 'none', 'None']:
            parts.append(f"Recipient: {recipient_name}")
        
        # Priority 2: NOTE (provides category context)
        if note and note not in ['', 'nan', 'none', 'None', 'upi', 'UPI']:
            parts.append(f"Note: {note}")
        
        # Priority 3: UPI_ID (contains merchant identifier)
        if upi_id and upi_id not in ['', 'nan', 'none', 'None']:
            # Extract alphabetic part for readability
            upi_base = ''.join([c for c in str(upi_id) if c.isalpha()])
            if upi_base and len(upi_base) > 3:
                parts.append(f"UPI ID: {upi_base}")
        
        # Priority 4: Merchant (fallback)
        if merchant and merchant not in ['', 'nan', 'none']:
            parts.append(f"Merchant: {merchant}")
        
        # Priority 5: Description (if different from merchant)
        if description and description not in ['', 'nan', 'none'] and description != merchant:
            parts.append(f"Description: {description}")
        
        # Priority 6: Amount
        if amount is not None:
            parts.append(f"Amount: ₹{amount:.2f}")
        
        # Priority 7: Transaction type
        if txn_type:
            parts.append(f"Type: {txn_type}")
        
        return ". ".join(parts) if parts else "Transaction"
    
    def batch_classify(self, transactions: List[Dict]) -> List[Tuple[Optional[str], float, Dict]]:
        """
        Classify multiple transactions in batch.
        ROBUST: Includes UPI fields in classification.
        
        Args:
            transactions: List of dicts with keys: description, merchant, amount, type,
                         recipient_name, upi_id, note
            
        Returns:
            List of (category, confidence, provenance) tuples
        """
        results = []
        
        for txn in transactions:
            result = self.classify(
                description=txn.get('description', ''),
                merchant=txn.get('merchant', ''),
                amount=txn.get('amount'),
                txn_type=txn.get('type', ''),
                recipient_name=txn.get('Recipient_Name', txn.get('recipient_name')),
                upi_id=txn.get('UPI_ID', txn.get('upi_id')),
                note=txn.get('Note', txn.get('note'))
            )
            results.append(result)
        
        return results
    
    def clear_cache(self):
        """Clear model from memory."""
        if self.classifier is not None:
            del self.classifier
            self.classifier = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

