import re
from typing import Dict, Tuple, Optional
from rapidfuzz import fuzz
import pandas as pd

class TextNormalizer:
    def __init__(self):
        # Enhanced noise patterns for the dataset
        self.noise_patterns = [
            r'upi', r'neft', r'imps', r'ref', r'txn', r'auth', 
            r'payment', r'paytm', r'\d{6,}', r'ach\d+', r'inb\d+',
            r'atm\d+', r'pos'
        ]
        
        # Description prefixes that indicate transaction types
        self.description_prefixes = {
            'SALARY-NEFT': 'Salary/Income',
            'INB-CREDIT': 'Salary/Income',
            'AUTO-DEBIT': 'Subscriptions',
            'SUBS': 'Subscriptions',
            'BILL-PAY': 'Bills & Utilities',
            'RECHARGE': 'Bills & Utilities',
            'ATM-CASH-WDL': 'Transfers',
            'NEFT-TRF': 'Transfers',
            'IMPS-TO': 'Transfers',
            'MANUAL-REVIEW': 'Transfers',
            'PENDING-REVIEW': 'Transfers',
            'TICKET': 'Commute/Transport',
            'IRCTC': 'Commute/Transport',
            'RIDE': 'Commute/Transport',
            'POS': None,  # POS can be anything, depends on merchant
            'ENT': 'Entertainment'
        }
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
            'imagicaa': ['imagicaa', 'imagica'],
            'indianr': ['indianr', 'indian railways'],
            'irctc': ['irctc', 'irctcuts', 'indian rail'],
            'snowcre': ['snowcre', 'snowcreat'],
            'julfikar': ['julfikar', 'julfika'],
            'bikaner': ['bikaner', 'bikanervala'],
            'airtel': ['airtel', 'bharti', 'bhartia'],
            'groupon': ['groupon'],
            'amul': ['amul'],
            'mcdonalds': ['mcdonalds', 'mcdonald'],
            'pizzahut': ['pizzahut', 'pizza hut'],
            'jiocinema': ['jiocinema', 'jio cinema'],
            'relishr': ['relishr'],
            'restaura': ['restaura'],
            'hotelha': ['hotelha'],
            'cakedine': ['cakedine'],
            'dailyfr': ['dailyfr']
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
            'irctc': 'Commute/Transport',
            'bikaner': 'Food & Dining',
            'julfikar': 'Food & Dining',
            'airtel': 'Bills & Utilities',
            'groupon': 'Entertainment',
            'amul': 'Food & Dining',
            'mcdonalds': 'Food & Dining',
            'pizzahut': 'Food & Dining',
            'jiocinema': 'Subscriptions',
            'relishr': 'Food & Dining',
            'restaura': 'Food & Dining',
            'hotelha': 'Food & Dining',
            'cakedine': 'Food & Dining',
            'dailyfr': 'Food & Dining'
        }
    
    def normalize(self, text: str, recipient_name: str = None, upi_id: str = None, 
                  note: str = None) -> Tuple[str, Dict]:
        """
        Normalize merchant/description text with enhanced UPI-based data.
        ROBUST: Prioritizes recipient_name, upi_id, and note fields FIRST.
        ENHANCED: Extracts merchant from Description patterns like "IMPS-TO-NEFT10453910-BhartiA"
        Returns: (normalized_text, metadata)
        
        Args:
            text: Primary description/merchant field (e.g., "UPI-PAYMENT-REF43626934-Amul")
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
        
        # ENHANCED: Extract merchant from Description field
        # Pattern: "IMPS-TO-NEFT10453910-BhartiA" → extract "BhartiA"
        # Pattern: "UPI-PAYMENT-REF43626934-Amul" → extract "Amul"
        extracted_merchant, txn_type_hint = self._extract_merchant_from_description(text)
        
        # ROBUST STRATEGY: Check UPI fields FIRST (they're more reliable)
        canonical = None
        confidence = 0.0
        matched_source = ''
        
        # Priority 1: Check RECIPIENT_NAME first (most reliable)
        # If no recipient_name but we extracted from Description, use that
        if not recipient and extracted_merchant:
            recipient = extracted_merchant.lower()
        
        if recipient:
            recipient_canonical, recipient_conf = self._match_canonical(recipient)
            if recipient_canonical and recipient_conf > confidence:
                canonical = recipient_canonical
                confidence = recipient_conf
                matched_source = 'recipient_name' if not extracted_merchant else 'description_extracted'
        
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
            # ENHANCED: Combine sources with UPI fields FIRST for rich context
            # Priority: recipient_name > note > upi_id > original_text
            # This ensures we capture the most informative parts
            combined_parts = []
            
            if recipient:
                combined_parts.append(recipient)
            if note_text:
                combined_parts.append(note_text)
            if upi:
                combined_parts.append(upi)
            if text:
                combined_parts.append(text.lower())
            
            combined_text = ' '.join(combined_parts).strip()
            text_normalized = combined_text.lower().strip()
            
            # Extract VPA/UPI ID
            vpa = self._extract_vpa(text_normalized) or upi
            
            # ENHANCED: Add context keywords from note field
            # e.g., if note says "baker", add it to help semantic understanding
            context_keywords = self._extract_context_keywords(note_text, recipient)
            
            # Remove noise patterns (but preserve context)
            for pattern in self.noise_patterns:
                text_normalized = re.sub(pattern, '', text_normalized, flags=re.IGNORECASE)
            
            # Keep only alphanumeric and @ . characters
            text_normalized = re.sub(r'[^a-z0-9@\.]', ' ', text_normalized)
            
            # Keep tokens > 2 chars
            tokens = [t for t in text_normalized.split() if len(t) > 2]
            
            # Add context keywords to enrich understanding
            if context_keywords:
                tokens.extend(context_keywords)
            
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
            'has_note': bool(note_text),
            'extracted_merchant': extracted_merchant,  # NEW: Merchant from description
            'txn_type_hint': txn_type_hint  # NEW: Transaction type hint from description prefix
        }
        
        return text if not canonical else canonical, metadata
    
    def _extract_vpa(self, text: str) -> Optional[str]:
        """Extract UPI VPA (Virtual Payment Address)."""
        vpa_pattern = r'\w+@[\w\.]+'
        match = re.search(vpa_pattern, text)
        return match.group(0) if match else None
    
    def _extract_merchant_from_description(self, description: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract merchant name from Description field patterns.
        
        Examples:
        - "IMPS-TO-NEFT10453910-BhartiA" → ("BhartiA", "Transfers")
        - "UPI-PAYMENT-REF43626934-Amul" → ("Amul", None)
        - "SALARY-NEFT-IMPS65087917-ACME PAYROLL" → ("ACME PAYROLL", "Salary/Income")
        - "ATM-CASH-WDL-ATM71827-ATM" → ("ATM", "Transfers")
        - "TICKET-IMPS59732292-IRCTCUTS" → ("IRCTCUTS", "Commute/Transport")
        
        Returns:
            (merchant_name, category_hint) or (None, None)
        """
        if not description or pd.isna(description):
            return None, None
        
        description = str(description).strip()
        
        # Pattern: PREFIX-REFID-MERCHANT
        # Look for the last part after the final dash
        parts = description.split('-')
        
        if len(parts) < 2:
            return None, None
        
        # Get potential merchant (last part)
        merchant = parts[-1].strip()
        
        # Skip if it's just numbers or reference codes
        if merchant.isdigit() or len(merchant) < 3:
            return None, None
        
        # Check if first part matches a known prefix
        prefix = parts[0].upper()
        category_hint = self.description_prefixes.get(prefix)
        
        # Also check for two-part prefixes like "SALARY-NEFT"
        if len(parts) >= 3:
            two_part_prefix = f"{parts[0].upper()}-{parts[1].upper()}"
            if two_part_prefix in self.description_prefixes:
                category_hint = self.description_prefixes[two_part_prefix]
        
        # Clean merchant name
        merchant = merchant.upper()  # Keep uppercase for consistency
        
        # Skip generic terms
        generic_terms = ['ATM', 'UPI', 'NEFT', 'IMPS', 'ACH', 'INB', 'POS', 'REF', 'TXN']
        if merchant in generic_terms:
            return None, category_hint
        
        return merchant, category_hint
    
    def _extract_context_keywords(self, note: str, recipient: str) -> list:
        """
        Extract contextual keywords from note and recipient fields.
        This helps semantic understanding by adding profession/business type hints.
        
        Example: 
        - note="baker" → adds "bakery", "food", "bread"
        - note="doctor" → adds "medical", "healthcare", "clinic"
        """
        context_map = {
            'baker': ['bakery', 'food', 'bread', 'pastry'],
            'bakery': ['baker', 'food', 'bread', 'pastry'],
            'doctor': ['medical', 'healthcare', 'clinic', 'health'],
            'medical': ['doctor', 'healthcare', 'clinic', 'health'],
            'clinic': ['doctor', 'healthcare', 'medical', 'health'],
            'hospital': ['medical', 'healthcare', 'health'],
            'cafe': ['coffee', 'food', 'dining', 'restaurant'],
            'coffee': ['cafe', 'food', 'beverage', 'dining'],
            'restaurant': ['food', 'dining', 'meal', 'eatery'],
            'gym': ['fitness', 'exercise', 'health', 'workout'],
            'fitness': ['gym', 'exercise', 'health', 'workout'],
            'salon': ['beauty', 'hair', 'grooming'],
            'parlour': ['beauty', 'salon', 'grooming'],
            'tutor': ['education', 'teaching', 'classes', 'learning'],
            'teacher': ['education', 'teaching', 'classes', 'tuition'],
            'coaching': ['education', 'classes', 'tuition', 'learning'],
            'driver': ['transport', 'travel', 'taxi', 'ride'],
            'cab': ['transport', 'travel', 'taxi', 'ride'],
            'taxi': ['transport', 'travel', 'cab', 'ride'],
            'electrician': ['repair', 'service', 'maintenance', 'home'],
            'plumber': ['repair', 'service', 'maintenance', 'home'],
            'mechanic': ['repair', 'service', 'vehicle', 'car'],
            'tailor': ['clothing', 'apparel', 'fashion', 'garment'],
            'laundry': ['cleaning', 'clothes', 'washing', 'service'],
            'milk': ['dairy', 'food', 'grocery', 'beverage'],
            'dairy': ['milk', 'food', 'grocery'],
            'vegetable': ['grocery', 'food', 'produce', 'market'],
            'grocery': ['food', 'market', 'shopping', 'supermarket'],
            'pharmacy': ['medical', 'medicine', 'healthcare', 'chemist'],
            'chemist': ['pharmacy', 'medical', 'medicine', 'healthcare'],
            'book': ['education', 'stationery', 'shopping', 'reading'],
            'stationery': ['office', 'supplies', 'shopping', 'writing']
        }
        
        keywords = []
        
        # Check note field
        if note:
            note_lower = note.lower().strip()
            if note_lower in context_map:
                keywords.extend(context_map[note_lower])
        
        # Check recipient field for business type hints
        if recipient:
            recipient_lower = recipient.lower().strip()
            for key in context_map:
                if key in recipient_lower:
                    keywords.extend(context_map[key])
                    break
        
        # Return unique keywords
        return list(set(keywords))
    
    def _match_canonical(self, text: str) -> Tuple[Optional[str], float]:
        """
        Match text against canonical merchant names.
        FIXED: Much stricter matching to prevent false positives like "burgerking" → subscription.
        
        Strategy:
        1. Exact match (100% confidence)
        2. Exact substring match with word boundaries (98% confidence)
        3. Fuzzy match with STRICT 95%+ threshold using ratio (not partial_ratio)
        """
        best_match = None
        best_score = 0
        best_method = None
        
        text_lower = text.lower().strip()
        
        for canonical, aliases in self.canonical_aliases.items():
            for alias in aliases:
                alias_lower = alias.lower().strip()
                
                # Strategy 1: Exact match (highest confidence)
                if text_lower == alias_lower:
                    if 100 > best_score:
                        best_score = 100
                        best_match = canonical
                        best_method = 'exact'
                    continue
                
                # Strategy 2: Exact substring with word boundaries
                # Ensures "burger" doesn't match "burgerking"
                if alias_lower in text_lower:
                    # Check if it's a complete word (not part of another word)
                    alias_start = text_lower.find(alias_lower)
                    alias_end = alias_start + len(alias_lower)
                    
                    # Check boundaries
                    is_word_start = alias_start == 0 or not text_lower[alias_start-1].isalnum()
                    is_word_end = alias_end == len(text_lower) or not text_lower[alias_end].isalnum()
                    
                    if is_word_start and is_word_end:
                        # Calculate match quality based on how much of text is the alias
                        match_ratio = len(alias_lower) / len(text_lower)
                        score = min(98, 90 + match_ratio * 8)  # 90-98 range
                        
                        if score > best_score:
                            best_score = score
                            best_match = canonical
                            best_method = 'substring'
                        continue
                
                # Strategy 3: Fuzzy match with STRICT 95%+ threshold
                # Use ratio (not partial_ratio) for precision
                fuzzy_score = fuzz.ratio(text_lower, alias_lower)
                
                if fuzzy_score >= 95:  # VERY strict threshold
                    if fuzzy_score > best_score:
                        best_score = fuzzy_score
                        best_match = canonical
                        best_method = 'fuzzy'
        
        # Only return if we have a confident match
        if best_score >= 95:
            return best_match, best_score / 100.0
        
        return None, 0.0
