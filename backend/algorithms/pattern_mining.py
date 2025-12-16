"""
Frequent and Maximal Pattern Mining Implementation.

This module implements a modified Apriori algorithm for discovering
frequent itemsets and extracting maximal patterns.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from itertools import combinations
import numpy as np
from ..utils.logger import get_logger
from ..utils.exceptions import PatternMiningError

logger = get_logger(__name__)


class FrequentPattern:
    """
    Represents a frequent pattern (itemset).
    
    Attributes:
        items: Frozenset of items in the pattern
        support: Absolute support count
        transactions: List of transaction IDs containing this pattern
    """
    
    def __init__(self, items: frozenset, support: int = 0, transactions: Optional[List[int]] = None):
        self.items = items
        self.support = support
        self.transactions = transactions or []
    
    def __repr__(self) -> str:
        return f"Pattern({self.items}, support={self.support})"
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __hash__(self) -> int:
        return hash(self.items)
    
    def __eq__(self, other) -> bool:
        return self.items == other.items


class FrequentPatternMiner:
    """
    Discovers frequent and maximal patterns using Apriori algorithm.
    
    The Apriori algorithm leverages the downward closure property:
    all subsets of a frequent itemset must also be frequent.
    
    Attributes:
        min_support: Minimum support threshold (absolute count or relative)
        max_pattern_length: Maximum length of patterns to mine
        frequent_patterns: Dictionary mapping pattern length to list of patterns
        maximal_patterns: List of maximal frequent patterns
    """
    
    def __init__(self, min_support: float = 0.05, max_pattern_length: int = 10):
        """
        Initialize the Frequent Pattern Miner.
        
        Args:
            min_support: Minimum support threshold (0-1 for relative, >1 for absolute)
            max_pattern_length: Maximum pattern length to consider
        
        Raises:
            PatternMiningError: If parameters are invalid
        """
        if min_support <= 0:
            raise PatternMiningError("Minimum support must be positive")
        
        if max_pattern_length < 1:
            raise PatternMiningError("Maximum pattern length must be at least 1")
        
        self.min_support = min_support
        self.max_pattern_length = max_pattern_length
        self.frequent_patterns: Dict[int, List[FrequentPattern]] = {}
        self.maximal_patterns: List[FrequentPattern] = []
        self.n_transactions = 0
        
        logger.info(
            f"Initialized FrequentPatternMiner with min_support={min_support}, "
            f"max_length={max_pattern_length}"
        )
    
    def fit(self, transactions: List[frozenset]) -> 'FrequentPatternMiner':
        """
        Mine frequent patterns from transactions.
        
        Args:
            transactions: List of transactions (each is a frozenset of items)
        
        Returns:
            self: Fitted miner instance
        
        Raises:
            PatternMiningError: If transactions are invalid
        """
        if not transactions:
            raise PatternMiningError("Cannot mine patterns from empty transaction list")
        
        self.n_transactions = len(transactions)
        
        # Convert relative support to absolute if needed
        if self.min_support <= 1.0:
            absolute_min_support = int(np.ceil(self.min_support * self.n_transactions))
        else:
            absolute_min_support = int(self.min_support)
        
        logger.info(
            f"Mining patterns from {self.n_transactions} transactions "
            f"(absolute min_support={absolute_min_support})"
        )
        
        # Phase 1: Find frequent 1-itemsets
        frequent_1_itemsets = self._find_frequent_1_itemsets(
            transactions, absolute_min_support
        )
        
        if not frequent_1_itemsets:
            logger.warning("No frequent 1-itemsets found. Consider lowering min_support.")
            return self
        
        self.frequent_patterns[1] = frequent_1_itemsets
        logger.info(f"Found {len(frequent_1_itemsets)} frequent 1-itemsets")
        
        # Phase 2: Generate frequent k-itemsets iteratively
        k = 2
        while k <= self.max_pattern_length:
            # Generate candidates from previous frequent patterns
            candidates = self._generate_candidates(self.frequent_patterns[k-1])
            
            if not candidates:
                logger.info(f"No more candidates at length {k}. Stopping.")
                break
            
            # Count support for candidates
            frequent_k_itemsets = self._count_support(
                candidates, transactions, absolute_min_support
            )
            
            if not frequent_k_itemsets:
                logger.info(f"No frequent {k}-itemsets found. Stopping.")
                break
            
            self.frequent_patterns[k] = frequent_k_itemsets
            logger.info(f"Found {len(frequent_k_itemsets)} frequent {k}-itemsets")
            
            k += 1
        
        # Phase 3: Extract maximal patterns
        self._extract_maximal_patterns()
        
        total_patterns = sum(len(patterns) for patterns in self.frequent_patterns.values())
        logger.info(
            f"Mining complete. Total frequent patterns: {total_patterns}, "
            f"Maximal patterns: {len(self.maximal_patterns)}"
        )
        
        return self
    
    def _find_frequent_1_itemsets(
        self, 
        transactions: List[frozenset], 
        min_support: int
    ) -> List[FrequentPattern]:
        """
        Find all frequent 1-itemsets (single items).
        
        Args:
            transactions: List of transactions
            min_support: Minimum absolute support
        
        Returns:
            List of frequent 1-itemsets
        """
        item_counts: Dict[str, Dict] = defaultdict(lambda: {'count': 0, 'transactions': []})
        
        # Count occurrence of each item
        for trans_id, transaction in enumerate(transactions):
            for item in transaction:
                item_counts[item]['count'] += 1
                item_counts[item]['transactions'].append(trans_id)
        
        # Filter by minimum support
        frequent = []
        for item, data in item_counts.items():
            if data['count'] >= min_support:
                pattern = FrequentPattern(
                    items=frozenset([item]),
                    support=data['count'],
                    transactions=data['transactions']
                )
                frequent.append(pattern)
        
        return frequent
    
    def _generate_candidates(self, frequent_k_minus_1: List[FrequentPattern]) -> List[frozenset]:
        """
        Generate candidate (k)-itemsets from frequent (k-1)-itemsets.
        
        Uses the join step: combine two (k-1)-itemsets that share (k-2) items.
        
        Args:
            frequent_k_minus_1: Frequent patterns of length k-1
        
        Returns:
            List of candidate k-itemsets
        """
        candidates = []
        n = len(frequent_k_minus_1)
        
        # Join step: combine patterns that differ by one item
        for i in range(n):
            for j in range(i + 1, n):
                pattern1 = frequent_k_minus_1[i].items
                pattern2 = frequent_k_minus_1[j].items
                
                # Union of two patterns
                union = pattern1 | pattern2
                
                # Check if they differ by exactly one item
                if len(union) == len(pattern1) + 1:
                    # Prune step: check if all (k-1)-subsets are frequent
                    if self._has_infrequent_subset(union, frequent_k_minus_1):
                        continue
                    
                    candidates.append(union)
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        return candidates
    
    def _has_infrequent_subset(
        self, 
        candidate: frozenset, 
        frequent_k_minus_1: List[FrequentPattern]
    ) -> bool:
        """
        Check if candidate has any infrequent (k-1)-subset.
        
        Implements the prune step of Apriori.
        
        Args:
            candidate: Candidate k-itemset
            frequent_k_minus_1: Frequent (k-1)-itemsets
        
        Returns:
            True if has infrequent subset, False otherwise
        """
        # Get all (k-1)-subsets
        k = len(candidate)
        subsets = [frozenset(s) for s in combinations(candidate, k - 1)]
        
        # Create set of frequent (k-1)-itemsets for fast lookup
        frequent_set = {pattern.items for pattern in frequent_k_minus_1}
        
        # Check if all subsets are frequent
        for subset in subsets:
            if subset not in frequent_set:
                return True  # Found infrequent subset
        
        return False
    
    def _count_support(
        self,
        candidates: List[frozenset],
        transactions: List[frozenset],
        min_support: int
    ) -> List[FrequentPattern]:
        """
        Count support for candidate itemsets and filter by minimum support.
        
        Args:
            candidates: List of candidate itemsets
            transactions: List of transactions
            min_support: Minimum absolute support
        
        Returns:
            List of frequent patterns meeting support threshold
        """
        # Initialize support counters
        candidate_data: Dict[frozenset, Dict] = {
            candidate: {'count': 0, 'transactions': []}
            for candidate in candidates
        }
        
        # Count support for each candidate
        for trans_id, transaction in enumerate(transactions):
            for candidate in candidates:
                # Check if candidate is subset of transaction
                if candidate.issubset(transaction):
                    candidate_data[candidate]['count'] += 1
                    candidate_data[candidate]['transactions'].append(trans_id)
        
        # Filter by minimum support
        frequent = []
        for candidate, data in candidate_data.items():
            if data['count'] >= min_support:
                pattern = FrequentPattern(
                    items=candidate,
                    support=data['count'],
                    transactions=data['transactions']
                )
                frequent.append(pattern)
        
        return frequent
    
    def _extract_maximal_patterns(self) -> None:
        """
        Extract maximal patterns from all frequent patterns.
        
        A pattern is maximal if it has no frequent superset.
        """
        # Collect all frequent patterns
        all_patterns = []
        for patterns in self.frequent_patterns.values():
            all_patterns.extend(patterns)
        
        # Sort by pattern length (descending) for efficiency
        all_patterns.sort(key=lambda p: len(p.items), reverse=True)
        
        maximal = []
        
        for pattern in all_patterns:
            is_maximal = True
            
            # Check if pattern is subset of any already identified maximal pattern
            for max_pattern in maximal:
                if pattern.items.issubset(max_pattern.items):
                    is_maximal = False
                    break
            
            if is_maximal:
                maximal.append(pattern)
        
        self.maximal_patterns = maximal
        logger.info(f"Extracted {len(maximal)} maximal patterns")
    
    def get_frequent_patterns(self, min_length: int = 1) -> List[FrequentPattern]:
        """
        Get all frequent patterns of minimum length.
        
        Args:
            min_length: Minimum pattern length
        
        Returns:
            List of frequent patterns
        """
        patterns = []
        for length, pattern_list in self.frequent_patterns.items():
            if length >= min_length:
                patterns.extend(pattern_list)
        return patterns
    
    def get_maximal_patterns(self) -> List[FrequentPattern]:
        """
        Get maximal frequent patterns.
        
        Returns:
            List of maximal patterns
        """
        return self.maximal_patterns
    
    def get_pattern_stats(self) -> Dict:
        """
        Get statistics about discovered patterns.
        
        Returns:
            Dictionary with pattern mining statistics
        """
        total_frequent = sum(len(p) for p in self.frequent_patterns.values())
        
        stats = {
            'n_transactions': self.n_transactions,
            'min_support_threshold': self.min_support,
            'total_frequent_patterns': total_frequent,
            'n_maximal_patterns': len(self.maximal_patterns),
            'patterns_by_length': {
                k: len(v) for k, v in self.frequent_patterns.items()
            },
            'maximal_pattern_lengths': [
                len(p.items) for p in self.maximal_patterns
            ]
        }
        
        if self.maximal_patterns:
            stats['avg_maximal_length'] = np.mean(stats['maximal_pattern_lengths'])
            stats['max_pattern_length'] = max(stats['maximal_pattern_lengths'])
            stats['min_pattern_length'] = min(stats['maximal_pattern_lengths'])
        
        return stats
    
    def print_patterns(self, pattern_type: str = 'maximal', top_n: int = 10) -> None:
        """
        Print discovered patterns in readable format.
        
        Args:
            pattern_type: Type of patterns to print ('frequent' or 'maximal')
            top_n: Number of patterns to print
        """
        if pattern_type == 'maximal':
            patterns = self.maximal_patterns
            title = "Maximal Patterns"
        else:
            patterns = self.get_frequent_patterns()
            title = "Frequent Patterns"
        
        print(f"\n{'='*60}")
        print(f"{title} (showing top {top_n})")
        print(f"{'='*60}")
        
        # Sort by support (descending)
        sorted_patterns = sorted(patterns, key=lambda p: p.support, reverse=True)
        
        for i, pattern in enumerate(sorted_patterns[:top_n], 1):
            support_pct = (pattern.support / self.n_transactions) * 100
            items_str = ', '.join(sorted(pattern.items))
            print(f"{i}. [{items_str}]")
            print(f"   Support: {pattern.support} ({support_pct:.1f}%)")
            print()
