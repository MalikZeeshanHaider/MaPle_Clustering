"""
MaPle Clustering Algorithm Implementation.

This module implements the core MaPle (Maximal Pattern-based Clustering) algorithm
that forms clusters based on maximal frequent patterns.
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from collections import defaultdict
from .pattern_mining import FrequentPattern, FrequentPatternMiner
from ..utils.logger import get_logger
from ..utils.exceptions import ClusteringError

logger = get_logger(__name__)


class Cluster:
    """
    Represents a cluster formed by a maximal pattern.
    
    Attributes:
        cluster_id: Unique cluster identifier
        pattern: Maximal pattern defining this cluster
        members: List of transaction IDs in this cluster
        centroid_pattern: Representative pattern (same as pattern for MaPle)
    """
    
    def __init__(self, cluster_id: int, pattern: FrequentPattern):
        self.cluster_id = cluster_id
        self.pattern = pattern
        self.members: List[int] = []
        self.centroid_pattern = pattern.items
    
    def add_member(self, transaction_id: int) -> None:
        """Add a transaction to this cluster."""
        self.members.append(transaction_id)
    
    def size(self) -> int:
        """Get cluster size."""
        return len(self.members)
    
    def __repr__(self) -> str:
        return f"Cluster(id={self.cluster_id}, size={self.size()}, pattern={self.pattern.items})"


class MaPleClusterer:
    """
    Maximal Pattern-based Clustering (MaPle) Algorithm.
    
    MaPle discovers clusters based on maximal frequent patterns rather than
    distance measures. Transactions sharing maximal patterns are grouped together.
    
    Process:
    1. Mine frequent patterns from transactions
    2. Extract maximal patterns
    3. Form clusters from maximal patterns
    4. Assign transactions to best-matching clusters
    5. Handle overlaps and outliers
    
    Attributes:
        min_support: Minimum support threshold for pattern mining
        max_pattern_length: Maximum pattern length
        overlap_threshold: Threshold for assigning to multiple clusters
        pattern_miner: FrequentPatternMiner instance
        clusters: Dictionary mapping cluster IDs to Cluster objects
        cluster_assignments: Dictionary mapping transaction IDs to cluster IDs
        outliers: List of transaction IDs not assigned to any cluster
    """
    
    def __init__(
        self,
        min_support: float = 0.05,
        max_pattern_length: int = 10,
        overlap_threshold: float = 0.5,
        assignment_strategy: str = 'best_match'
    ):
        """
        Initialize MaPle Clusterer.
        
        Args:
            min_support: Minimum support for frequent patterns (0-1 or absolute count)
            max_pattern_length: Maximum length of patterns to consider
            overlap_threshold: Threshold for multi-cluster assignment (0-1)
            assignment_strategy: Strategy for cluster assignment
                                ('best_match', 'all_matching', 'threshold')
        
        Raises:
            ClusteringError: If parameters are invalid
        """
        if min_support <= 0:
            raise ClusteringError("Minimum support must be positive")
        
        if overlap_threshold < 0 or overlap_threshold > 1:
            raise ClusteringError("Overlap threshold must be between 0 and 1")
        
        if assignment_strategy not in ['best_match', 'all_matching', 'threshold']:
            raise ClusteringError(f"Invalid assignment strategy: {assignment_strategy}")
        
        self.min_support = min_support
        self.max_pattern_length = max_pattern_length
        self.overlap_threshold = overlap_threshold
        self.assignment_strategy = assignment_strategy
        
        # Pattern miner instance
        self.pattern_miner = FrequentPatternMiner(
            min_support=min_support,
            max_pattern_length=max_pattern_length
        )
        
        # Clustering results
        self.clusters: Dict[int, Cluster] = {}
        self.cluster_assignments: Dict[int, int] = {}
        self.outliers: List[int] = []
        self.transactions: List[frozenset] = []
        
        logger.info(
            f"Initialized MaPleClusterer with min_support={min_support}, "
            f"strategy={assignment_strategy}"
        )
    
    def fit(self, transactions: List[frozenset]) -> 'MaPleClusterer':
        """
        Fit the MaPle clustering algorithm on transactions.
        
        Args:
            transactions: List of transactions (frozensets of items)
        
        Returns:
            self: Fitted clusterer instance
        
        Raises:
            ClusteringError: If clustering fails
        """
        if not transactions:
            raise ClusteringError("Cannot cluster empty transaction list")
        
        self.transactions = transactions
        logger.info(f"Starting MaPle clustering on {len(transactions)} transactions")
        
        # Step 1: Mine frequent and maximal patterns
        logger.info("Step 1: Mining frequent patterns...")
        self.pattern_miner.fit(transactions)
        
        maximal_patterns = self.pattern_miner.get_maximal_patterns()
        
        if not maximal_patterns:
            logger.warning("No maximal patterns found. All transactions will be outliers.")
            self.outliers = list(range(len(transactions)))
            return self
        
        logger.info(f"Found {len(maximal_patterns)} maximal patterns")
        
        # Step 2: Create clusters from maximal patterns
        logger.info("Step 2: Forming clusters from maximal patterns...")
        self._create_clusters(maximal_patterns)
        
        # Step 3: Assign transactions to clusters
        logger.info("Step 3: Assigning transactions to clusters...")
        self._assign_transactions()
        
        # Step 4: Remove empty clusters
        self._remove_empty_clusters()
        
        # Log results
        logger.info(
            f"Clustering complete. Clusters: {len(self.clusters)}, "
            f"Outliers: {len(self.outliers)}"
        )
        
        return self
    
    def _create_clusters(self, maximal_patterns: List[FrequentPattern]) -> None:
        """
        Create cluster objects from maximal patterns.
        
        Each maximal pattern becomes a potential cluster.
        
        Args:
            maximal_patterns: List of maximal frequent patterns
        """
        self.clusters = {}
        
        # Sort patterns by support (descending) for better cluster quality
        sorted_patterns = sorted(maximal_patterns, key=lambda p: p.support, reverse=True)
        
        for cluster_id, pattern in enumerate(sorted_patterns):
            cluster = Cluster(cluster_id=cluster_id, pattern=pattern)
            self.clusters[cluster_id] = cluster
        
        logger.info(f"Created {len(self.clusters)} clusters from maximal patterns")
    
    def _assign_transactions(self) -> None:
        """
        Assign each transaction to best-matching cluster(s).
        
        Uses the configured assignment strategy.
        """
        self.cluster_assignments = {}
        self.outliers = []
        
        for trans_id, transaction in enumerate(self.transactions):
            assigned = False
            
            if self.assignment_strategy == 'best_match':
                assigned = self._assign_best_match(trans_id, transaction)
            elif self.assignment_strategy == 'all_matching':
                assigned = self._assign_all_matching(trans_id, transaction)
            elif self.assignment_strategy == 'threshold':
                assigned = self._assign_threshold(trans_id, transaction)
            
            if not assigned:
                self.outliers.append(trans_id)
    
    def _assign_best_match(self, trans_id: int, transaction: frozenset) -> bool:
        """
        Assign transaction to single best-matching cluster.
        
        Args:
            trans_id: Transaction ID
            transaction: Transaction items
        
        Returns:
            True if assigned, False otherwise
        """
        best_cluster_id = -1
        best_score = 0.0
        
        # Find best matching cluster
        for cluster_id, cluster in self.clusters.items():
            pattern = cluster.pattern.items
            
            # Check if pattern is subset of transaction
            if pattern.issubset(transaction):
                # Calculate matching score
                score = self._calculate_match_score(transaction, pattern)
                
                if score > best_score:
                    best_score = score
                    best_cluster_id = cluster_id
        
        # Assign to best cluster if found
        if best_cluster_id != -1:
            self.cluster_assignments[trans_id] = best_cluster_id
            self.clusters[best_cluster_id].add_member(trans_id)
            return True
        
        return False
    
    def _assign_all_matching(self, trans_id: int, transaction: frozenset) -> bool:
        """
        Assign transaction to all matching clusters (soft clustering).
        
        Args:
            trans_id: Transaction ID
            transaction: Transaction items
        
        Returns:
            True if assigned to at least one cluster
        """
        matching_clusters = []
        
        for cluster_id, cluster in self.clusters.items():
            pattern = cluster.pattern.items
            
            if pattern.issubset(transaction):
                score = self._calculate_match_score(transaction, pattern)
                matching_clusters.append((cluster_id, score))
        
        if matching_clusters:
            # Assign to cluster with highest score (can be extended for multi-assignment)
            best_cluster_id = max(matching_clusters, key=lambda x: x[1])[0]
            self.cluster_assignments[trans_id] = best_cluster_id
            self.clusters[best_cluster_id].add_member(trans_id)
            return True
        
        return False
    
    def _assign_threshold(self, trans_id: int, transaction: frozenset) -> bool:
        """
        Assign transaction to clusters meeting overlap threshold.
        
        Args:
            trans_id: Transaction ID
            transaction: Transaction items
        
        Returns:
            True if assigned to at least one cluster
        """
        matching_clusters = []
        
        for cluster_id, cluster in self.clusters.items():
            pattern = cluster.pattern.items
            
            if pattern.issubset(transaction):
                score = self._calculate_match_score(transaction, pattern)
                
                if score >= self.overlap_threshold:
                    matching_clusters.append(cluster_id)
        
        if matching_clusters:
            # Assign to first matching cluster (can be extended)
            best_cluster = matching_clusters[0]
            self.cluster_assignments[trans_id] = best_cluster
            self.clusters[best_cluster].add_member(trans_id)
            return True
        
        return False
    
    def _calculate_match_score(self, transaction: frozenset, pattern: frozenset) -> float:
        """
        Calculate match score between transaction and pattern.
        
        Uses Jaccard-like similarity: |pattern ∩ transaction| / |transaction|
        
        Args:
            transaction: Transaction items
            pattern: Cluster pattern items
        
        Returns:
            Match score between 0 and 1
        """
        if not transaction:
            return 0.0
        
        intersection = len(pattern & transaction)
        
        # Score based on pattern coverage of transaction
        score = intersection / len(transaction)
        
        # Alternative: Can also weight by pattern length
        # score = intersection / len(pattern)
        
        return score
    
    def _remove_empty_clusters(self) -> None:
        """Remove clusters with no members."""
        empty_clusters = [
            cluster_id for cluster_id, cluster in self.clusters.items()
            if cluster.size() == 0
        ]
        
        for cluster_id in empty_clusters:
            del self.clusters[cluster_id]
        
        if empty_clusters:
            logger.info(f"Removed {len(empty_clusters)} empty clusters")
    
    def predict(self, new_transactions: List[frozenset]) -> List[int]:
        """
        Predict cluster assignments for new transactions.
        
        Args:
            new_transactions: List of new transactions
        
        Returns:
            List of cluster IDs (-1 for outliers)
        """
        predictions = []
        
        for transaction in new_transactions:
            best_cluster_id = -1
            best_score = 0.0
            
            for cluster_id, cluster in self.clusters.items():
                pattern = cluster.pattern.items
                
                if pattern.issubset(transaction):
                    score = self._calculate_match_score(transaction, pattern)
                    
                    if score > best_score:
                        best_score = score
                        best_cluster_id = cluster_id
            
            predictions.append(best_cluster_id)
        
        return predictions
    
    def get_cluster_labels(self) -> np.ndarray:
        """
        Get cluster labels for all transactions.
        
        Returns:
            Array of cluster labels (-1 for outliers)
        """
        labels = np.full(len(self.transactions), -1, dtype=int)
        
        for trans_id, cluster_id in self.cluster_assignments.items():
            labels[trans_id] = cluster_id
        
        return labels
    
    def get_cluster_info(self) -> Dict[int, Dict]:
        """
        Get detailed information about each cluster.
        
        Returns:
            Dictionary mapping cluster IDs to cluster information
        """
        cluster_info = {}
        
        for cluster_id, cluster in self.clusters.items():
            cluster_info[cluster_id] = {
                'cluster_id': cluster_id,
                'size': cluster.size(),
                'pattern': sorted(cluster.pattern.items),
                'pattern_length': len(cluster.pattern.items),
                'pattern_support': cluster.pattern.support,
                'members': cluster.members
            }
        
        return cluster_info
    
    def get_clustering_stats(self) -> Dict:
        """
        Get statistics about the clustering results.
        
        Returns:
            Dictionary with clustering statistics
        """
        cluster_sizes = [cluster.size() for cluster in self.clusters.values()]
        
        stats = {
            'n_transactions': len(self.transactions),
            'n_clusters': len(self.clusters),
            'n_outliers': len(self.outliers),
            'outlier_percentage': (len(self.outliers) / len(self.transactions)) * 100,
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'pattern_lengths': [len(c.pattern.items) for c in self.clusters.values()]
        }
        
        if cluster_sizes:
            stats['std_cluster_size'] = np.std(cluster_sizes)
        
        return stats
    
    def print_clusters(self, top_n: int = 10) -> None:
        """
        Print cluster information in readable format.
        
        Args:
            top_n: Number of clusters to print
        """
        print(f"\n{'='*70}")
        print(f"MaPle Clustering Results")
        print(f"{'='*70}")
        print(f"Total Transactions: {len(self.transactions)}")
        print(f"Number of Clusters: {len(self.clusters)}")
        print(f"Number of Outliers: {len(self.outliers)}")
        print(f"{'='*70}\n")
        
        # Sort clusters by size
        sorted_clusters = sorted(
            self.clusters.values(),
            key=lambda c: c.size(),
            reverse=True
        )
        
        for i, cluster in enumerate(sorted_clusters[:top_n], 1):
            print(f"Cluster {cluster.cluster_id} (Size: {cluster.size()})")
            print(f"  Pattern: {sorted(cluster.pattern.items)}")
            print(f"  Support: {cluster.pattern.support}")
            print(f"  Members (first 10): {cluster.members[:10]}")
            print()
        
        if self.outliers:
            print(f"Outliers: {len(self.outliers)} transactions")
            print(f"  IDs (first 20): {self.outliers[:20]}")
