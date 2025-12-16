"""
MaPle service for orchestrating clustering workflow
"""

from typing import Dict, List, Optional
import time
from ..algorithms.maple_clustering import MaPleClusterer
from ..algorithms.pattern_mining import FrequentPattern
from .data_service import DataService
from ..utils.logger import get_logger
from ..utils.exceptions import ClusteringError

logger = get_logger(__name__)


class MapleService:
    """
    Service for orchestrating MaPle clustering workflow.
    
    Coordinates data service and clustering algorithm to provide
    a complete clustering solution.
    """
    
    def __init__(self, data_service: DataService):
        """
        Initialize MaPle service.
        
        Args:
            data_service: DataService instance for data operations
        """
        self.data_service = data_service
        self.clusterer: Optional[MaPleClusterer] = None
        self.execution_time: float = 0.0
        
        logger.info("MapleService initialized")
    
    def execute_clustering(
        self,
        min_support: float = 0.05,
        max_pattern_length: int = 10,
        n_bins: int = 5,
        discretization_strategy: str = 'quantile',
        assignment_strategy: str = 'best_match',
        overlap_threshold: float = 0.5,
        numerical_columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Execute complete MaPle clustering workflow.
        
        Args:
            min_support: Minimum support for pattern mining
            max_pattern_length: Maximum pattern length
            n_bins: Number of bins for discretization
            discretization_strategy: Discretization strategy
            assignment_strategy: Cluster assignment strategy
            overlap_threshold: Overlap threshold for assignment
            numerical_columns: Specific columns to discretize
        
        Returns:
            Dictionary with clustering results
        
        Raises:
            ClusteringError: If clustering fails
        """
        start_time = time.time()
        
        try:
            logger.info("Starting MaPle clustering workflow")
            
            # Step 1: Preprocess data
            logger.info("Step 1/4: Preprocessing data...")
            self.data_service.preprocess_data()
            
            # Step 2: Discretize data
            logger.info("Step 2/4: Discretizing data...")
            self.data_service.discretize_data(
                n_bins=n_bins,
                strategy=discretization_strategy,
                numerical_columns=numerical_columns
            )
            
            # Step 3: Convert to transactions
            logger.info("Step 3/4: Converting to transactions...")
            transactions = self.data_service.get_transactions()
            
            # Step 4: Run MaPle clustering
            logger.info("Step 4/4: Running MaPle clustering...")
            self.clusterer = MaPleClusterer(
                min_support=min_support,
                max_pattern_length=max_pattern_length,
                overlap_threshold=overlap_threshold,
                assignment_strategy=assignment_strategy
            )
            
            self.clusterer.fit(transactions)
            
            # Calculate execution time
            self.execution_time = time.time() - start_time
            
            logger.info(f"Clustering complete in {self.execution_time:.2f} seconds")
            
            # Compile results
            results = self._compile_results()
            
            return results
        
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            raise ClusteringError(f"Clustering failed: {str(e)}")
    
    def _compile_results(self) -> Dict:
        """
        Compile comprehensive clustering results.
        
        Returns:
            Dictionary with all results
        """
        if self.clusterer is None:
            raise ClusteringError("Clusterer not initialized")
        
        # Get pattern mining stats
        pattern_stats = self.clusterer.pattern_miner.get_pattern_stats()
        
        # Get clustering stats
        clustering_stats = self.clusterer.get_clustering_stats()
        
        # Get cluster information
        cluster_info = self.clusterer.get_cluster_info()
        
        # Get maximal patterns
        maximal_patterns = self.clusterer.pattern_miner.get_maximal_patterns()
        
        # Compile results
        results = {
            'success': True,
            'message': 'Clustering completed successfully',
            'execution_time_seconds': self.execution_time,
            
            # Pattern mining results
            'n_frequent_patterns': pattern_stats['total_frequent_patterns'],
            'n_maximal_patterns': pattern_stats['n_maximal_patterns'],
            'patterns_by_length': {
                str(k): v for k, v in pattern_stats['patterns_by_length'].items()
            },
            
            # Clustering results
            'n_clusters': clustering_stats['n_clusters'],
            'n_outliers': clustering_stats['n_outliers'],
            'outlier_percentage': clustering_stats['outlier_percentage'],
            'avg_cluster_size': clustering_stats['avg_cluster_size'],
            'min_cluster_size': clustering_stats['min_cluster_size'],
            'max_cluster_size': clustering_stats['max_cluster_size'],
            
            # Detailed cluster information
            'clusters': self._format_cluster_info(cluster_info),
            
            # Top maximal patterns
            'top_patterns': self._format_patterns(maximal_patterns)
        }
        
        return results
    
    def _format_cluster_info(self, cluster_info: Dict) -> List[Dict]:
        """
        Format cluster information for response.
        
        Args:
            cluster_info: Raw cluster information
        
        Returns:
            List of formatted cluster info
        """
        formatted_clusters = []
        
        for cluster_id, info in cluster_info.items():
            formatted_clusters.append({
                'cluster_id': cluster_id,
                'size': info['size'],
                'pattern': info['pattern'],
                'pattern_length': info['pattern_length'],
                'pattern_support': info['pattern_support'],
                'member_ids': info['members'][:100]  # Limit to first 100
            })
        
        # Sort by size (descending)
        formatted_clusters.sort(key=lambda x: x['size'], reverse=True)
        
        return formatted_clusters
    
    def _format_patterns(self, patterns: List[FrequentPattern], top_n: int = 20) -> List[Dict]:
        """
        Format patterns for response.
        
        Args:
            patterns: List of FrequentPattern objects
            top_n: Number of top patterns to return
        
        Returns:
            List of formatted patterns
        """
        formatted_patterns = []
        
        # Sort by support (descending)
        sorted_patterns = sorted(patterns, key=lambda p: p.support, reverse=True)
        
        n_transactions = len(self.clusterer.transactions) if self.clusterer else 1
        
        for pattern in sorted_patterns[:top_n]:
            formatted_patterns.append({
                'pattern': sorted(list(pattern.items)),
                'support': pattern.support,
                'support_percentage': (pattern.support / n_transactions) * 100,
                'is_maximal': True
            })
        
        return formatted_patterns
    
    def get_cluster_labels(self) -> List[int]:
        """
        Get cluster labels for all transactions.
        
        Returns:
            List of cluster labels
        """
        if self.clusterer is None:
            raise ClusteringError("Clusterer not initialized")
        
        return self.clusterer.get_cluster_labels().tolist()
    
    def get_clusterer(self) -> MaPleClusterer:
        """
        Get the clusterer instance.
        
        Returns:
            MaPleClusterer instance
        """
        if self.clusterer is None:
            raise ClusteringError("Clusterer not initialized")
        
        return self.clusterer
    
    def reset(self) -> None:
        """Reset the service."""
        self.clusterer = None
        self.execution_time = 0.0
        logger.info("MapleService reset")
