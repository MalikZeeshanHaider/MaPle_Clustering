"""
Evaluation service for clustering quality metrics
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA
from ..algorithms.maple_clustering import MaPleClusterer
from .data_service import DataService
from ..utils.logger import get_logger
from ..utils.exceptions import ClusteringError

logger = get_logger(__name__)


class EvaluationService:
    """
    Service for evaluating clustering quality.
    
    Computes various clustering metrics including:
    - Silhouette score
    - Davies-Bouldin index
    - Calinski-Harabasz index
    - Cluster purity (if labels available)
    - Pattern coverage
    """
    
    def __init__(self, data_service: DataService, clusterer: MaPleClusterer):
        """
        Initialize evaluation service.
        
        Args:
            data_service: DataService instance
            clusterer: Fitted MaPleClusterer instance
        """
        self.data_service = data_service
        self.clusterer = clusterer
        
        logger.info("EvaluationService initialized")
    
    def evaluate(
        self,
        label_column: Optional[str] = None,
        include_visualization: bool = True
    ) -> Dict:
        """
        Perform comprehensive clustering evaluation.
        
        Args:
            label_column: Column name for ground truth labels
            include_visualization: Include data for visualization
        
        Returns:
            Dictionary with evaluation metrics and visualization data
        """
        logger.info("Starting clustering evaluation")
        
        try:
            # Get cluster labels
            cluster_labels = self.clusterer.get_cluster_labels()
            
            # Filter out outliers for some metrics
            valid_indices = cluster_labels != -1
            n_valid = np.sum(valid_indices)
            
            if n_valid < 2:
                logger.warning("Too few clustered points for evaluation")
                return self._create_empty_evaluation()
            
            # Get data for evaluation
            data = self._prepare_data_for_evaluation()
            
            # Calculate metrics
            metrics = {}
            
            # Silhouette Score (only for non-outliers)
            if n_valid >= 2 and len(np.unique(cluster_labels[valid_indices])) > 1:
                try:
                    silhouette = silhouette_score(
                        data[valid_indices],
                        cluster_labels[valid_indices]
                    )
                    metrics['silhouette_score'] = float(silhouette)
                    logger.info(f"Silhouette Score: {silhouette:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to calculate silhouette score: {e}")
                    metrics['silhouette_score'] = None
            else:
                metrics['silhouette_score'] = None
            
            # Davies-Bouldin Index
            if n_valid >= 2 and len(np.unique(cluster_labels[valid_indices])) > 1:
                try:
                    db_score = davies_bouldin_score(
                        data[valid_indices],
                        cluster_labels[valid_indices]
                    )
                    metrics['davies_bouldin_score'] = float(db_score)
                    logger.info(f"Davies-Bouldin Score: {db_score:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to calculate Davies-Bouldin score: {e}")
                    metrics['davies_bouldin_score'] = None
            else:
                metrics['davies_bouldin_score'] = None
            
            # Calinski-Harabasz Index
            if n_valid >= 2 and len(np.unique(cluster_labels[valid_indices])) > 1:
                try:
                    ch_score = calinski_harabasz_score(
                        data[valid_indices],
                        cluster_labels[valid_indices]
                    )
                    metrics['calinski_harabasz_score'] = float(ch_score)
                    logger.info(f"Calinski-Harabasz Score: {ch_score:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to calculate Calinski-Harabasz score: {e}")
                    metrics['calinski_harabasz_score'] = None
            else:
                metrics['calinski_harabasz_score'] = None
            
            # Cluster purity (if labels provided)
            if label_column:
                try:
                    purity = self._calculate_purity(cluster_labels, label_column)
                    metrics['cluster_purity'] = float(purity)
                    logger.info(f"Cluster Purity: {purity:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to calculate purity: {e}")
                    metrics['cluster_purity'] = None
            else:
                metrics['cluster_purity'] = None
            
            # Pattern coverage
            coverage = self._calculate_pattern_coverage()
            metrics['pattern_coverage'] = float(coverage)
            logger.info(f"Pattern Coverage: {coverage:.2f}%")
            
            # Cluster distribution
            cluster_distribution = self._get_cluster_distribution(cluster_labels)
            metrics['cluster_distribution'] = cluster_distribution
            
            # Average intra-cluster similarity
            avg_similarity = self._calculate_avg_intra_cluster_similarity()
            metrics['avg_intra_cluster_similarity'] = float(avg_similarity) if avg_similarity else None
            
            # Prepare response
            response = {
                'success': True,
                'message': 'Evaluation completed successfully',
                'metrics': metrics
            }
            
            # Add visualization data if requested
            if include_visualization:
                viz_data = self._prepare_visualization_data(data, cluster_labels)
                response['visualization_data'] = viz_data
            
            logger.info("Evaluation complete")
            return response
        
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise ClusteringError(f"Evaluation failed: {str(e)}")
    
    def _prepare_data_for_evaluation(self) -> np.ndarray:
        """
        Prepare data matrix for evaluation.
        
        Converts transactions to binary feature matrix.
        
        Returns:
            NumPy array of features
        """
        transactions = self.clusterer.transactions
        
        # Get all unique items
        all_items = set()
        for transaction in transactions:
            all_items.update(transaction)
        
        all_items = sorted(all_items)
        item_to_idx = {item: idx for idx, item in enumerate(all_items)}
        
        # Create binary matrix
        data_matrix = np.zeros((len(transactions), len(all_items)))
        
        for trans_id, transaction in enumerate(transactions):
            for item in transaction:
                data_matrix[trans_id, item_to_idx[item]] = 1
        
        return data_matrix
    
    def _calculate_purity(self, cluster_labels: np.ndarray, label_column: str) -> float:
        """
        Calculate cluster purity using ground truth labels.
        
        Args:
            cluster_labels: Predicted cluster labels
            label_column: Column name for ground truth
        
        Returns:
            Purity score (0-1)
        """
        try:
            original_data = self.data_service.get_original_data()
            
            if label_column not in original_data.columns:
                logger.warning(f"Label column '{label_column}' not found")
                return 0.0
            
            true_labels = original_data[label_column].values
            
            # Remove outliers
            valid_indices = cluster_labels != -1
            cluster_labels = cluster_labels[valid_indices]
            true_labels = true_labels[valid_indices]
            
            if len(cluster_labels) == 0:
                return 0.0
            
            # Calculate purity
            total_correct = 0
            unique_clusters = np.unique(cluster_labels)
            
            for cluster_id in unique_clusters:
                # Get true labels for this cluster
                cluster_mask = cluster_labels == cluster_id
                cluster_true_labels = true_labels[cluster_mask]
                
                # Find most common true label
                if len(cluster_true_labels) > 0:
                    unique, counts = np.unique(cluster_true_labels, return_counts=True)
                    max_count = np.max(counts)
                    total_correct += max_count
            
            purity = total_correct / len(cluster_labels)
            return purity
        
        except Exception as e:
            logger.error(f"Failed to calculate purity: {e}")
            return 0.0
    
    def _calculate_pattern_coverage(self) -> float:
        """
        Calculate percentage of transactions covered by patterns.
        
        Returns:
            Coverage percentage (0-100)
        """
        n_transactions = len(self.clusterer.transactions)
        n_outliers = len(self.clusterer.outliers)
        n_covered = n_transactions - n_outliers
        
        coverage = (n_covered / n_transactions) * 100
        return coverage
    
    def _get_cluster_distribution(self, cluster_labels: np.ndarray) -> Dict[str, int]:
        """
        Get size of each cluster.
        
        Args:
            cluster_labels: Cluster labels
        
        Returns:
            Dictionary mapping cluster IDs to sizes
        """
        unique, counts = np.unique(cluster_labels, return_counts=True)
        
        distribution = {}
        for cluster_id, count in zip(unique, counts):
            if cluster_id == -1:
                distribution['outliers'] = int(count)
            else:
                distribution[f'cluster_{cluster_id}'] = int(count)
        
        return distribution
    
    def _calculate_avg_intra_cluster_similarity(self) -> Optional[float]:
        """
        Calculate average Jaccard similarity within clusters.
        
        Returns:
            Average similarity score
        """
        try:
            similarities = []
            
            for cluster in self.clusterer.clusters.values():
                if cluster.size() < 2:
                    continue
                
                # Get transactions in this cluster
                member_transactions = [
                    self.clusterer.transactions[tid]
                    for tid in cluster.members
                ]
                
                # Calculate pairwise Jaccard similarities
                for i in range(len(member_transactions)):
                    for j in range(i + 1, len(member_transactions)):
                        t1 = member_transactions[i]
                        t2 = member_transactions[j]
                        
                        if len(t1 | t2) > 0:
                            similarity = len(t1 & t2) / len(t1 | t2)
                            similarities.append(similarity)
            
            if similarities:
                return np.mean(similarities)
            return None
        
        except Exception as e:
            logger.error(f"Failed to calculate intra-cluster similarity: {e}")
            return None
    
    def _prepare_visualization_data(
        self,
        data: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Dict:
        """
        Prepare data for visualization.
        
        Uses PCA for dimensionality reduction to 2D.
        
        Args:
            data: Feature matrix
            cluster_labels: Cluster labels
        
        Returns:
            Dictionary with visualization data
        """
        try:
            # Apply PCA for 2D visualization
            if data.shape[1] > 2:
                pca = PCA(n_components=2)
                data_2d = pca.fit_transform(data)
                explained_variance = pca.explained_variance_ratio_.tolist()
            else:
                data_2d = data
                explained_variance = [1.0, 0.0]
            
            # Prepare data for frontend
            viz_data = {
                'x': data_2d[:, 0].tolist(),
                'y': data_2d[:, 1].tolist(),
                'cluster_labels': cluster_labels.tolist(),
                'n_features_original': int(data.shape[1]),
                'explained_variance': explained_variance,
                'n_points': int(data.shape[0])
            }
            
            return viz_data
        
        except Exception as e:
            logger.error(f"Failed to prepare visualization data: {e}")
            return {}
    
    def _create_empty_evaluation(self) -> Dict:
        """Create empty evaluation response."""
        return {
            'success': False,
            'message': 'Insufficient data for evaluation',
            'metrics': {
                'silhouette_score': None,
                'davies_bouldin_score': None,
                'calinski_harabasz_score': None,
                'cluster_purity': None,
                'pattern_coverage': 0.0,
                'cluster_distribution': {},
                'avg_intra_cluster_similarity': None
            },
            'visualization_data': None
        }
