"""
Data discretization module for converting continuous features to categorical items.

This module implements various discretization strategies essential for
transforming numerical data into transactional format suitable for pattern mining.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from ..utils.logger import get_logger
from ..utils.exceptions import DataValidationError

logger = get_logger(__name__)


class DataDiscretizer:
    """
    Discretizes continuous numerical features into categorical bins.
    
    Supports multiple discretization strategies:
    - Equal-width binning
    - Equal-frequency (quantile) binning
    - K-Means based binning
    
    Attributes:
        n_bins: Number of bins for discretization
        strategy: Discretization strategy ('uniform', 'quantile', 'kmeans')
        encode: Output encoding ('ordinal', 'onehot', 'onehot-dense')
        bin_edges: Fitted bin edges for each feature
        feature_names: Original feature names
    """
    
    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = 'quantile',
        encode: str = 'ordinal',
        labels: Optional[List[str]] = None
    ):
        """
        Initialize the DataDiscretizer.
        
        Args:
            n_bins: Number of bins (default: 5)
            strategy: Discretization strategy - 'uniform', 'quantile', or 'kmeans'
            encode: Encoding type - 'ordinal', 'onehot', or 'onehot-dense'
            labels: Custom labels for bins (e.g., ['Low', 'Medium', 'High'])
        
        Raises:
            DataValidationError: If parameters are invalid
        """
        if n_bins < 2:
            raise DataValidationError("Number of bins must be at least 2")
        
        if strategy not in ['uniform', 'quantile', 'kmeans']:
            raise DataValidationError(
                f"Invalid strategy: {strategy}. Must be 'uniform', 'quantile', or 'kmeans'"
            )
        
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        self.labels = labels or self._generate_default_labels(n_bins)
        self.discretizer: Optional[KBinsDiscretizer] = None
        self.feature_names: List[str] = []
        self.bin_edges: Dict[str, np.ndarray] = {}
        
        logger.info(
            f"Initialized DataDiscretizer with n_bins={n_bins}, "
            f"strategy={strategy}, encode={encode}"
        )
    
    def _generate_default_labels(self, n_bins: int) -> List[str]:
        """Generate default bin labels based on number of bins."""
        if n_bins == 2:
            return ['Low', 'High']
        elif n_bins == 3:
            return ['Low', 'Medium', 'High']
        elif n_bins == 4:
            return ['Very_Low', 'Low', 'High', 'Very_High']
        elif n_bins == 5:
            return ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
        else:
            return [f'Bin_{i}' for i in range(n_bins)]
    
    def fit(self, data: pd.DataFrame, numerical_columns: Optional[List[str]] = None) -> 'DataDiscretizer':
        """
        Fit the discretizer on numerical data.
        
        Args:
            data: Input DataFrame
            numerical_columns: List of numerical columns to discretize
                              (if None, auto-detect numerical columns)
        
        Returns:
            self: Fitted discretizer instance
        
        Raises:
            DataValidationError: If data is invalid
        """
        if data.empty:
            raise DataValidationError("Cannot fit on empty DataFrame")
        
        # Auto-detect numerical columns if not provided
        if numerical_columns is None:
            numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_columns:
            raise DataValidationError("No numerical columns found for discretization")
        
        self.feature_names = numerical_columns
        
        # Initialize sklearn's KBinsDiscretizer
        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode='ordinal',  # We'll handle encoding ourselves
            strategy=self.strategy
        )
        
        # Fit on numerical data
        numerical_data = data[numerical_columns].values
        self.discretizer.fit(numerical_data)
        
        # Store bin edges for each feature
        for idx, col_name in enumerate(numerical_columns):
            self.bin_edges[col_name] = self.discretizer.bin_edges_[idx]
        
        logger.info(f"Fitted discretizer on {len(numerical_columns)} features")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numerical features to discretized bins.
        
        Args:
            data: Input DataFrame with numerical features
        
        Returns:
            DataFrame with discretized features
        
        Raises:
            DataValidationError: If discretizer is not fitted
        """
        if self.discretizer is None:
            raise DataValidationError("Discretizer must be fitted before transform")
        
        # Create a copy to avoid modifying original
        result = data.copy()
        
        # Transform numerical columns
        numerical_data = data[self.feature_names].values
        discretized = self.discretizer.transform(numerical_data)
        
        # Replace with labeled bins
        for idx, col_name in enumerate(self.feature_names):
            bin_indices = discretized[:, idx].astype(int)
            result[col_name] = [self.labels[i] for i in bin_indices]
        
        logger.info(f"Transformed {len(self.feature_names)} features to discrete bins")
        return result
    
    def fit_transform(self, data: pd.DataFrame, numerical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            data: Input DataFrame
            numerical_columns: List of numerical columns to discretize
        
        Returns:
            DataFrame with discretized features
        """
        self.fit(data, numerical_columns)
        return self.transform(data)
    
    def to_transactions(self, data: pd.DataFrame) -> List[frozenset]:
        """
        Convert discretized DataFrame to transaction format.
        
        Each row becomes a transaction (set of items).
        Each item has format: "FeatureName=Value"
        
        Args:
            data: Discretized DataFrame
        
        Returns:
            List of transactions (frozensets)
        
        Example:
            Input DataFrame:
            | Age    | Income |
            | Young  | Low    |
            | Middle | High   |
            
            Output:
            [
                frozenset({'Age=Young', 'Income=Low'}),
                frozenset({'Age=Middle', 'Income=High'})
            ]
        """
        transactions = []
        
        for _, row in data.iterrows():
            items = set()
            for col_name, value in row.items():
                # Skip NaN values
                if pd.notna(value):
                    # Create item string: "Feature=Value"
                    item = f"{col_name}={value}"
                    items.add(item)
            
            transactions.append(frozenset(items))
        
        logger.info(f"Converted {len(transactions)} rows to transaction format")
        return transactions
    
    def get_bin_info(self) -> Dict[str, Dict]:
        """
        Get detailed information about bins for each feature.
        
        Returns:
            Dictionary mapping feature names to bin information
        """
        if not self.bin_edges:
            return {}
        
        bin_info = {}
        for feature, edges in self.bin_edges.items():
            bin_info[feature] = {
                'n_bins': len(edges) - 1,
                'edges': edges.tolist(),
                'labels': self.labels,
                'ranges': [
                    f"[{edges[i]:.2f}, {edges[i+1]:.2f})"
                    for i in range(len(edges) - 1)
                ]
            }
        
        return bin_info
    
    def get_item_description(self, item: str) -> str:
        """
        Get human-readable description of an item.
        
        Args:
            item: Item string (e.g., "Age=Young")
        
        Returns:
            Description with actual value range
        """
        try:
            feature, label = item.split('=', 1)
            
            if feature in self.bin_edges and label in self.labels:
                bin_idx = self.labels.index(label)
                edges = self.bin_edges[feature]
                if bin_idx < len(edges) - 1:
                    return f"{feature}: [{edges[bin_idx]:.2f}, {edges[bin_idx+1]:.2f})"
            
            return item  # Return as-is if can't describe
        except (ValueError, IndexError):
            return item


# Helper functions for standalone usage

def discretize_dataframe(
    df: pd.DataFrame,
    n_bins: int = 5,
    strategy: str = 'quantile',
    numerical_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, DataDiscretizer]:
    """
    Convenience function to discretize a DataFrame.
    
    Args:
        df: Input DataFrame
        n_bins: Number of bins
        strategy: Discretization strategy
        numerical_columns: Columns to discretize
    
    Returns:
        Tuple of (discretized_dataframe, fitted_discretizer)
    """
    discretizer = DataDiscretizer(n_bins=n_bins, strategy=strategy)
    discretized_df = discretizer.fit_transform(df, numerical_columns)
    return discretized_df, discretizer


def dataframe_to_transactions(df: pd.DataFrame) -> List[frozenset]:
    """
    Convert a DataFrame directly to transaction format.
    
    Assumes DataFrame is already discretized/categorical.
    
    Args:
        df: Input DataFrame with categorical values
    
    Returns:
        List of transactions (frozensets)
    """
    transactions = []
    
    for _, row in df.iterrows():
        items = set()
        for col_name, value in row.items():
            if pd.notna(value):
                items.add(f"{col_name}={value}")
        transactions.append(frozenset(items))
    
    return transactions
