"""
Data service for dataset handling and preprocessing
"""

from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
from io import StringIO
from ..utils.logger import get_logger
from ..utils.exceptions import DataValidationError
from ..algorithms.discretization import DataDiscretizer

logger = get_logger(__name__)


class DataService:
    """
    Service for managing dataset operations.
    
    Handles:
    - Dataset loading and validation
    - Missing value handling
    - Data preprocessing
    - Discretization
    - Transaction conversion
    """
    
    def __init__(self):
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.discretized_data: Optional[pd.DataFrame] = None
        self.transactions: Optional[List[frozenset]] = None
        self.discretizer: Optional[DataDiscretizer] = None
        
        logger.info("DataService initialized")
    
    def load_csv(self, csv_content: str, file_name: str) -> pd.DataFrame:
        """
        Load CSV data from string content.
        
        Args:
            csv_content: CSV file content as string
            file_name: Original file name
        
        Returns:
            Loaded DataFrame
        
        Raises:
            DataValidationError: If loading fails
        """
        try:
            # Try to load CSV with different encodings
            try:
                df = pd.read_csv(StringIO(csv_content))
            except UnicodeDecodeError:
                df = pd.read_csv(StringIO(csv_content), encoding='latin-1')
            
            if df.empty:
                raise DataValidationError("Loaded DataFrame is empty")
            
            self.raw_data = df
            logger.info(f"Loaded dataset '{file_name}' with shape {df.shape}")
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to load CSV: {str(e)}")
            raise DataValidationError(f"Failed to load CSV: {str(e)}")
    
    def get_dataset_info(self) -> Dict:
        """
        Get comprehensive information about the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics and metadata
        
        Raises:
            DataValidationError: If no data is loaded
        """
        if self.raw_data is None:
            raise DataValidationError("No dataset loaded")
        
        df = self.raw_data
        
        # Identify column types
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Missing value counts
        missing_values = df.isnull().sum().to_dict()
        
        # Column types
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Sample data (first 5 rows)
        sample_data = df.head(5).to_dict('records')
        
        info = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'column_types': column_types,
            'numerical_columns': numerical_cols,
            'categorical_columns': categorical_cols,
            'missing_values': {k: int(v) for k, v in missing_values.items()},
            'sample_data': sample_data
        }
        
        logger.info(f"Dataset info: {len(df)} rows, {len(df.columns)} columns")
        return info
    
    def preprocess_data(
        self,
        handle_missing: str = 'drop',
        drop_id_columns: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess the dataset.
        
        Args:
            handle_missing: Strategy for missing values ('drop', 'mean', 'median', 'mode')
            drop_id_columns: Whether to drop ID columns
        
        Returns:
            Preprocessed DataFrame
        
        Raises:
            DataValidationError: If preprocessing fails
        """
        if self.raw_data is None:
            raise DataValidationError("No dataset loaded")
        
        df = self.raw_data.copy()
        
        # Drop ID columns (typically first column with 'id' in name)
        if drop_id_columns:
            id_cols = [col for col in df.columns if 'id' in col.lower()]
            if id_cols:
                df = df.drop(columns=id_cols)
                logger.info(f"Dropped ID columns: {id_cols}")
        
        # Handle missing values
        if handle_missing == 'drop':
            original_len = len(df)
            df = df.dropna()
            dropped = original_len - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped} rows with missing values")
        
        elif handle_missing in ['mean', 'median']:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().any():
                    fill_value = df[col].mean() if handle_missing == 'mean' else df[col].median()
                    df[col].fillna(fill_value, inplace=True)
            
            # Fill categorical with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)
            
            logger.info(f"Filled missing values using {handle_missing}")
        
        elif handle_missing == 'mode':
            for col in df.columns:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)
            logger.info("Filled missing values using mode")
        
        # Verify no missing values remain
        if df.isnull().any().any():
            logger.warning("Some missing values remain after preprocessing")
        
        self.processed_data = df
        logger.info(f"Preprocessing complete. Shape: {df.shape}")
        
        return df
    
    def discretize_data(
        self,
        n_bins: int = 5,
        strategy: str = 'quantile',
        numerical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Discretize numerical features.
        
        Args:
            n_bins: Number of bins
            strategy: Discretization strategy
            numerical_columns: Specific columns to discretize
        
        Returns:
            Discretized DataFrame
        
        Raises:
            DataValidationError: If discretization fails
        """
        if self.processed_data is None:
            raise DataValidationError("Data must be preprocessed before discretization")
        
        df = self.processed_data.copy()
        
        # Initialize discretizer
        self.discretizer = DataDiscretizer(
            n_bins=n_bins,
            strategy=strategy
        )
        
        # Discretize
        self.discretized_data = self.discretizer.fit_transform(df, numerical_columns)
        
        logger.info(f"Discretized data with {n_bins} bins using {strategy} strategy")
        return self.discretized_data
    
    def convert_to_transactions(self) -> List[frozenset]:
        """
        Convert discretized data to transaction format.
        
        Returns:
            List of transactions
        
        Raises:
            DataValidationError: If data is not discretized
        """
        if self.discretized_data is None:
            raise DataValidationError("Data must be discretized before conversion")
        
        if self.discretizer is None:
            raise DataValidationError("Discretizer not initialized")
        
        self.transactions = self.discretizer.to_transactions(self.discretized_data)
        
        logger.info(f"Converted {len(self.transactions)} rows to transaction format")
        return self.transactions
    
    def get_transactions(self) -> List[frozenset]:
        """
        Get transactions (convert if necessary).
        
        Returns:
            List of transactions
        """
        if self.transactions is None:
            self.convert_to_transactions()
        
        return self.transactions
    
    def get_discretization_info(self) -> Dict:
        """
        Get information about discretization.
        
        Returns:
            Dictionary with bin information
        """
        if self.discretizer is None:
            return {}
        
        return self.discretizer.get_bin_info()
    
    def get_original_data(self) -> pd.DataFrame:
        """Get original raw data."""
        if self.raw_data is None:
            raise DataValidationError("No data loaded")
        return self.raw_data
    
    def get_processed_data(self) -> pd.DataFrame:
        """Get preprocessed data."""
        if self.processed_data is None:
            raise DataValidationError("Data not preprocessed")
        return self.processed_data
    
    def get_discretized_data(self) -> pd.DataFrame:
        """Get discretized data."""
        if self.discretized_data is None:
            raise DataValidationError("Data not discretized")
        return self.discretized_data
    
    def reset(self) -> None:
        """Reset all data."""
        self.raw_data = None
        self.processed_data = None
        self.discretized_data = None
        self.transactions = None
        self.discretizer = None
        logger.info("DataService reset")
