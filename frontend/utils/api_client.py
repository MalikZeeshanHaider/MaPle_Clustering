"""
API client for communicating with MaPle backend
"""

import requests
from typing import Dict, Optional, Any
import pandas as pd


class MapleAPIClient:
    """
    Client for interacting with MaPle clustering API.
    
    Handles all HTTP communication with the FastAPI backend.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.api_prefix = "/api/v1"
    
    def _get_url(self, endpoint: str) -> str:
        """Construct full URL for endpoint."""
        return f"{self.base_url}{self.api_prefix}{endpoint}"
    
    def health_check(self) -> Dict:
        """
        Check API health status.
        
        Returns:
            Health status dictionary
        """
        try:
            response = requests.get(self._get_url("/health"), timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def upload_dataset(self, df: pd.DataFrame, file_name: str) -> Dict:
        """
        Upload dataset to backend.
        
        Args:
            df: DataFrame to upload
            file_name: Name of the file
        
        Returns:
            Dataset information
        """
        # Convert DataFrame to CSV string
        csv_content = df.to_csv(index=False)
        
        payload = {
            "file_content": csv_content,
            "file_name": file_name
        }
        
        response = requests.post(
            self._get_url("/upload-dataset"),
            json=payload,
            timeout=30
        )
        
        if not response.ok:
            try:
                error_detail = response.json().get('detail', response.text)
            except Exception:
                error_detail = response.text
            raise requests.HTTPError(f"{response.status_code}: {error_detail}", response=response)
        
        return response.json()
    
    def run_maple(
        self,
        min_support: float = 0.05,
        max_pattern_length: int = 10,
        n_bins: int = 5,
        discretization_strategy: str = 'quantile',
        assignment_strategy: str = 'best_match',
        overlap_threshold: float = 0.5,
        numerical_columns: Optional[list] = None
    ) -> Dict:
        """
        Execute MaPle clustering algorithm.
        
        Args:
            min_support: Minimum support threshold
            max_pattern_length: Maximum pattern length
            n_bins: Number of bins for discretization
            discretization_strategy: Discretization strategy
            assignment_strategy: Cluster assignment strategy
            overlap_threshold: Overlap threshold
            numerical_columns: Specific columns to discretize
        
        Returns:
            Clustering results
        """
        payload = {
            "min_support": min_support,
            "max_pattern_length": max_pattern_length,
            "n_bins": n_bins,
            "discretization_strategy": discretization_strategy,
            "assignment_strategy": assignment_strategy,
            "overlap_threshold": overlap_threshold,
        }
        
        # Only include numerical_columns if specified
        if numerical_columns is not None:
            payload["numerical_columns"] = numerical_columns
        
        response = requests.post(
            self._get_url("/run-maple"),
            json=payload,
            timeout=300  # 5 minutes for large datasets
        )
        
        if not response.ok:
            # Try to extract error detail from response
            try:
                error_detail = response.json().get('detail', response.text)
            except Exception:
                error_detail = response.text
            raise requests.HTTPError(f"{response.status_code}: {error_detail}", response=response)
        
        return response.json()
    
    def get_clusters(self) -> Dict:
        """
        Get detailed cluster information.
        
        Returns:
            Cluster information
        """
        response = requests.get(self._get_url("/get-clusters"), timeout=30)
        response.raise_for_status()
        return response.json()
    
    def evaluate_clustering(
        self,
        include_visualization: bool = True,
        label_column: Optional[str] = None
    ) -> Dict:
        """
        Evaluate clustering quality.
        
        Args:
            include_visualization: Include visualization data
            label_column: Column name for ground truth labels
        
        Returns:
            Evaluation metrics
        """
        payload = {
            "include_visualization": include_visualization,
            "label_column": label_column
        }
        
        response = requests.post(
            self._get_url("/evaluate"),
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    
    def get_discretization_info(self) -> Dict:
        """
        Get discretization bin information.
        
        Returns:
            Discretization information
        """
        response = requests.get(self._get_url("/discretization-info"), timeout=10)
        response.raise_for_status()
        return response.json()
    
    def reset(self) -> Dict:
        """
        Reset backend services.
        
        Returns:
            Success message
        """
        response = requests.post(self._get_url("/reset"), timeout=10)
        response.raise_for_status()
        return response.json()
