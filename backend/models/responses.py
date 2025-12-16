"""
Pydantic response models for API
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class DatasetInfoResponse(BaseModel):
    """Response model for dataset information"""
    
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Status message")
    
    n_rows: int = Field(..., description="Number of rows")
    n_columns: int = Field(..., description="Number of columns")
    column_names: List[str] = Field(..., description="Column names")
    column_types: Dict[str, str] = Field(..., description="Data types of columns")
    
    numerical_columns: List[str] = Field(..., description="Numerical column names")
    categorical_columns: List[str] = Field(..., description="Categorical column names")
    
    missing_values: Dict[str, int] = Field(..., description="Missing value counts per column")
    
    sample_data: List[Dict] = Field(..., description="Sample rows (first 5)")


class PatternInfo(BaseModel):
    """Information about a frequent pattern"""
    
    pattern: List[str] = Field(..., description="Items in the pattern")
    support: int = Field(..., description="Absolute support count")
    support_percentage: float = Field(..., description="Relative support percentage")
    is_maximal: bool = Field(..., description="Whether pattern is maximal")


class ClusterInfo(BaseModel):
    """Information about a cluster"""
    
    cluster_id: int = Field(..., description="Cluster identifier")
    size: int = Field(..., description="Number of members in cluster")
    pattern: List[str] = Field(..., description="Maximal pattern defining cluster")
    pattern_length: int = Field(..., description="Length of defining pattern")
    pattern_support: int = Field(..., description="Support of defining pattern")
    member_ids: List[int] = Field(..., description="Transaction IDs in cluster")


class MapleResultResponse(BaseModel):
    """Response model for MaPle execution results"""
    
    success: bool = Field(..., description="Whether clustering was successful")
    message: str = Field(..., description="Status message")
    
    execution_time_seconds: float = Field(..., description="Total execution time")
    
    # Pattern mining results
    n_frequent_patterns: int = Field(..., description="Total frequent patterns found")
    n_maximal_patterns: int = Field(..., description="Number of maximal patterns")
    patterns_by_length: Dict[str, int] = Field(..., description="Pattern count by length")
    
    # Clustering results
    n_clusters: int = Field(..., description="Number of clusters formed")
    n_outliers: int = Field(..., description="Number of outliers")
    outlier_percentage: float = Field(..., description="Percentage of outliers")
    
    avg_cluster_size: float = Field(..., description="Average cluster size")
    min_cluster_size: int = Field(..., description="Minimum cluster size")
    max_cluster_size: int = Field(..., description="Maximum cluster size")
    
    # Detailed information
    clusters: List[ClusterInfo] = Field(..., description="Detailed cluster information")
    top_patterns: List[PatternInfo] = Field(..., description="Top maximal patterns")


class ClusterInfoResponse(BaseModel):
    """Response model for detailed cluster information"""
    
    cluster_id: int
    size: int
    pattern: List[str]
    members: List[int]
    pattern_support: int
    
    # Cluster characteristics
    representative_items: List[str] = Field(..., description="Most common items")
    coverage: float = Field(..., description="Pattern coverage of members")


class EvaluationMetrics(BaseModel):
    """Clustering evaluation metrics"""
    
    silhouette_score: Optional[float] = Field(None, description="Silhouette coefficient (-1 to 1)")
    davies_bouldin_score: Optional[float] = Field(None, description="Davies-Bouldin index")
    calinski_harabasz_score: Optional[float] = Field(None, description="Calinski-Harabasz index")
    
    cluster_purity: Optional[float] = Field(None, description="Cluster purity (if labels available)")
    
    pattern_coverage: float = Field(..., description="Percentage of data covered by patterns")
    
    # Cluster distribution
    cluster_distribution: Dict[str, int] = Field(..., description="Size of each cluster")
    
    # Additional metrics
    avg_intra_cluster_similarity: Optional[float] = Field(None, description="Average within-cluster similarity")


class EvaluationResponse(BaseModel):
    """Response model for clustering evaluation"""
    
    success: bool
    message: str
    
    metrics: EvaluationMetrics
    
    # Visualization data (optional)
    visualization_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Data for generating visualizations"
    )


class ErrorResponse(BaseModel):
    """Response model for errors"""
    
    success: bool = False
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
