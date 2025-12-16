"""
Pydantic request models for API validation
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class DatasetUploadRequest(BaseModel):
    """Request model for dataset upload"""
    
    file_content: str = Field(..., description="CSV file content as string")
    file_name: str = Field(..., description="Original file name")
    
    @field_validator('file_name')
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        if not v.endswith('.csv'):
            raise ValueError("Only CSV files are supported")
        return v


class MapleExecutionRequest(BaseModel):
    """Request model for executing MaPle algorithm"""
    
    min_support: float = Field(
        default=0.05,
        ge=0.01,
        le=1.0,
        description="Minimum support threshold (0.01-1.0)"
    )
    
    max_pattern_length: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum pattern length (1-20)"
    )
    
    n_bins: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of bins for discretization (2-10)"
    )
    
    discretization_strategy: str = Field(
        default='quantile',
        description="Discretization strategy: uniform, quantile, or kmeans"
    )
    
    assignment_strategy: str = Field(
        default='best_match',
        description="Cluster assignment strategy: best_match, all_matching, or threshold"
    )
    
    overlap_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overlap threshold for cluster assignment (0.0-1.0)"
    )
    
    numerical_columns: Optional[List[str]] = Field(
        default=None,
        description="Specific numerical columns to discretize (None = all numerical)"
    )
    
    @field_validator('discretization_strategy')
    @classmethod
    def validate_discretization_strategy(cls, v: str) -> str:
        valid_strategies = ['uniform', 'quantile', 'kmeans']
        if v not in valid_strategies:
            raise ValueError(f"Strategy must be one of: {valid_strategies}")
        return v
    
    @field_validator('assignment_strategy')
    @classmethod
    def validate_assignment_strategy(cls, v: str) -> str:
        valid_strategies = ['best_match', 'all_matching', 'threshold']
        if v not in valid_strategies:
            raise ValueError(f"Strategy must be one of: {valid_strategies}")
        return v


class EvaluationRequest(BaseModel):
    """Request model for clustering evaluation"""
    
    include_visualization: bool = Field(
        default=True,
        description="Include visualization data in response"
    )
    
    label_column: Optional[str] = Field(
        default=None,
        description="Column name for ground truth labels (if available)"
    )
