"""
FastAPI routes for MaPle clustering API
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict
import traceback

from ..models.requests import (
    DatasetUploadRequest,
    MapleExecutionRequest,
    EvaluationRequest
)
from ..models.responses import (
    DatasetInfoResponse,
    MapleResultResponse,
    EvaluationResponse,
    ErrorResponse
)
from ..services.data_service import DataService
from ..services.maple_service import MapleService
from ..services.evaluation_service import EvaluationService
from ..utils.logger import get_logger
from ..utils.exceptions import (
    DataValidationError,
    PatternMiningError,
    ClusteringError
)

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["maple"])

# Global service instances (in production, use dependency injection)
data_service = DataService()
maple_service = MapleService(data_service)


@router.get("/health")
async def health_check() -> Dict:
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "MaPle Clustering API",
        "version": "1.0.0"
    }


@router.post("/upload-dataset", response_model=DatasetInfoResponse)
async def upload_dataset(request: DatasetUploadRequest) -> DatasetInfoResponse:
    """
    Upload and validate dataset.
    
    Args:
        request: Dataset upload request
    
    Returns:
        Dataset information
    
    Raises:
        HTTPException: If upload fails
    """
    try:
        logger.info(f"Received dataset upload: {request.file_name}")
        
        # Reset previous data
        data_service.reset()
        maple_service.reset()
        
        # Load CSV
        data_service.load_csv(request.file_content, request.file_name)
        
        # Get dataset info
        dataset_info = data_service.get_dataset_info()
        
        # Create response
        response = DatasetInfoResponse(
            success=True,
            message=f"Dataset '{request.file_name}' uploaded successfully",
            **dataset_info
        )
        
        logger.info(f"Dataset uploaded: {dataset_info['n_rows']} rows, {dataset_info['n_columns']} columns")
        
        return response
    
    except DataValidationError as e:
        logger.error(f"Data validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload dataset: {str(e)}"
        )


@router.post("/run-maple", response_model=MapleResultResponse)
async def run_maple(request: MapleExecutionRequest) -> MapleResultResponse:
    """
    Execute MaPle clustering algorithm.
    
    Args:
        request: MaPle execution parameters
    
    Returns:
        Clustering results
    
    Raises:
        HTTPException: If clustering fails
    """
    try:
        logger.info(f"Starting MaPle execution with min_support={request.min_support}")
        
        # Check if data is loaded
        if data_service.raw_data is None:
            raise DataValidationError("No dataset loaded. Please upload a dataset first.")
        
        # Execute clustering
        results = maple_service.execute_clustering(
            min_support=request.min_support,
            max_pattern_length=request.max_pattern_length,
            n_bins=request.n_bins,
            discretization_strategy=request.discretization_strategy,
            assignment_strategy=request.assignment_strategy,
            overlap_threshold=request.overlap_threshold,
            numerical_columns=request.numerical_columns
        )
        
        # Create response
        response = MapleResultResponse(**results)
        
        logger.info(
            f"MaPle execution complete: {results['n_clusters']} clusters, "
            f"{results['n_outliers']} outliers"
        )
        
        return response
    
    except DataValidationError as e:
        logger.error(f"Data validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except (PatternMiningError, ClusteringError) as e:
        logger.error(f"Clustering error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clustering failed: {str(e)}"
        )


@router.get("/get-clusters")
async def get_clusters() -> Dict:
    """
    Get detailed cluster information.
    
    Returns:
        Cluster information
    
    Raises:
        HTTPException: If no clustering has been performed
    """
    try:
        logger.info("Fetching cluster information")
        
        # Check if clustering has been performed
        clusterer = maple_service.get_clusterer()
        
        # Get cluster info
        cluster_info = clusterer.get_cluster_info()
        
        # Get cluster labels
        cluster_labels = clusterer.get_cluster_labels().tolist()
        
        return {
            "success": True,
            "message": "Cluster information retrieved",
            "clusters": cluster_info,
            "cluster_labels": cluster_labels,
            "n_clusters": len(cluster_info),
            "n_transactions": len(cluster_labels)
        }
    
    except ClusteringError as e:
        logger.error(f"Clustering error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No clustering results available. Please run MaPle first."
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get clusters: {str(e)}"
        )


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_clustering(request: EvaluationRequest) -> EvaluationResponse:
    """
    Evaluate clustering quality.
    
    Args:
        request: Evaluation parameters
    
    Returns:
        Evaluation metrics
    
    Raises:
        HTTPException: If evaluation fails
    """
    try:
        logger.info("Starting clustering evaluation")
        
        # Check if clustering has been performed
        clusterer = maple_service.get_clusterer()
        
        # Create evaluation service
        eval_service = EvaluationService(data_service, clusterer)
        
        # Perform evaluation
        results = eval_service.evaluate(
            label_column=request.label_column,
            include_visualization=request.include_visualization
        )
        
        # Create response
        response = EvaluationResponse(**results)
        
        logger.info("Evaluation complete")
        
        return response
    
    except ClusteringError as e:
        logger.error(f"Clustering error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No clustering results available. Please run MaPle first."
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.get("/discretization-info")
async def get_discretization_info() -> Dict:
    """
    Get discretization information.
    
    Returns:
        Discretization bin information
    """
    try:
        discretization_info = data_service.get_discretization_info()
        
        return {
            "success": True,
            "message": "Discretization information retrieved",
            "discretization": discretization_info
        }
    
    except Exception as e:
        logger.error(f"Error getting discretization info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/reset")
async def reset_service() -> Dict:
    """
    Reset all services and clear data.
    
    Returns:
        Success message
    """
    try:
        logger.info("Resetting services")
        
        data_service.reset()
        maple_service.reset()
        
        return {
            "success": True,
            "message": "All services reset successfully"
        }
    
    except Exception as e:
        logger.error(f"Error resetting services: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
