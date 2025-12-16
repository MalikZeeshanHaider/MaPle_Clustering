"""
Custom exceptions for MaPle application
"""


class MapleException(Exception):
    """Base exception for all MaPle-related errors"""
    pass


class DataValidationError(MapleException):
    """Raised when data validation fails"""
    pass


class PatternMiningError(MapleException):
    """Raised when pattern mining encounters an error"""
    pass


class ClusteringError(MapleException):
    """Raised when clustering process fails"""
    pass


class ConfigurationError(MapleException):
    """Raised when configuration is invalid"""
    pass
