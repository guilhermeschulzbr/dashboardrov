"""Módulos de análise"""


from .fleet import FleetAnalysis
from .drivers import DriverAnalysis
from .anomalies import AnomalyDetection
from .timeline import TimelineAnalysis
from .suspicions import SuspicionAnalysis

__all__ = [
    'FleetAnalysis',
    'DriverAnalysis', 
    'AnomalyDetection',
    'TimelineAnalysis',
    'SuspicionAnalysis'
]
