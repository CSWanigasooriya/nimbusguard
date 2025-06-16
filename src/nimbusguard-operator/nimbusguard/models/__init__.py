"""
CRD models for NimbusGuard operator using kubecrd.
"""

from .nimbusguard_hpa_crd import (
    NimbusGuardHPA,
    NimbusGuardHPASpec,
    NimbusGuardHPAStatus,
)

__all__ = [
    # IntelligentScaling models
    "NimbusGuardHPA",
    "NimbusGuardHPASpec",
    "NimbusGuardHPAStatus"
    ]
