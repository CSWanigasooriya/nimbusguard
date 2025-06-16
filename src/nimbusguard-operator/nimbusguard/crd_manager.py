"""
CRD management utilities for NimbusGuard operator.
"""

import logging
from typing import Optional
import kubernetes

from .models.nimbusguard_hpa_crd import NimbusGuardHPA
from .health import set_component_health

LOG = logging.getLogger(__name__)


class CRDManager:
    """Manages CRD installation and lifecycle"""
    
    def __init__(self):
        self.api_client: Optional[kubernetes.client.ApiClient] = None
        self.extensions_api: Optional[kubernetes.client.ApiextensionsV1Api] = None
    
    def initialize(self, k8s_client: kubernetes.client.ApiClient):
        """Initialize CRD manager with Kubernetes client"""
        self.api_client = k8s_client
        self.extensions_api = kubernetes.client.ApiextensionsV1Api(k8s_client)
    
    async def install_crds(self) -> bool:
        """Install all required CRDs using the resource's install method"""
        try:
            LOG.info("Validating/installing NimbusGuardHPA CRD...")
            
            # Check if CRD already exists (likely installed manually for deployment ordering)
            crd_name = "nimbusguardhpa.nimbusguard.io"
            if await self._crd_exists(crd_name):
                LOG.info(f"CRD {crd_name} already exists (likely from manual installation)")
                set_component_health("crd_manager", True)
                return True
            
            # Install the CRD using the resource's install method as backup
            LOG.info("Installing CRD using NimbusGuardHPA.install() as fallback...")
            NimbusGuardHPA.install(self.api_client, exist_ok=True)
            
            # Wait for CRD to be ready
            if await self.wait_for_crd_ready(crd_name):
                LOG.info("✅ NimbusGuardHPA CRD validated/installed successfully")
                set_component_health("crd_manager", True)
                return True
            else:
                LOG.error("❌ CRD installed but not ready")
                set_component_health("crd_manager", False)
                return False
            
        except Exception as e:
            LOG.error(f"Failed to install CRDs: {e}")
            set_component_health("crd_manager", False)
            return False
    
    async def _crd_exists(self, crd_name: str) -> bool:
        """Check if a CRD exists"""
        try:
            self.extensions_api.read_custom_resource_definition(name=crd_name)
            return True
        except kubernetes.client.exceptions.ApiException as e:
            if e.status == 404:
                return False
            raise
    
    async def wait_for_crd_ready(self, crd_name: str, timeout: int = 60) -> bool:
        """Wait for CRD to be established and ready"""
        import asyncio
        
        LOG.info(f"Waiting for CRD {crd_name} to be ready...")
        
        for _ in range(timeout):
            try:
                crd = self.extensions_api.read_custom_resource_definition(name=crd_name)
                
                # Check if CRD is established
                if crd.status and crd.status.conditions:
                    for condition in crd.status.conditions:
                        if (condition.type == "Established" and 
                            condition.status == "True"):
                            LOG.info(f"✅ CRD {crd_name} is ready")
                            return True
                
                await asyncio.sleep(1)
                
            except Exception as e:
                LOG.warning(f"Error checking CRD status: {e}")
                await asyncio.sleep(1)
        
        LOG.error(f"Timeout waiting for CRD {crd_name} to be ready")
        return False
    
    def get_crd_schema(self) -> str:
        """Get the CRD schema for debugging"""
        return NimbusGuardHPA.crd_schema()
    
    def cleanup(self):
        """Cleanup CRD manager resources"""
        self.api_client = None
        self.extensions_api = None
        LOG.info("CRD manager cleaned up")


# Global CRD manager instance
crd_manager = CRDManager()