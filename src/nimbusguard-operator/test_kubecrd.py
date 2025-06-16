#!/usr/bin/env python3
"""
Test script to verify kubecrd integration works correctly.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nimbusguard'))

try:
    from nimbusguard.models.nimbusguard_hpa_crd import NimbusGuardHPA
    
    print("✅ Successfully imported NimbusGuardHPA")
    
    # Test CRD schema generation
    print("\n📋 Generated CRD Schema:")
    print("=" * 50)
    crd_schema = NimbusGuardHPA.crd_schema()
    print(crd_schema)
    
    # Test creating a sample resource
    print("\n🔧 Creating sample resource:")
    print("=" * 50)
    
    sample = NimbusGuardHPA(
        namespace="default",
        target_labels={"app": "test-app"},
        prometheus_url="http://prometheus:9090",
        evaluation_interval=30,
        decision_window="5m",
        metrics=[
            {
                "query": "rate(http_requests_total[5m])",
                "threshold": 10,
                "condition": "gt"
            }
        ]
    )
    
    print(f"Sample resource created: {sample.__class__.__name__}")
    print(f"Namespace: {sample.namespace}")
    print(f"Target labels: {sample.target_labels}")
    print(f"Prometheus URL: {sample.prometheus_url}")
    print(f"Metrics: {len(sample.metrics)} configured")
    
    print("\n✅ kubecrd integration test passed!")
    print("\n🗑️  Manual CRD files (crd.yaml) are no longer needed!")
    print("📦 CRDs will be installed automatically by the operator using kubecrd")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure kubecrd and apischema are installed:")
    print("pip install kubecrd apischema")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
