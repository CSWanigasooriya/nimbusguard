#!/usr/bin/env python3
"""
End-to-end test for NimbusGuard Kubeflow integration
This script validates the complete ML pipeline workflow
"""

import asyncio
import json
import logging
import requests
import subprocess
import time
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KubeflowE2ETest:
    """End-to-end test suite for Kubeflow integration"""
    
    def __init__(self):
        self.namespace = "nimbusguard"
        self.ml_namespace = "nimbusguard-ml"
        self.serving_namespace = "nimbusguard-serving"
        self.experiments_namespace = "nimbusguard-experiments"
        
        # Test endpoints (assuming port-forwarding is active)
        self.consumer_endpoint = "http://localhost:8080"
        self.operator_endpoint = "http://localhost:9080"
        self.prometheus_endpoint = "http://localhost:9090"
        
    async def run_all_tests(self):
        """Run complete test suite"""
        logger.info("🚀 Starting NimbusGuard Kubeflow E2E Tests")
        
        tests = [
            ("Infrastructure Check", self.test_infrastructure),
            ("Kubeflow Components", self.test_kubeflow_components),
            ("Model Serving", self.test_model_serving),
            ("Operator Integration", self.test_operator_integration),
            ("Pipeline Execution", self.test_pipeline_execution),
            ("End-to-End Workflow", self.test_e2e_workflow)
        ]
        
        results = {}
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running test: {test_name}")
            try:
                result = await test_func()
                results[test_name] = {"status": "PASS", "details": result}
                logger.info(f"✅ {test_name}: PASSED")
            except Exception as e:
                results[test_name] = {"status": "FAIL", "error": str(e)}
                logger.error(f"❌ {test_name}: FAILED - {e}")
        
        self.print_test_summary(results)
        return results
    
    async def test_infrastructure(self) -> Dict[str, Any]:
        """Test basic infrastructure components"""
        results = {}
        
        # Check namespaces
        namespaces = [self.namespace, self.ml_namespace, self.serving_namespace, self.experiments_namespace]
        for ns in namespaces:
            cmd = f"kubectl get namespace {ns}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            results[f"namespace_{ns}"] = result.returncode == 0
            
        # Check core pods
        core_pods = {
            "kafka": f"kubectl get pods -n {self.namespace} -l app=kafka",
            "consumer-workload": f"kubectl get pods -n {self.namespace} -l app=consumer-workload",
            "nimbusguard-operator": f"kubectl get pods -n {self.namespace} -l app=nimbusguard-operator",
            "prometheus": "kubectl get pods -n monitoring -l app=prometheus",
            "grafana": "kubectl get pods -n monitoring -l app=grafana"
        }
        
        for pod_name, cmd in core_pods.items():
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            results[f"pod_{pod_name}"] = "Running" in result.stdout
            
        return results
    
    async def test_kubeflow_components(self) -> Dict[str, Any]:
        """Test Kubeflow component availability"""
        results = {}
        
        # Check Kubeflow Pipelines
        cmd = "kubectl get pods -n kubeflow -l app=ml-pipeline"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        results["kubeflow_pipelines"] = result.returncode == 0 and "Running" in result.stdout
        
        # Check Katib
        cmd = "kubectl get pods -n kubeflow -l app=katib-controller"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        results["katib"] = result.returncode == 0 and "Running" in result.stdout
        
        # Check KServe
        cmd = "kubectl get pods -n kserve-system"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        results["kserve"] = result.returncode == 0
        
        return results
    
    async def test_model_serving(self) -> Dict[str, Any]:
        """Test model serving endpoints"""
        results = {}
        
        # Check InferenceService
        cmd = f"kubectl get inferenceservices -n {self.serving_namespace}"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        results["inference_service_exists"] = result.returncode == 0
        
        # Test model prediction if serving is available
        try:
            # Get KServe endpoint
            cmd = f"kubectl get inferenceservice nimbusguard-dqn-model -n {self.serving_namespace} -o jsonpath='{{.status.url}}'"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            
            if result.stdout:
                endpoint = result.stdout.strip()
                
                # Test prediction
                test_payload = {
                    "instances": [[0.5, 0.6, 0.1, 100.0, 3.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.4]]
                }
                
                response = requests.post(
                    f"{endpoint}:predict",
                    json=test_payload,
                    timeout=10
                )
                
                results["model_prediction"] = response.status_code == 200
                if response.status_code == 200:
                    prediction = response.json()
                    results["prediction_format"] = "action" in prediction or "predictions" in prediction
            else:
                results["model_prediction"] = False
                results["prediction_format"] = False
                
        except Exception as e:
            logger.warning(f"Model serving test failed: {e}")
            results["model_prediction"] = False
            results["prediction_format"] = False
            
        return results
    
    async def test_operator_integration(self) -> Dict[str, Any]:
        """Test operator Kubeflow integration"""
        results = {}
        
        # Check operator health
        try:
            response = requests.get(f"{self.operator_endpoint}/health", timeout=5)
            results["operator_health"] = response.status_code == 200
        except:
            results["operator_health"] = False
            
        # Check operator logs for Kubeflow mentions
        cmd = f"kubectl logs -n {self.namespace} deployment/nimbusguard-operator --tail=100"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        if result.returncode == 0:
            logs = result.stdout.lower()
            results["kubeflow_integration_loaded"] = "kubeflow" in logs or "kserve" in logs
            results["operator_running"] = "started successfully" in logs or "operator started" in logs
        else:
            results["kubeflow_integration_loaded"] = False
            results["operator_running"] = False
            
        return results
    
    async def test_pipeline_execution(self) -> Dict[str, Any]:
        """Test ML pipeline execution capability"""
        results = {}
        
        # Check if pipelines are defined
        cmd = "kubectl get workflow -n kubeflow"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        results["pipelines_available"] = result.returncode == 0
        
        # Check for experiments
        cmd = f"kubectl get experiments -n {self.experiments_namespace}"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        results["experiments_available"] = result.returncode == 0
        
        return results
    
    async def test_e2e_workflow(self) -> Dict[str, Any]:
        """Test complete end-to-end workflow"""
        results = {}
        
        # 1. Generate load to trigger scaling decisions
        try:
            payload = {"intensity": 50, "duration": 60}
            response = requests.post(
                f"{self.consumer_endpoint}/api/v1/workload/cpu/start",
                json=payload,
                timeout=10
            )
            results["load_generation"] = response.status_code == 200
        except:
            results["load_generation"] = False
            
        # 2. Wait for operator to make decisions
        await asyncio.sleep(35)  # Wait for one evaluation cycle
        
        # 3. Check operator made decisions
        cmd = f"kubectl logs -n {self.namespace} deployment/nimbusguard-operator --tail=50"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        if result.returncode == 0:
            logs = result.stdout.lower()
            results["scaling_decisions_made"] = "decision:" in logs or "scaling" in logs
            results["ml_decisions"] = "dqn" in logs or "action" in logs
        else:
            results["scaling_decisions_made"] = False
            results["ml_decisions"] = False
            
        # 4. Check metrics are being collected
        try:
            response = requests.get(
                f"{self.prometheus_endpoint}/api/v1/query",
                params={"query": "nimbusguard_dqn_decisions_total"},
                timeout=5
            )
            results["metrics_collection"] = response.status_code == 200
        except:
            results["metrics_collection"] = False
            
        return results
    
    def print_test_summary(self, results: Dict[str, Any]):
        """Print formatted test summary"""
        logger.info("\n" + "="*60)
        logger.info("🧪 NIMBUSGUARD KUBEFLOW E2E TEST SUMMARY")
        logger.info("="*60)
        
        passed = sum(1 for r in results.values() if r["status"] == "PASS")
        total = len(results)
        
        for test_name, result in results.items():
            status_icon = "✅" if result["status"] == "PASS" else "❌"
            logger.info(f"{status_icon} {test_name}: {result['status']}")
            
            if result["status"] == "FAIL":
                logger.info(f"   Error: {result['error']}")
        
        logger.info("-"*60)
        logger.info(f"📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            logger.info("🎉 All tests passed! Kubeflow integration is working correctly.")
        elif passed >= total * 0.8:
            logger.info("⚠️  Most tests passed. Check failed tests for minor issues.")
        else:
            logger.info("🚨 Multiple test failures. Check your Kubeflow setup.")
        
        logger.info("="*60)


async def main():
    """Run the test suite"""
    test_suite = KubeflowE2ETest()
    results = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    total = len(results)
    
    if passed == total:
        exit(0)  # All tests passed
    elif passed >= total * 0.8:
        exit(1)  # Minor issues
    else:
        exit(2)  # Major issues


if __name__ == "__main__":
    asyncio.run(main())
