#!/usr/bin/env python3
"""
Training Data Collection Script for NimbusGuard

This script uses the load generator to create various load patterns while collecting
comprehensive system metrics to build a rich training dataset with all 11 state dimensions.
"""

import asyncio
import aiohttp
import csv
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataCollector:
    def __init__(self):
        # Service endpoints
        self.load_generator_endpoint = "http://localhost:8081"  # Port forward load generator
        self.consumer_endpoint = "http://localhost:8080"       # Port forward consumer
        self.prometheus_endpoint = "http://localhost:9090"     # Port forward Prometheus
        
        # Data collection settings
        self.collection_interval = 10  # seconds between data points
        self.dataset_path = Path("/Users/chamathwanigasooriya/Documents/FYP/nimbusguard/datasets")
        self.dataset_path.mkdir(exist_ok=True)
        
        # Load patterns to test
        self.load_patterns = [
            {"name": "low_constant", "pattern": "constant", "intensity": 20, "duration": 300},
            {"name": "medium_constant", "pattern": "constant", "intensity": 50, "duration": 300},
            {"name": "high_constant", "pattern": "constant", "intensity": 80, "duration": 300},
            {"name": "gradual_increase", "pattern": "gradual", "intensity": 70, "duration": 400},
            {"name": "spike_pattern", "pattern": "spike", "intensity": 90, "duration": 200},
            {"name": "burst_pattern", "pattern": "burst", "intensity": 85, "duration": 250},
            {"name": "idle_period", "pattern": "constant", "intensity": 5, "duration": 180},
        ]
        
        # Collected data
        self.collected_data = []
        
    async def collect_comprehensive_dataset(self):
        """Run complete data collection with various load patterns"""
        logger.info("🚀 Starting comprehensive training data collection")
        logger.info(f"Will collect data for {len(self.load_patterns)} different load patterns")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.dataset_path / f"comprehensive_training_data_{timestamp}.csv"
        
        try:
            # Wait for initial system stabilization
            logger.info("⏳ Waiting for system stabilization...")
            await asyncio.sleep(30)
            
            # Collect baseline data (no load)
            await self._collect_baseline_data()
            
            # Run each load pattern
            for i, pattern in enumerate(self.load_patterns, 1):
                logger.info(f"📊 Running load pattern {i}/{len(self.load_patterns)}: {pattern['name']}")
                await self._run_load_pattern_and_collect(pattern)
                
                # Rest period between patterns
                if i < len(self.load_patterns):
                    logger.info("😴 Rest period between load patterns...")
                    await asyncio.sleep(60)
            
            # Final baseline collection
            await self._collect_baseline_data(suffix="final")
            
            # Save collected data
            self._save_dataset(output_file)
            
            logger.info(f"✅ Data collection complete! Collected {len(self.collected_data)} samples")
            logger.info(f"📁 Dataset saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Data collection failed: {e}")
            raise
    
    async def _collect_baseline_data(self, suffix: str = "baseline"):
        """Collect baseline data with no artificial load"""
        logger.info(f"📈 Collecting {suffix} data...")
        
        # Collect data for 2 minutes
        for i in range(12):  # 12 samples over 2 minutes
            await self._collect_single_datapoint(f"{suffix}_{i}")
            await asyncio.sleep(10)
    
    async def _run_load_pattern_and_collect(self, pattern: Dict[str, Any]):
        """Run a specific load pattern and collect data throughout"""
        pattern_name = pattern["name"]
        
        try:
            # Start load generation
            logger.info(f"🔥 Starting load pattern: {pattern_name}")
            await self._start_load_generation(pattern)
            
            # Collect data during load generation
            duration = pattern["duration"]
            samples = duration // self.collection_interval
            
            for i in range(samples):
                await self._collect_single_datapoint(f"{pattern_name}_{i}")
                await asyncio.sleep(self.collection_interval)
            
            # Stop load generation
            await self._stop_load_generation()
            logger.info(f"✅ Completed load pattern: {pattern_name}")
            
        except Exception as e:
            logger.error(f"❌ Error in load pattern {pattern_name}: {e}")
            await self._stop_load_generation()  # Ensure load is stopped
    
    async def _start_load_generation(self, pattern: Dict[str, Any]):
        """Start load generation with specified pattern"""
        payload = {
            "pattern": pattern["pattern"],
            "duration": pattern["duration"],
            "target": "http",
            "intensity": pattern["intensity"],
            "target_url": "http://consumer-workload:8080"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.load_generator_endpoint}/load/generate",
                json=payload,
                timeout=10
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to start load generation: {response.status}")
    
    async def _stop_load_generation(self):
        """Stop current load generation"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.load_generator_endpoint}/load/stop",
                    timeout=10
                ) as response:
                    pass  # Don't fail if already stopped
        except Exception as e:
            logger.warning(f"Error stopping load generation: {e}")
    
    async def _collect_single_datapoint(self, label: str):
        """Collect a single comprehensive datapoint"""
        try:
            timestamp = datetime.now()
            
            # Collect all metrics from Prometheus
            metrics = await self._collect_prometheus_metrics()
            
            # Get current pod count
            pod_count = await self._get_current_pod_count()
            
            # Create comprehensive state vector (11 dimensions)
            state_vector = self._create_state_vector(metrics, pod_count)
            
            # Determine action based on system state
            action = self._infer_action_from_state(state_vector, label)
            
            # Calculate reward
            reward = self._calculate_reward(state_vector, metrics)
            
            # Create data point
            datapoint = {
                'timestamp': timestamp.isoformat(),
                'label': label,
                'pod_count': pod_count,
                
                # Raw metrics
                'cpu_utilization': metrics.get('cpu_utilization', 0.0),
                'memory_utilization': metrics.get('memory_utilization', 0.0),
                'network_io_rate': metrics.get('network_io_rate', 0.0),
                'request_rate': metrics.get('request_rate', 0.0),
                'response_time_p95': metrics.get('response_time_p95', 0.0),
                'error_rate': metrics.get('error_rate', 0.0),
                'queue_depth': metrics.get('queue_depth', 0.0),
                'cpu_throttling': metrics.get('cpu_throttling', 0.0),
                'memory_pressure': metrics.get('memory_pressure', 0.0),
                'node_utilization': metrics.get('node_utilization', 0.0),
                
                # Processed state vector (11 dimensions)
                'state_0_cpu': state_vector[0],
                'state_1_memory': state_vector[1],
                'state_2_network': state_vector[2],
                'state_3_requests': state_vector[3],
                'state_4_pods': state_vector[4],
                'state_5_response_time': state_vector[5],
                'state_6_error_rate': state_vector[6],
                'state_7_queue_depth': state_vector[7],
                'state_8_cpu_throttling': state_vector[8],
                'state_9_memory_pressure': state_vector[9],
                'state_10_node_util': state_vector[10],
                
                # ML training data
                'action': action,
                'action_name': self._action_to_name(action),
                'reward': reward,
            }
            
            self.collected_data.append(datapoint)
            
            logger.info(f"📊 Collected datapoint: {label} | CPU: {metrics.get('cpu_utilization', 0):.2f} | Pods: {pod_count} | Action: {self._action_to_name(action)}")
            
        except Exception as e:
            logger.error(f"❌ Error collecting datapoint {label}: {e}")
    
    async def _collect_prometheus_metrics(self) -> Dict[str, float]:
        """Collect comprehensive metrics from Prometheus"""
        queries = {
            'cpu_utilization': 'avg(rate(container_cpu_usage_seconds_total{namespace="nimbusguard"}[5m]))',
            'memory_utilization': 'avg(container_memory_working_set_bytes{namespace="nimbusguard"}) / avg(container_spec_memory_limit_bytes{namespace="nimbusguard"})',
            'network_io_rate': 'avg(rate(container_network_transmit_bytes_total{namespace="nimbusguard"}[5m]))',
            'request_rate': 'avg(rate(http_requests_total{namespace="nimbusguard"}[5m]))',
            'response_time_p95': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{namespace="nimbusguard"}[5m]))',
            'error_rate': 'avg(rate(http_requests_total{namespace="nimbusguard",status=~"5.."}[5m]))',
            'queue_depth': 'avg(kafka_consumer_lag_sum{namespace="nimbusguard"})',
            'cpu_throttling': 'avg(rate(container_cpu_cfs_throttled_seconds_total{namespace="nimbusguard"}[5m]))',
            'memory_pressure': 'avg(container_memory_working_set_bytes{namespace="nimbusguard"} / container_spec_memory_limit_bytes{namespace="nimbusguard"})',
            'node_utilization': 'avg(1 - rate(node_cpu_seconds_total{mode="idle"}[5m]))'
        }
        
        metrics = {}
        
        async with aiohttp.ClientSession() as session:
            for metric_name, query in queries.items():
                try:
                    async with session.get(
                        f"{self.prometheus_endpoint}/api/v1/query",
                        params={'query': query},
                        timeout=5
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            result = data.get('data', {}).get('result', [])
                            if result:
                                metrics[metric_name] = float(result[0]['value'][1])
                            else:
                                metrics[metric_name] = 0.0
                        else:
                            metrics[metric_name] = 0.0
                except Exception as e:
                    logger.warning(f"Failed to collect {metric_name}: {e}")
                    metrics[metric_name] = 0.0
        
        return metrics
    
    async def _get_current_pod_count(self) -> int:
        """Get current pod count from Kubernetes API or Prometheus"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.prometheus_endpoint}/api/v1/query",
                    params={'query': 'count(kube_pod_info{namespace="nimbusguard"})'},
                    timeout=5
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get('data', {}).get('result', [])
                        if result:
                            return int(float(result[0]['value'][1]))
        except Exception as e:
            logger.warning(f"Failed to get pod count: {e}")
        
        return 1  # Default fallback
    
    def _create_state_vector(self, metrics: Dict[str, float], pod_count: int) -> List[float]:
        """Create 11-dimensional state vector from collected metrics"""
        state_vector = [
            min(max(metrics.get('cpu_utilization', 0.0), 0.0), 1.0),                    # [0] CPU utilization
            min(max(metrics.get('memory_utilization', 0.0), 0.0), 1.0),                # [1] Memory utilization
            min(max(metrics.get('network_io_rate', 0.0) / 1e6, 0.0), 1.0),            # [2] Network I/O rate
            min(max(metrics.get('request_rate', 0.0) / 100, 0.0), 1.0),               # [3] Request rate
            min(max(pod_count / 20.0, 0.0), 1.0),                                      # [4] Pod count
            min(max(metrics.get('response_time_p95', 0.0), 0.0), 1.0),                # [5] Response time
            min(max(metrics.get('error_rate', 0.0), 0.0), 1.0),                       # [6] Error rate
            min(max(metrics.get('queue_depth', 0.0) / 1000, 0.0), 1.0),              # [7] Queue depth
            min(max(metrics.get('cpu_throttling', 0.0), 0.0), 1.0),                   # [8] CPU throttling
            min(max(metrics.get('memory_pressure', 0.0), 0.0), 1.0),                  # [9] Memory pressure
            min(max(metrics.get('node_utilization', 0.0), 0.0), 1.0)                  # [10] Node utilization
        ]
        
        return state_vector
    
    def _infer_action_from_state(self, state_vector: List[float], label: str) -> int:
        """Infer appropriate scaling action from current state"""
        cpu_util = state_vector[0]
        pod_ratio = state_vector[4]
        error_rate = state_vector[6]
        response_time = state_vector[5]
        
        # Determine action based on system state
        if cpu_util > 0.8 or error_rate > 0.1 or response_time > 0.5:
            # High load - scale up
            if pod_ratio < 0.5:  # Few pods, can scale up significantly
                return 4  # SCALE_UP_2
            else:
                return 3  # SCALE_UP_1
        elif cpu_util < 0.3 and error_rate < 0.01 and response_time < 0.2:
            # Low load - scale down
            if pod_ratio > 0.6:  # Many pods, can scale down
                return 0 if pod_ratio > 0.8 else 1  # SCALE_DOWN_2 or SCALE_DOWN_1
            else:
                return 2  # NO_ACTION
        else:
            # Moderate load - maintain
            return 2  # NO_ACTION
    
    def _calculate_reward(self, state_vector: List[float], metrics: Dict[str, float]) -> float:
        """Calculate reward based on system performance"""
        cpu_util = state_vector[0]
        pod_ratio = state_vector[4]
        error_rate = state_vector[6]
        response_time = state_vector[5]
        
        # Performance reward (target 60-70% CPU)
        optimal_cpu = 0.65
        cpu_distance = abs(cpu_util - optimal_cpu)
        performance_reward = 1.0 - (cpu_distance * 2.0)
        
        # Efficiency reward (penalize over-provisioning)
        if cpu_util < 0.3 and pod_ratio > 0.6:
            efficiency_reward = -0.5
        elif cpu_util > 0.8 and pod_ratio < 0.5:
            efficiency_reward = -0.3
        else:
            efficiency_reward = 0.3
        
        # Quality reward (penalize errors and high response times)
        quality_reward = 1.0 - (error_rate + response_time * 0.5)
        
        # Combined reward
        total_reward = (performance_reward * 0.5 + efficiency_reward * 0.3 + quality_reward * 0.2)
        
        return max(-1.0, min(1.0, total_reward))
    
    def _action_to_name(self, action: int) -> str:
        """Convert action index to name"""
        action_names = {
            0: "SCALE_DOWN_2",
            1: "SCALE_DOWN_1", 
            2: "NO_ACTION",
            3: "SCALE_UP_1",
            4: "SCALE_UP_2"
        }
        return action_names.get(action, "UNKNOWN")
    
    def _save_dataset(self, output_file: Path):
        """Save collected data to CSV file"""
        if not self.collected_data:
            logger.warning("No data collected to save")
            return
        
        # Get all field names from the first datapoint
        fieldnames = list(self.collected_data[0].keys())
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.collected_data)
        
        logger.info(f"💾 Saved {len(self.collected_data)} datapoints to {output_file}")
        
        # Create summary statistics
        self._create_dataset_summary(output_file)
    
    def _create_dataset_summary(self, dataset_file: Path):
        """Create summary statistics for the collected dataset"""
        summary_file = dataset_file.with_suffix('.summary.json')
        
        # Calculate statistics
        actions = [d['action'] for d in self.collected_data]
        rewards = [d['reward'] for d in self.collected_data]
        cpu_utils = [d['cpu_utilization'] for d in self.collected_data]
        pod_counts = [d['pod_count'] for d in self.collected_data]
        
        summary = {
            'dataset_info': {
                'filename': dataset_file.name,
                'total_samples': len(self.collected_data),
                'collection_date': datetime.now().isoformat(),
                'collection_duration_minutes': len(self.collected_data) * self.collection_interval / 60
            },
            'action_distribution': {
                'SCALE_DOWN_2': actions.count(0),
                'SCALE_DOWN_1': actions.count(1),
                'NO_ACTION': actions.count(2),
                'SCALE_UP_1': actions.count(3),
                'SCALE_UP_2': actions.count(4)
            },
            'metrics_stats': {
                'cpu_utilization': {
                    'mean': np.mean(cpu_utils),
                    'std': np.std(cpu_utils),
                    'min': np.min(cpu_utils),
                    'max': np.max(cpu_utils)
                },
                'pod_count': {
                    'mean': np.mean(pod_counts),
                    'std': np.std(pod_counts),
                    'min': int(np.min(pod_counts)),
                    'max': int(np.max(pod_counts))
                },
                'reward': {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'min': np.min(rewards),
                    'max': np.max(rewards)
                }
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📈 Dataset summary saved to {summary_file}")

async def main():
    """Main function to run data collection"""
    collector = TrainingDataCollector()
    
    print("🎯 NimbusGuard Training Data Collection")
    print("=" * 50)
    print("This script will:")
    print("1. Run various load patterns using your load generator")
    print("2. Collect comprehensive 11-dimensional state data")
    print("3. Generate actions and rewards for each state")
    print("4. Save a complete training dataset")
    print()
    
    # Check if services are accessible
    try:
        async with aiohttp.ClientSession() as session:
            # Test load generator
            async with session.get(f"{collector.load_generator_endpoint}/health", timeout=5) as response:
                if response.status != 200:
                    raise Exception("Load generator not accessible")
            
            # Test Prometheus
            async with session.get(f"{collector.prometheus_endpoint}/api/v1/query", 
                                 params={'query': 'up'}, timeout=5) as response:
                if response.status != 200:
                    raise Exception("Prometheus not accessible")
        
        print("✅ All services accessible")
        print()
        
        # Start data collection
        await collector.collect_comprehensive_dataset()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Make sure you have port forwarding active:")
        print("kubectl port-forward -n nimbusguard svc/load-generator 8081:8080")
        print("kubectl port-forward -n nimbusguard svc/consumer-workload 8080:8080")
        print("kubectl port-forward -n monitoring svc/prometheus 9090:9090")

if __name__ == "__main__":
    asyncio.run(main()) 