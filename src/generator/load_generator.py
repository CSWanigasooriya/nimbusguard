#!/usr/bin/env python3
"""
NimbusGuard Load Generator

A tool to generate various types of load against the consumer service
to test KEDA autoscaling behavior.
"""

import asyncio
import aiohttp
import time
import json
import argparse
import sys
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LoadTest:
    name: str
    description: str
    concurrent_requests: int
    total_requests: int
    cpu_intensity: int
    memory_size: int
    duration: int
    async_mode: str  # Changed from bool to str to avoid aiohttp issues
    delay_between_requests: float = 0.1

class LoadGenerator:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """Check if the consumer service is healthy"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Service is healthy: {data}")
                    return True
                else:
                    logger.error(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
    
    async def send_process_request(self, load_test: LoadTest, request_id: int) -> Dict[str, Any]:
        """Send a single process request"""
        start_time = time.time()
        
        params = {
            'cpu_intensity': load_test.cpu_intensity,
            'memory_size': load_test.memory_size,
            'duration': load_test.duration,
            'async_mode': load_test.async_mode  # Now already a string
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/process",
                params=params,
                timeout=aiohttp.ClientTimeout(total=load_test.duration + 30)
            ) as response:
                end_time = time.time()
                logger.info(f"🔍 Request {request_id}: Got response status {response.status}")
                try:
                    response_data = await response.json()
                    logger.info(f"🔍 Request {request_id}: JSON parsed successfully")
                except Exception as json_error:
                    logger.error(f"🔍 Request {request_id}: JSON parse failed: {json_error}")
                    response_text = await response.text()
                    logger.error(f"🔍 Request {request_id}: Response text: {response_text[:200]}")
                    raise json_error
                
                result = {
                    'request_id': request_id,
                    'status_code': response.status,
                    'response_time': end_time - start_time,
                    'response_data': response_data,
                    'success': response.status == 200
                }
                
                if result['success']:
                    logger.info(f"✅ Request {request_id}: {response.status} in {result['response_time']:.2f}s")
                else:
                    logger.error(f"❌ Request {request_id}: {response.status}")
                
                return result
                
        except asyncio.TimeoutError:
            logger.error(f"⏰ Request {request_id}: Timeout")
            return {
                'request_id': request_id,
                'status_code': 408,
                'response_time': time.time() - start_time,
                'response_data': {'error': 'timeout'},
                'success': False
            }
        except Exception as e:
            logger.error(f"💥 Request {request_id}: {e}")
            return {
                'request_id': request_id,
                'status_code': 500,
                'response_time': time.time() - start_time,
                'response_data': {'error': str(e)},
                'success': False
            }
    
    async def run_load_test(self, load_test: LoadTest) -> Dict[str, Any]:
        """Run a complete load test"""
        logger.info(f"🚀 Starting load test: {load_test.name}")
        logger.info(f"📝 Description: {load_test.description}")
        logger.info(f"⚙️  Config: {load_test.concurrent_requests} concurrent, {load_test.total_requests} total requests")
        
        # Check if this is the special comparison pattern that needs multi-phase execution
        if 'comparison' in load_test.name.lower():
            return await self.run_comparison_pattern_test(load_test)
        
        start_time = time.time()
        semaphore = asyncio.Semaphore(load_test.concurrent_requests)
        tasks = []
        
        async def bounded_request(request_id: int):
            async with semaphore:
                await asyncio.sleep(request_id * load_test.delay_between_requests)
                return await self.send_process_request(load_test, request_id)
        
        # Create all request tasks
        for i in range(load_test.total_requests):
            task = asyncio.create_task(bounded_request(i + 1))
            tasks.append(task)
        
        # Wait for all requests to complete
        logger.info(f"⏳ Sending {load_test.total_requests} requests...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Process results
        successful_requests = [r for r in results if isinstance(r, dict) and r.get('success', False)]
        failed_requests = [r for r in results if isinstance(r, dict) and not r.get('success', False)]
        exceptions = [r for r in results if not isinstance(r, dict)]
        
        avg_response_time = sum(r['response_time'] for r in successful_requests) / len(successful_requests) if successful_requests else 0
        
        test_summary = {
            'test_name': load_test.name,
            'total_time': total_time,
            'total_requests': load_test.total_requests,
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'exceptions': len(exceptions),
            'success_rate': len(successful_requests) / load_test.total_requests * 100,
            'avg_response_time': avg_response_time,
            'requests_per_second': load_test.total_requests / total_time,
            'results': results
        }
        
        logger.info(f"✅ Load test completed in {total_time:.2f}s")
        logger.info(f"📊 Success rate: {test_summary['success_rate']:.1f}% ({len(successful_requests)}/{load_test.total_requests})")
        logger.info(f"⚡ RPS: {test_summary['requests_per_second']:.2f}")
        logger.info(f"⏱️  Avg response time: {avg_response_time:.2f}s")
        
        self.results.append(test_summary)
        return test_summary
    
    async def monitor_service_status(self):
        """Monitor service status during load testing"""
        try:
            async with self.session.get(f"{self.base_url}/process/status") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"📊 Service status: {data['memory_store_objects']} objects, ~{data['memory_store_size_estimate_mb']}MB")
                    return data
        except Exception as e:
            logger.warning(f"⚠️  Could not get service status: {e}")
        return None
    
    async def run_comparison_pattern_test(self, load_test: LoadTest) -> Dict[str, Any]:
        """Run the deterministic multi-phase comparison pattern"""
        logger.info(f"🎯 Starting DETERMINISTIC multi-phase comparison pattern")
        logger.info(f"📋 Pattern: Baseline(5min) → High Load(10min) → Medium Load(10min) → Cool Down(5min)")
        
        # Deterministic phase definitions (30 minutes total)
        phases = [
            {
                'name': 'Baseline Load',
                'duration_minutes': 5,
                'requests_per_minute': 2,
                'concurrent_requests': 3,
                'cpu_intensity': 3,
                'memory_size': 50,
                'request_duration': 15
            },
            {
                'name': 'High Load Phase',
                'duration_minutes': 10, 
                'requests_per_minute': 8,
                'concurrent_requests': 12,
                'cpu_intensity': 9,
                'memory_size': 200,
                'request_duration': 30
            },
            {
                'name': 'Medium Load Phase',
                'duration_minutes': 10,
                'requests_per_minute': 4,
                'concurrent_requests': 6,
                'cpu_intensity': 6,
                'memory_size': 120,
                'request_duration': 20
            },
            {
                'name': 'Cool Down Phase',
                'duration_minutes': 5,
                'requests_per_minute': 1,
                'concurrent_requests': 2,
                'cpu_intensity': 2,
                'memory_size': 30,
                'request_duration': 10
            }
        ]
        
        total_start_time = time.time()
        all_results = []
        
        for i, phase in enumerate(phases, 1):
            logger.info(f"🔄 Phase {i}/4: {phase['name']} ({phase['duration_minutes']} minutes)")
            logger.info(f"   📊 Load: {phase['requests_per_minute']} req/min, CPU={phase['cpu_intensity']}, Memory={phase['memory_size']}MB")
            
            # Create a temporary LoadTest for this phase
            phase_test = LoadTest(
                name=f"Phase {i}: {phase['name']}",
                description=f"Deterministic phase {i} of comparison pattern",
                concurrent_requests=phase['concurrent_requests'],
                total_requests=phase['requests_per_minute'] * phase['duration_minutes'],
                cpu_intensity=phase['cpu_intensity'],
                memory_size=phase['memory_size'],
                duration=phase['request_duration'],
                async_mode="true",
                delay_between_requests=60.0 / phase['requests_per_minute']  # Convert to seconds between requests
            )
            
            # Run this phase (using original logic to avoid recursion)
            phase_start_time = time.time()
            semaphore = asyncio.Semaphore(phase_test.concurrent_requests)
            tasks = []
            
            async def bounded_request(request_id: int):
                async with semaphore:
                    await asyncio.sleep(request_id * phase_test.delay_between_requests)
                    return await self.send_process_request(phase_test, request_id)
            
            # Create all request tasks for this phase
            for j in range(phase_test.total_requests):
                task = asyncio.create_task(bounded_request(j + 1))
                tasks.append(task)
            
            # Wait for all requests in this phase to complete
            logger.info(f"   ⏳ Sending {phase_test.total_requests} requests over {phase['duration_minutes']} minutes...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            phase_end_time = time.time()
            phase_duration = phase_end_time - phase_start_time
            
            # Process results for this phase
            successful_requests = [r for r in results if isinstance(r, dict) and r.get('success', False)]
            success_rate = (len(successful_requests) / len(results)) * 100 if results else 0
            
            phase_summary = {
                'phase': i,
                'phase_name': phase['name'],
                'duration_seconds': phase_duration,
                'total_requests': len(results),
                'successful_requests': len(successful_requests),
                'success_rate': success_rate,
                'requests_per_second': len(results) / phase_duration if phase_duration > 0 else 0
            }
            
            all_results.append(phase_summary)
            logger.info(f"   ✅ Phase {i} completed: {success_rate:.1f}% success rate in {phase_duration:.1f}s")
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # Create overall summary
        total_requests = sum(r['total_requests'] for r in all_results)
        total_successful = sum(r['successful_requests'] for r in all_results)
        overall_success_rate = (total_successful / total_requests) * 100 if total_requests > 0 else 0
        
        comparison_summary = {
            'test_name': 'Deterministic Comparison Pattern',
            'total_duration_seconds': total_duration,
            'total_duration_minutes': total_duration / 60,
            'phases': all_results,
            'overall_total_requests': total_requests,
            'overall_successful_requests': total_successful,
            'overall_success_rate': overall_success_rate,
            'overall_requests_per_second': total_requests / total_duration if total_duration > 0 else 0
        }
        
        logger.info(f"🎉 DETERMINISTIC comparison pattern completed!")
        logger.info(f"📊 Total duration: {total_duration/60:.1f} minutes")
        logger.info(f"📈 Overall success rate: {overall_success_rate:.1f}% ({total_successful}/{total_requests})")
        logger.info(f"🎯 Pattern guaranteed to trigger: Scale Up → Keep Same → Scale Down")
        
        self.results.append(comparison_summary)
        return comparison_summary
    
    async def cleanup_service(self):
        """Clean up service memory"""
        try:
            async with self.session.delete(f"{self.base_url}/process/cleanup") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"🧹 Cleanup: {data['message']}")
                    return data
        except Exception as e:
            logger.warning(f"⚠️  Could not cleanup service: {e}")
        return None

# Predefined load test scenarios
LOAD_TESTS = {
    'light': LoadTest(
        name="Light Load",
        description="Low CPU and memory usage to baseline",
        concurrent_requests=2,
        total_requests=10,
        cpu_intensity=2,
        memory_size=20,
        duration=5,
        async_mode="true"  # Changed to async for realistic behavior
    ),
    
    'medium': LoadTest(
        name="Medium Load", 
        description="Moderate load that should start triggering scaling",
        concurrent_requests=5,
        total_requests=20,
        cpu_intensity=6,
        memory_size=80,
        duration=15,
        async_mode="true"  # Changed to async for realistic behavior
    ),
    
    'heavy': LoadTest(
        name="Heavy Load",
        description="High load that should definitely trigger KEDA scaling",
        concurrent_requests=10,
        total_requests=30,
        cpu_intensity=9,
        memory_size=150,
        duration=20,
        async_mode="true"  # Changed to async for realistic behavior
    ),
    
    'sustained': LoadTest(
        name="Sustained Load",
        description="Long-running background tasks to maintain high resource usage",
        concurrent_requests=8,
        total_requests=25,
        cpu_intensity=8,
        memory_size=120,
        duration=60,
        async_mode="true",
        delay_between_requests=2.0
    ),
    
    'burst': LoadTest(
        name="Burst Load",
        description="Quick burst of high-intensity requests",
        concurrent_requests=15,
        total_requests=40,
        cpu_intensity=10,
        memory_size=100,
        duration=10,
        async_mode="true",
        delay_between_requests=0.1
    ),
    
    'memory_stress': LoadTest(
        name="Memory Stress",
        description="Focus on memory consumption to trigger memory-based scaling",
        concurrent_requests=6,
        total_requests=15,
        cpu_intensity=3,
        memory_size=250,
        duration=30,
        async_mode="true"  # Changed to async for realistic behavior
    ),
    
    'cpu_stress': LoadTest(
        name="CPU Stress", 
        description="Focus on CPU consumption to trigger CPU-based scaling",
        concurrent_requests=8,
        total_requests=20,
        cpu_intensity=10,
        memory_size=50,
        duration=25,
        async_mode="false"  # Keep one synchronous test for comparison
    ),
    
    'comparison_pattern': LoadTest(
        name="HPA vs DQN Comparison Pattern",
        description="Deterministic pattern for fair HPA vs DQN comparison with multiple phases",
        concurrent_requests=8,
        total_requests=100,  # More requests for 30-minute test
        cpu_intensity=7,
        memory_size=120,
        duration=20,
        async_mode="true",
        delay_between_requests=18.0  # 100 requests over 30 minutes = ~18 second intervals
    )
}

async def main():
    parser = argparse.ArgumentParser(description="NimbusGuard Load Generator")
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL of the consumer service')
    parser.add_argument('--test', choices=list(LOAD_TESTS.keys()) + ['all'], default='medium', 
                        help='Load test to run')
    parser.add_argument('--cleanup', action='store_true', help='Cleanup service memory before starting')
    parser.add_argument('--monitor', action='store_true', help='Monitor service status during tests')
    parser.add_argument('--list', action='store_true', help='List available load tests')
    parser.add_argument('--seed', type=int, help='Random seed for deterministic testing (for comparison mode)')
    parser.add_argument('--duration', type=int, help='Override total test duration in seconds')
    
    args = parser.parse_args()
    
    # Set random seed for deterministic testing
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed) if 'numpy' in sys.modules else None
        logger.info(f"🎲 Using deterministic seed: {args.seed}")
    
    if args.list:
        print("Available load tests:")
        for key, test in LOAD_TESTS.items():
            print(f"  {key:18} - {test.description}")
        return
    
    async with LoadGenerator(args.url) as generator:
        # Health check first
        if not await generator.health_check():
            logger.error("❌ Service is not healthy. Exiting.")
            sys.exit(1)
        
        # Cleanup if requested
        if args.cleanup:
            await generator.cleanup_service()
            await asyncio.sleep(2)
        
        # Run load tests
        if args.test == 'all':
            logger.info("🚀 Running ALL load tests...")
            for test_name in LOAD_TESTS.keys():
                test_config = LOAD_TESTS[test_name]
                
                # Override duration if specified
                if args.duration:
                    test_config = LoadTest(
                        name=test_config.name,
                        description=f"{test_config.description} (duration override: {args.duration}s)",
                        concurrent_requests=test_config.concurrent_requests,
                        total_requests=max(test_config.total_requests, args.duration // 10),  # Scale requests with duration
                        cpu_intensity=test_config.cpu_intensity,
                        memory_size=test_config.memory_size,
                        duration=test_config.duration,  # Keep individual request duration same
                        async_mode=test_config.async_mode,
                        delay_between_requests=max(0.1, args.duration / max(test_config.total_requests, args.duration // 10))
                    )
                    logger.info(f"⏱️  Duration override: {args.duration}s")
                
                await generator.run_load_test(test_config)
                
                if args.monitor:
                    await generator.monitor_service_status()
                
                # Wait between tests
                if test_name != list(LOAD_TESTS.keys())[-1]:
                    logger.info("⏳ Waiting 30s before next test...")
                    await asyncio.sleep(30)
        else:
            if args.test not in LOAD_TESTS:
                logger.error(f"❌ Unknown test: {args.test}")
                sys.exit(1)
            
            test_config = LOAD_TESTS[args.test]
            
            # Override duration if specified (for comparison testing)
            if args.duration:
                test_config = LoadTest(
                    name=test_config.name,
                    description=f"{test_config.description} (duration override: {args.duration}s)",
                    concurrent_requests=test_config.concurrent_requests,
                    total_requests=max(test_config.total_requests, args.duration // 10),  # Scale requests with duration
                    cpu_intensity=test_config.cpu_intensity,
                    memory_size=test_config.memory_size,
                    duration=test_config.duration,  # Keep individual request duration same
                    async_mode=test_config.async_mode,
                    delay_between_requests=max(0.1, args.duration / max(test_config.total_requests, args.duration // 10))
                )
                logger.info(f"⏱️  Duration override: {args.duration}s for deterministic comparison")
            
            await generator.run_load_test(test_config)
            
            if args.monitor:
                await generator.monitor_service_status()
        
        # Final summary
        logger.info("🎉 Load generation complete!")
        if generator.results:
            total_requests = sum(r['total_requests'] for r in generator.results)
            total_successful = sum(r['successful_requests'] for r in generator.results)
            overall_success_rate = total_successful / total_requests * 100 if total_requests > 0 else 0
            
            logger.info(f"📊 Overall Summary:")
            logger.info(f"   Tests run: {len(generator.results)}")
            logger.info(f"   Total requests: {total_requests}")
            logger.info(f"   Overall success rate: {overall_success_rate:.1f}%")

if __name__ == "__main__":
    asyncio.run(main()) 