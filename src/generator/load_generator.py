#!/usr/bin/env python3
"""
NimbusGuard Load Generator

A tool to generate various types of load against the consumer service
to test HPA autoscaling behavior with fixed resource usage per request.
"""

import argparse
import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Dict, Any

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Consumer service fixed resource configuration (from consumer.py)
CONSUMER_FIXED_RESOURCES = {
    "cpu_per_request": 0.04,  # 40m CPU per request
    "memory_per_request": 80,  # 80MB memory per request
    "duration_per_request": 35,  # 35 seconds per request (10s ramp + 20s sustain + 5s ramp-down)
    "hpa_cpu_threshold": 0.14,  # 140m CPU (70% of 200m request)
    "hpa_memory_threshold": 410,  # 410MB memory (80% of 512Mi request)
    "requests_to_trigger_hpa": 4,  # 4 requests needed to trigger HPA (4 * 40m = 160m > 140m)
    "requests_per_pod_capacity": 5  # ~5 requests per pod before approaching limits
}


@dataclass
class LoadTest:
    name: str
    description: str
    concurrent_requests: int
    total_requests: int
    async_mode: bool = True
    delay_between_requests: float = 0.1


class LoadGenerator:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.results = []

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=0,
            limit_per_host=1,
            ttl_dns_cache=1,
            use_dns_cache=True,
            force_close=True,
            enable_cleanup_closed=True
        )
        self.session = aiohttp.ClientSession(connector=connector)
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
        """Send a single process request with fixed resource usage"""
        start_time = time.time()

        # Updated parameters for fixed resource consumer
        params = {
            'async_mode': load_test.async_mode
        }

        # Send HTTP request and wait for response (realistic behavior)
        try:
            # Use fixed duration + buffer for timeout
            timeout_seconds = CONSUMER_FIXED_RESOURCES["duration_per_request"] + 30
            
            async with self.session.post(
                    f"{self.base_url}/process",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
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
        
        # Log expected resource usage
        expected_cpu = load_test.concurrent_requests * CONSUMER_FIXED_RESOURCES["cpu_per_request"]
        expected_memory = load_test.concurrent_requests * CONSUMER_FIXED_RESOURCES["memory_per_request"]
        expected_pods = max(1, (load_test.concurrent_requests + CONSUMER_FIXED_RESOURCES["requests_per_pod_capacity"] - 1) // CONSUMER_FIXED_RESOURCES["requests_per_pod_capacity"])
        
        logger.info(f"📊 Expected peak usage: {expected_cpu:.2f} CPU cores, {expected_memory}MB memory")
        logger.info(f"🏗️  Expected pods needed: {expected_pods}")
        
        hpa_will_trigger = (expected_cpu > CONSUMER_FIXED_RESOURCES["hpa_cpu_threshold"] or 
                           expected_memory > CONSUMER_FIXED_RESOURCES["hpa_memory_threshold"])
        logger.info(f"🎯 HPA will trigger: {'Yes' if hpa_will_trigger else 'No'}")

        # Handle special baseline comparison test
        if "Baseline Comparison Test" in load_test.name:
            return await self.run_baseline_comparison_phases(load_test)

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

        avg_response_time = sum(r['response_time'] for r in successful_requests) / len(
            successful_requests) if successful_requests else 0

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
            'expected_resource_usage': {
                'peak_cpu_cores': expected_cpu,
                'peak_memory_mb': expected_memory,
                'expected_pods': expected_pods,
                'hpa_triggered': hpa_will_trigger
            },
            'results': results
        }

        logger.info(f"✅ Load test completed in {total_time:.2f}s")
        logger.info(f"📊 Success rate: {test_summary['success_rate']:.1f}% ({len(successful_requests)}/{load_test.total_requests})")
        logger.info(f"⚡ RPS: {test_summary['requests_per_second']:.2f}")
        logger.info(f"⏱️  Avg response time: {avg_response_time:.2f}s")

        self.results.append(test_summary)
        return test_summary

    async def run_baseline_comparison_phases(self, load_test: LoadTest) -> Dict[str, Any]:
        """Run production-like baseline comparison test with fixed resource usage"""
        logger.info(f"🎯 Starting Production-Like Baseline Comparison Test")
        logger.info(f"📋 Using fixed resource model: {CONSUMER_FIXED_RESOURCES['cpu_per_request']} CPU, {CONSUMER_FIXED_RESOURCES['memory_per_request']}MB per request")

        # Production-like load phases optimized for fixed resource usage
        phases = [
            {
                'name': 'Morning Light Load',
                'duration_minutes': 5,
                'requests_per_minute': 6,  # 2 requests every 20 seconds
                'concurrent_requests': 2,  # 2 * 40m = 80m CPU, 2 * 80MB = 160MB (below HPA thresholds)
                'description': 'Light morning traffic - stays below HPA thresholds'
            },
            {
                'name': 'Business Hours Peak',
                'duration_minutes': 10,
                'requests_per_minute': 12,  # 1 request every 5 seconds
                'concurrent_requests': 6,  # 6 * 40m = 240m CPU, 6 * 80MB = 480MB (triggers HPA)
                'description': 'Peak business hours - triggers HPA scaling'
            },
            {
                'name': 'Sustained Operations',
                'duration_minutes': 10,
                'requests_per_minute': 9,  # 1 request every 6.7 seconds
                'concurrent_requests': 4,  # 4 * 40m = 160m CPU, 4 * 80MB = 320MB (triggers HPA)
                'description': 'Sustained afternoon load - maintains HPA scaling'
            },
            {
                'name': 'Evening Wind-Down',
                'duration_minutes': 5,
                'requests_per_minute': 4,  # 1 request every 15 seconds
                'concurrent_requests': 2,  # 2 * 40m = 80m CPU, 2 * 80MB = 160MB (below HPA thresholds)
                'description': 'Evening cooldown - scales back down'
            }
        ]

        total_start_time = time.time()
        all_results = []

        for i, phase in enumerate(phases, 1):
            logger.info(f"🏢 Phase {i}/4: {phase['name']} ({phase['duration_minutes']} minutes)")
            
            # Calculate expected resource usage
            expected_cpu = phase['concurrent_requests'] * CONSUMER_FIXED_RESOURCES["cpu_per_request"]
            expected_memory = phase['concurrent_requests'] * CONSUMER_FIXED_RESOURCES["memory_per_request"]
            expected_pods = max(1, (phase['concurrent_requests'] + CONSUMER_FIXED_RESOURCES["requests_per_pod_capacity"] - 1) // CONSUMER_FIXED_RESOURCES["requests_per_pod_capacity"])
            
            hpa_will_trigger = (expected_cpu > CONSUMER_FIXED_RESOURCES["hpa_cpu_threshold"] or 
                               expected_memory > CONSUMER_FIXED_RESOURCES["hpa_memory_threshold"])
            
            logger.info(f"   📊 Expected: {expected_cpu:.2f} CPU cores, {expected_memory}MB memory, {expected_pods} pods")
            logger.info(f"   🎯 HPA trigger: {'Yes' if hpa_will_trigger else 'No'} - {phase['description']}")

            # Calculate timing
            total_requests = phase['requests_per_minute'] * phase['duration_minutes']
            delay_between_requests = 60.0 / phase['requests_per_minute'] if phase['requests_per_minute'] > 0 else 60.0

            # Create LoadTest for this phase
            phase_test = LoadTest(
                name=f"Phase {i}: {phase['name']}",
                description=phase['description'],
                concurrent_requests=phase['concurrent_requests'],
                total_requests=total_requests,
                async_mode=load_test.async_mode,
                delay_between_requests=delay_between_requests
            )

            # Run this phase
            phase_start_time = time.time()
            semaphore = asyncio.Semaphore(phase_test.concurrent_requests)
            tasks = []

            async def bounded_request(request_id: int):
                async with semaphore:
                    await asyncio.sleep((request_id - 1) * phase_test.delay_between_requests)
                    return await self.send_process_request(phase_test, request_id)

            for j in range(phase_test.total_requests):
                task = asyncio.create_task(bounded_request(j + 1))
                tasks.append(task)

            logger.info(f"   🚀 Sending {phase_test.total_requests} requests over {phase['duration_minutes']} minutes...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            phase_end_time = time.time()
            actual_phase_duration = phase_end_time - phase_start_time

            # Process results
            successful_requests = [r for r in results if isinstance(r, dict) and r.get('success', False)]
            failed_requests = [r for r in results if isinstance(r, dict) and not r.get('success', False)]
            success_rate = (len(successful_requests) / len(results)) * 100 if results else 0

            avg_response_time = sum(r['response_time'] for r in successful_requests) / len(
                successful_requests) if successful_requests else 0

            phase_summary = {
                'phase': i,
                'phase_name': phase['name'],
                'duration_seconds': actual_phase_duration,
                'duration_minutes': actual_phase_duration / 60,
                'total_requests': len(results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'requests_per_second': len(results) / actual_phase_duration if actual_phase_duration > 0 else 0,
                'expected_resource_usage': {
                    'concurrent_requests': phase['concurrent_requests'],
                    'expected_cpu_cores': expected_cpu,
                    'expected_memory_mb': expected_memory,
                    'expected_pods': expected_pods,
                    'hpa_triggered': hpa_will_trigger
                }
            }

            all_results.append(phase_summary)

            logger.info(f"   ✅ Phase {i} completed: {success_rate:.1f}% success rate in {actual_phase_duration / 60:.1f} minutes")

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # Create overall summary
        total_requests = sum(r['total_requests'] for r in all_results)
        total_successful = sum(r['successful_requests'] for r in all_results)
        total_failed = sum(r['failed_requests'] for r in all_results)
        overall_success_rate = (total_successful / total_requests) * 100 if total_requests > 0 else 0

        baseline_summary = {
            'test_name': 'Production-Like Baseline Comparison (Fixed Resources)',
            'total_duration_seconds': total_duration,
            'total_duration_minutes': total_duration / 60,
            'phases': all_results,
            'overall_total_requests': total_requests,
            'overall_successful_requests': total_successful,
            'overall_failed_requests': total_failed,
            'overall_success_rate': overall_success_rate,
            'overall_requests_per_second': total_requests / total_duration if total_duration > 0 else 0,
            'fixed_resource_model': CONSUMER_FIXED_RESOURCES,
            'pattern_description': 'Fixed resource model: Light Load(2 req) → Peak Load(6 req) → Sustained Load(4 req) → Wind-Down(2 req)',
            'mode': 'fire_and_forget' if load_test.fire_and_forget else 'standard'
        }

        logger.info(f"🎉 Production-Like Baseline test completed!")
        logger.info(f"📊 Total duration: {total_duration / 60:.1f} minutes")
        logger.info(f"📈 Overall success rate: {overall_success_rate:.1f}% ({total_successful}/{total_requests})")
        logger.info(f"🏗️  Fixed resource model: {CONSUMER_FIXED_RESOURCES['cpu_per_request']} CPU, {CONSUMER_FIXED_RESOURCES['memory_per_request']}MB per request")

        self.results.append(baseline_summary)
        return baseline_summary

    async def monitor_service_status(self, interval_seconds: int = 10):
        """Monitor service status continuously"""
        logger.info(f"📊 Starting continuous monitoring (every {interval_seconds}s)")
        while True:
            try:
                async with self.session.get(f"{self.base_url}/process/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        current_stats = data.get('processing_stats', {})
                        active_requests = current_stats.get('active_requests', 0)
                        completed_requests = current_stats.get('completed_requests', 0)
                        peak_memory = current_stats.get('peak_memory_mb', 0)
                        
                        logger.info(f"📊 Monitor: {active_requests} active, {completed_requests} completed, {peak_memory}MB peak memory")
                    else:
                        logger.warning(f"⚠️  Monitor: Status check returned {response.status}")
            except Exception as e:
                logger.warning(f"⚠️  Monitor: Could not get service status: {e}")

            await asyncio.sleep(interval_seconds)

    async def cleanup_service(self):
        """Clean up service if cleanup endpoints exist"""
        try:
            # Try to get status first to see if service is using old or new API
            async with self.session.get(f"{self.base_url}/process/status") as response:
                if response.status == 200:
                    logger.info("🧹 Service is responsive, cleanup not needed for fixed resource model")
                    return {"message": "Fixed resource model - no cleanup needed"}
        except Exception as e:
            logger.warning(f"⚠️  Could not check service status: {e}")
        return None


# Updated predefined load test scenarios for fixed resource model
LOAD_TESTS = {
    'light': LoadTest(
        name="Light Load",
        description="Light load that stays below HPA thresholds",
        concurrent_requests=2,  # 2 * 40m = 80m CPU, 2 * 80MB = 160MB (below thresholds)
        total_requests=10,
        async_mode=True,
        delay_between_requests=2.0
    ),

    'medium': LoadTest(
        name="Medium Load",
        description="Medium load that should trigger HPA scaling",
        concurrent_requests=4,  # 4 * 40m = 160m CPU, 4 * 80MB = 320MB (triggers HPA)
        total_requests=20,
        async_mode=True,
        delay_between_requests=1.0
    ),

    'heavy': LoadTest(
        name="Heavy Load",
        description="Heavy load that definitely triggers HPA scaling",
        concurrent_requests=8,  # 8 * 40m = 320m CPU, 8 * 80MB = 640MB (multiple pods needed)
        total_requests=30,
        async_mode=True,
        delay_between_requests=0.5
    ),

    'sustained': LoadTest(
        name="Sustained Load",
        description="Sustained load to maintain HPA scaling",
        concurrent_requests=6,  # 6 * 40m = 240m CPU, 6 * 80MB = 480MB (triggers HPA)
        total_requests=40,
        async_mode=True,
        delay_between_requests=2.0
    ),

    'burst': LoadTest(
        name="Burst Load",
        description="Quick burst to test rapid scaling",
        concurrent_requests=12,  # 12 * 40m = 480m CPU, 12 * 80MB = 960MB (multiple pods)
        total_requests=50,
        async_mode=True,
        delay_between_requests=0.1
    ),

    'scale_test': LoadTest(
        name="Scale Test",
        description="Progressive scaling test",
        concurrent_requests=15,  # 15 * 40m = 600m CPU, 15 * 80MB = 1200MB (3 pods needed)
        total_requests=60,
        async_mode=True,
        delay_between_requests=0.2
    ),

    'baseline': LoadTest(
        name="Production-Like Baseline Comparison Test",
        description="Production-like load pattern with fixed resource usage for HPA testing",
        concurrent_requests=4,  # Will be overridden by phases
        total_requests=120,  # Will be calculated by phases
        async_mode=True,
        delay_between_requests=12.0  # Will be overridden by phases
    )
}


async def main():
    parser = argparse.ArgumentParser(description="Load Generator for Fixed Resource Consumer Service")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the consumer service")
    parser.add_argument("--test", default=None,
                        help="Predefined test scenario (light, medium, heavy, sustained, burst, scale_test, baseline)")
    parser.add_argument("--test-name", default="Custom Load Test", help="Name of the test")
    parser.add_argument("--description", default="Custom load test with fixed resource usage", help="Test description")
    parser.add_argument("--concurrent", type=int, default=4, help="Number of concurrent requests")
    parser.add_argument("--total", type=int, default=20, help="Total number of requests")
    parser.add_argument("--async-mode", action="store_true", help="Use async mode (default: True)")
    parser.add_argument("--sync-mode", action="store_true", help="Use sync mode (overrides async)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for deterministic testing")
    parser.add_argument("--monitor", action="store_true", help="Enable monitoring during test")
    parser.add_argument("--cleanup", action="store_true", help="Enable cleanup after test")

    args = parser.parse_args()

    # Set random seed for deterministic testing
    random.seed(args.seed)
    logger.info(f"🎲 Using deterministic seed: {args.seed}")

    # Log fixed resource model
    logger.info(f"🔧 Fixed resource model: {CONSUMER_FIXED_RESOURCES['cpu_per_request']} CPU, {CONSUMER_FIXED_RESOURCES['memory_per_request']}MB per request")
    logger.info(f"📊 HPA thresholds: {CONSUMER_FIXED_RESOURCES['hpa_cpu_threshold']} CPU, {CONSUMER_FIXED_RESOURCES['hpa_memory_threshold']}MB memory")

    # Create load test configuration
    if args.test and args.test in LOAD_TESTS:
        load_test = LOAD_TESTS[args.test]
        logger.info(f"📋 Using predefined test scenario: {args.test}")
    else:
        # Determine async mode
        async_mode = True  # Default to async
        if args.sync_mode:
            async_mode = False
        elif args.async_mode:
            async_mode = True
        
        load_test = LoadTest(
            name=args.test_name,
            description=args.description,
            concurrent_requests=args.concurrent,
            total_requests=args.total,
            async_mode=async_mode,
            delay_between_requests=args.delay
        )
        if args.test:
            logger.warning(f"⚠️  Unknown test scenario '{args.test}', using custom parameters")

    # Log mode and expected resource usage
    expected_cpu = load_test.concurrent_requests * CONSUMER_FIXED_RESOURCES["cpu_per_request"]
    expected_memory = load_test.concurrent_requests * CONSUMER_FIXED_RESOURCES["memory_per_request"]
    expected_pods = max(1, (load_test.concurrent_requests + CONSUMER_FIXED_RESOURCES["requests_per_pod_capacity"] - 1) // CONSUMER_FIXED_RESOURCES["requests_per_pod_capacity"])
    
    hpa_will_trigger = (expected_cpu > CONSUMER_FIXED_RESOURCES["hpa_cpu_threshold"] or 
                       expected_memory > CONSUMER_FIXED_RESOURCES["hpa_memory_threshold"])

    mode_desc = "async" if load_test.async_mode else "sync"
    logger.info(f"🔄 Mode: {mode_desc} (waits for HTTP response)")
    logger.info(f"📊 Expected resource usage: {expected_cpu:.2f} CPU cores, {expected_memory}MB memory")
    logger.info(f"🏗️  Expected pods: {expected_pods}")
    logger.info(f"🎯 HPA will trigger: {'Yes' if hpa_will_trigger else 'No'}")

    # Run the load test
    async with LoadGenerator(args.url) as generator:
        # Health check
        if not await generator.health_check():
            logger.error("❌ Service health check failed")
            return

        # Cleanup before test
        await generator.cleanup_service()

        # Extended monitoring if requested
        if args.monitor:
            logger.info("📊 Extended monitoring enabled")
            monitoring_task = asyncio.create_task(generator.monitor_service_status())

        # Run load test
        test_result = await generator.run_load_test(load_test)

        # Stop monitoring
        if args.monitor:
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass

        # Cleanup after test
        if args.cleanup:
            logger.info("🧹 Post-test cleanup requested")
            await generator.cleanup_service()

        logger.info(f"✅ Load test '{load_test.name}' completed successfully")


if __name__ == "__main__":
    asyncio.run(main())