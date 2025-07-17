#!/usr/bin/env python3
"""
NimbusGuard Load Generator

A tool to generate various types of load against the consumer service
to test HPA autoscaling behavior with realistic resource usage per request.
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

# Consumer service resource configuration (aligned with deployment)
CONSUMER_RESOURCES = {
    "cpu_request_per_pod": 0.2,  # 200m CPU request per pod
    "memory_request_per_pod": 512,  # 512Mi memory request per pod
    "cpu_per_request": 0.025,  # ~25m CPU per request (realistic load)
    "memory_per_request": 75,  # 75MB memory per request
    "duration_per_request": 15,  # 15 seconds per request
    "hpa_cpu_threshold": 0.14,  # 140m CPU (70% of 200m request)
    "hpa_memory_threshold": 410,  # 410MB memory (80% of 512Mi request)
    "requests_to_trigger_hpa": 6,  # 6 requests needed to trigger HPA (6 * 25m = 150m > 140m)
    "requests_per_pod_capacity": 8  # ~8 requests per pod before approaching limits
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
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
            force_close=False,
            enable_cleanup_closed=True,
            keepalive_timeout=60
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30, connect=10)
        )
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
        """Send a single process request with retry logic for resilience"""
        start_time = time.time()
        max_retries = 3
        retry_delay = 1.0

        params = {
            'async_mode': str(load_test.async_mode).lower()
        }

        for attempt in range(max_retries + 1):
            try:
                timeout_seconds = CONSUMER_RESOURCES["duration_per_request"] + 10
                
                async with self.session.post(
                        f"{self.base_url}/process",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=timeout_seconds, connect=5)
                ) as response:
                    end_time = time.time()
                    
                    try:
                        response_data = await response.json()
                    except Exception as json_error:
                        logger.error(f"Request {request_id}: JSON parse failed: {json_error}")
                        response_text = await response.text()
                        logger.error(f"Request {request_id}: Response text: {response_text[:200]}")
                        response_data = {'error': 'json_parse_failed'}

                    result = {
                        'request_id': request_id,
                        'status_code': response.status,
                        'response_time': end_time - start_time,
                        'response_data': response_data,
                        'success': response.status == 200,
                        'attempts': attempt + 1
                    }

                    if result['success']:
                        if attempt > 0:
                            logger.info(f"✅ Request {request_id}: {response.status} in {result['response_time']:.2f}s (succeeded on attempt {attempt + 1})")
                        else:
                            logger.info(f"✅ Request {request_id}: {response.status} in {result['response_time']:.2f}s")
                    else:
                        logger.error(f"❌ Request {request_id}: {response.status}")

                    return result

            except (aiohttp.ClientConnectorError, aiohttp.ClientOSError, aiohttp.ServerDisconnectedError) as e:
                if attempt < max_retries:
                    logger.warning(f"🔄 Request {request_id}: Connection failed (attempt {attempt + 1}), retrying in {retry_delay}s: {str(e)[:100]}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                    continue
                else:
                    logger.error(f"💥 Request {request_id}: Connection failed after {max_retries + 1} attempts: {str(e)[:100]}")
                    return {
                        'request_id': request_id,
                        'status_code': 503,
                        'response_time': time.time() - start_time,
                        'response_data': {'error': f'connection_failed_after_retries: {str(e)[:100]}'},
                        'success': False,
                        'attempts': max_retries + 1
                    }
            
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    logger.warning(f"🔄 Request {request_id}: Timeout (attempt {attempt + 1}), retrying in {retry_delay}s")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                else:
                    logger.error(f"⏰ Request {request_id}: Timeout after {max_retries + 1} attempts")
                    return {
                        'request_id': request_id,
                        'status_code': 408,
                        'response_time': time.time() - start_time,
                        'response_data': {'error': 'timeout_after_retries'},
                        'success': False,
                        'attempts': max_retries + 1
                    }
            
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"🔄 Request {request_id}: Unexpected error (attempt {attempt + 1}), retrying in {retry_delay}s: {str(e)[:100]}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                else:
                    logger.error(f"💥 Request {request_id}: Unexpected error after {max_retries + 1} attempts: {str(e)[:100]}")
                    return {
                        'request_id': request_id,
                        'status_code': 500,
                        'response_time': time.time() - start_time,
                        'response_data': {'error': f'unexpected_error_after_retries: {str(e)[:100]}'},
                        'success': False,
                        'attempts': max_retries + 1
                    }

    async def run_load_test(self, load_test: LoadTest) -> Dict[str, Any]:
        """Run a complete load test"""
        logger.info(f"🚀 Starting load test: {load_test.name}")
        logger.info(f"📝 Description: {load_test.description}")
        logger.info(f"⚙️  Config: {load_test.concurrent_requests} concurrent, {load_test.total_requests} total requests")
        
        # Calculate expected resource usage
        expected_cpu = load_test.concurrent_requests * CONSUMER_RESOURCES["cpu_per_request"]
        expected_memory = load_test.concurrent_requests * CONSUMER_RESOURCES["memory_per_request"]
        expected_pods = max(2, (load_test.concurrent_requests + CONSUMER_RESOURCES["requests_per_pod_capacity"] - 1) // CONSUMER_RESOURCES["requests_per_pod_capacity"])
        
        logger.info(f"📊 Expected peak usage: {expected_cpu:.3f} CPU cores, {expected_memory}MB memory")
        logger.info(f"🏗️  Expected pods needed: {expected_pods}")
        
        hpa_will_trigger = (expected_cpu > CONSUMER_RESOURCES["hpa_cpu_threshold"] or 
                           expected_memory > CONSUMER_RESOURCES["hpa_memory_threshold"])
        logger.info(f"🎯 HPA will trigger: {'Yes' if hpa_will_trigger else 'No'}")

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
        
        # Count retry statistics
        requests_with_retries = [r for r in successful_requests + failed_requests if r.get('attempts', 1) > 1]
        total_attempts = sum(r.get('attempts', 1) for r in successful_requests + failed_requests)

        avg_response_time = sum(r['response_time'] for r in successful_requests) / len(
            successful_requests) if successful_requests else 0

        test_summary = {
            'test_name': load_test.name,
            'total_time': total_time,
            'total_requests': load_test.total_requests,
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'exceptions': len(exceptions),
            'requests_with_retries': len(requests_with_retries),
            'total_attempts': total_attempts,
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
        if requests_with_retries:
            logger.info(f"🔄 Retries: {len(requests_with_retries)} requests needed retries (total attempts: {total_attempts})")
        logger.info(f"⚡ RPS: {test_summary['requests_per_second']:.2f}")
        logger.info(f"⏱️  Avg response time: {avg_response_time:.2f}s")

        self.results.append(test_summary)
        return test_summary




# Predefined load test scenarios aligned with HPA configuration
LOAD_TESTS = {
    'minimal': LoadTest(
        name="Minimal Load",
        description="Very light load - should not trigger HPA",
        concurrent_requests=2,  # 2 * 25m = 50m CPU, 2 * 75MB = 150MB (well below thresholds)
        total_requests=20,
        async_mode=True,
        delay_between_requests=3.0
    ),

    'light': LoadTest(
        name="Light Load", 
        description="Light load approaching HPA threshold",
        concurrent_requests=5,  # 5 * 25m = 125m CPU, 5 * 75MB = 375MB (approaching thresholds)
        total_requests=40,
        async_mode=True,
        delay_between_requests=2.0
    ),

    'medium': LoadTest(
        name="Medium Load",
        description="Medium load that should trigger HPA scaling",
        concurrent_requests=8,  # 8 * 25m = 200m CPU, 8 * 75MB = 600MB (triggers HPA)
        total_requests=60,
        async_mode=True,
        delay_between_requests=1.5
    ),

    'heavy': LoadTest(
        name="Heavy Load",
        description="Heavy load requiring multiple pods",
        concurrent_requests=15,  # 15 * 25m = 375m CPU, 15 * 75MB = 1125MB (multiple pods)
        total_requests=80,
        async_mode=True,
        delay_between_requests=1.0
    ),

    'sustained': LoadTest(
        name="Sustained Load",
        description="Sustained load to test steady-state scaling",
        concurrent_requests=12,  # 12 * 25m = 300m CPU, 12 * 75MB = 900MB
        total_requests=100,
        async_mode=True,
        delay_between_requests=2.0
    ),

    'burst': LoadTest(
        name="Burst Load",
        description="Quick burst to test rapid scaling response",
        concurrent_requests=20,  # 20 * 25m = 500m CPU, 20 * 75MB = 1500MB
        total_requests=60,
        async_mode=True,
        delay_between_requests=0.5
    )
}


async def main():
    parser = argparse.ArgumentParser(description="Load Generator for Consumer Service HPA Testing")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the consumer service")
    parser.add_argument("--test", default=None,
                        help="Predefined test scenario (minimal, light, medium, heavy, sustained, burst)")
    parser.add_argument("--test-name", default="Custom Load Test", help="Name of the test")
    parser.add_argument("--description", default="Custom load test", help="Test description")
    parser.add_argument("--concurrent", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--total", type=int, default=30, help="Total number of requests")
    parser.add_argument("--async-mode", action="store_true", default=True, help="Use async mode")
    parser.add_argument("--sync-mode", action="store_true", help="Use sync mode (overrides async)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for deterministic testing")


    args = parser.parse_args()

    # Set random seed for deterministic testing
    random.seed(args.seed)
    logger.info(f"🎲 Using deterministic seed: {args.seed}")

    # Log resource model
    logger.info(f"🔧 Resource model: {CONSUMER_RESOURCES['cpu_per_request']*1000:.0f}m CPU, {CONSUMER_RESOURCES['memory_per_request']}MB per request")
    logger.info(f"📊 HPA thresholds: {CONSUMER_RESOURCES['hpa_cpu_threshold']*1000:.0f}m CPU, {CONSUMER_RESOURCES['hpa_memory_threshold']}MB memory")

    # Create load test configuration
    if args.test and args.test in LOAD_TESTS:
        load_test = LOAD_TESTS[args.test]
        logger.info(f"📋 Using predefined test scenario: {args.test}")
    else:
        # Determine async mode
        async_mode = True
        if args.sync_mode:
            async_mode = False
        
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

    # Log expected resource usage
    expected_cpu = load_test.concurrent_requests * CONSUMER_RESOURCES["cpu_per_request"]
    expected_memory = load_test.concurrent_requests * CONSUMER_RESOURCES["memory_per_request"]
    expected_pods = max(2, (load_test.concurrent_requests + CONSUMER_RESOURCES["requests_per_pod_capacity"] - 1) // CONSUMER_RESOURCES["requests_per_pod_capacity"])
    
    hpa_will_trigger = (expected_cpu > CONSUMER_RESOURCES["hpa_cpu_threshold"] or 
                       expected_memory > CONSUMER_RESOURCES["hpa_memory_threshold"])

    logger.info(f"📊 Expected resource usage: {expected_cpu:.3f} CPU cores, {expected_memory}MB memory")
    logger.info(f"🏗️  Expected pods: {expected_pods}")
    logger.info(f"🎯 HPA will trigger: {'Yes' if hpa_will_trigger else 'No'}")

    # Run the load test
    async with LoadGenerator(args.url) as generator:
        # Health check
        if not await generator.health_check():
            logger.error("❌ Service health check failed")
            return

        # Run load test
        test_result = await generator.run_load_test(load_test)

        logger.info(f"✅ Load test '{load_test.name}' completed successfully")


if __name__ == "__main__":
    asyncio.run(main())