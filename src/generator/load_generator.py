#!/usr/bin/env python3
"""
NimbusGuard Load Generator - FIXED FOR POD DISTRIBUTION

A tool to generate various types of load against the consumer service
to test HPA autoscaling behavior with proper pod distribution.

FIXED: Ensures requests are distributed across multiple pods by:
1. Disabling connection pooling and keep-alive
2. Forcing new connections for each request
3. Adding connection rotation strategies
4. DNS cache busting to ensure fresh service discovery
"""

import argparse
import asyncio
import logging
import random
import time
import socket
from dataclasses import dataclass
from typing import Dict, Any, List

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


class PodDistributedLoadGenerator:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session_pool: List[aiohttp.ClientSession] = []
        self.session_pool_size = 5  # Multiple sessions for better distribution
        self.current_session_index = 0
        self.results = []
        
        # Parse URL to get host and port for direct IP resolution
        self.parsed_url = self._parse_url(base_url)
        
    def _parse_url(self, url: str) -> Dict[str, str]:
        """Parse URL to extract components"""
        if "://" in url:
            scheme, rest = url.split("://", 1)
        else:
            scheme = "http"
            rest = url
            
        if ":" in rest and not rest.startswith("["):  # IPv6 handling
            host, port = rest.rsplit(":", 1)
            try:
                port = int(port)
            except ValueError:
                host = rest
                port = 80 if scheme == "http" else 443
        else:
            host = rest
            port = 80 if scheme == "http" else 443
            
        return {
            "scheme": scheme,
            "host": host,
            "port": str(port),
            "base_url": f"{scheme}://{host}:{port}"
        }

    async def __aenter__(self):
        """Create multiple sessions with different connection strategies"""
        await self._create_session_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up all sessions"""
        await self._cleanup_session_pool()

    async def _create_session_pool(self):
        """Create a pool of sessions with different configurations for better pod distribution"""
        logger.info(f"🔧 Creating session pool with {self.session_pool_size} sessions for pod distribution")
        
        for i in range(self.session_pool_size):
            # Each session has different connection behavior
            # Try to use AsyncResolver if aiodns is available, otherwise use default
            try:
                resolver = aiohttp.AsyncResolver()
            except RuntimeError:
                # aiodns not available, use default resolver
                resolver = None
            
            connector = aiohttp.TCPConnector(
                limit=1,  # Only 1 connection per session - forces distribution
                limit_per_host=1,  # Max 1 connection per host per session
                ttl_dns_cache=10,  # Short DNS cache for fresh service discovery
                use_dns_cache=False,  # Disable DNS caching for fresh lookups
                force_close=True,  # Force connection closure after each request
                enable_cleanup_closed=True,
                # keepalive_timeout=0,  # <<< FIX: This line caused the ValueError and is removed.
                resolver=resolver,  # Use async resolver if available, otherwise default
            )
            
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30, connect=5),
                headers={
                    'Connection': 'close',  # HTTP header to force connection closure
                    'Cache-Control': 'no-cache',  # Prevent any caching
                    'User-Agent': f'LoadGen-Session-{i}-{random.randint(1000,9999)}'  # Unique UA per session
                }
            )
            
            self.session_pool.append(session)
            
            # Small delay between session creation to avoid thundering herd
            await asyncio.sleep(0.1)

    async def _cleanup_session_pool(self):
        """Clean up all sessions in the pool"""
        logger.info("🧹 Cleaning up session pool")
        for session in self.session_pool:
            try:
                await session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
        self.session_pool.clear()

    async def _get_distributed_session(self) -> aiohttp.ClientSession:
        """Get a session from the pool in round-robin fashion"""
        if not self.session_pool:
            raise RuntimeError("Session pool not initialized")
            
        session = self.session_pool[self.current_session_index]
        self.current_session_index = (self.current_session_index + 1) % len(self.session_pool)
        return session

    async def _resolve_service_ips(self) -> List[str]:
        """Resolve service to get all backend pod IPs"""
        try:
            host = self.parsed_url["host"]
            
            # Try to resolve the service to see available IPs
            try:
                addrs = await asyncio.get_event_loop().getaddrinfo(
                    host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
                )
                ips = list(set(addr[4][0] for addr in addrs))
                logger.info(f"🌐 Resolved service '{host}' to IPs: {ips}")
                return ips
            except Exception as e:
                logger.info(f"🌐 Could not resolve service '{host}': {e}")
                return [host]  # Fallback to hostname
                
        except Exception as e:
            logger.warning(f"IP resolution failed: {e}")
            return [self.parsed_url["host"]]

    async def health_check(self) -> bool:
        """Check if the consumer service is healthy using distributed approach"""
        try:
            session = await self._get_distributed_session()
            
            # Add random delay to avoid all health checks hitting at once
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            async with session.get(f"{self.base_url}/health") as response:
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
        """Send a single process request with pod distribution strategy"""
        start_time = time.time()
        max_retries = 3
        retry_delay = 1.0

        params = {
            'async_mode': str(load_test.async_mode).lower()
        }

        for attempt in range(max_retries + 1):
            try:
                # Get a distributed session (round-robin)
                session = await self._get_distributed_session()
                
                # Add random jitter to request timing to break patterns
                jitter = random.uniform(0, 0.1)
                await asyncio.sleep(jitter)
                
                timeout_seconds = CONSUMER_RESOURCES["duration_per_request"] + 10
                
                # Add unique headers to prevent any proxy caching
                headers = {
                    'X-Request-ID': f'{request_id}-{int(time.time()*1000000)}',
                    'X-Load-Test': load_test.name,
                    'X-Attempt': str(attempt + 1),
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Connection': 'close'
                }
                
                async with session.post(
                        f"{self.base_url}/process",
                        params=params,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout_seconds, connect=5)
                ) as response:
                    end_time = time.time()
                    
                    # Log which server/pod responded (if available in headers)
                    server_info = response.headers.get('Server', 'unknown')
                    pod_info = response.headers.get('X-Pod-Name', response.headers.get('X-Hostname', 'unknown'))
                    
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
                        'attempts': attempt + 1,
                        'server_info': server_info,
                        'pod_info': pod_info,
                        'session_id': id(session)  # Track which session was used
                    }

                    if result['success']:
                        log_msg = f"✅ Request {request_id}: {response.status} in {result['response_time']:.2f}s"
                        if pod_info != 'unknown':
                            log_msg += f" [pod: {pod_info}]"
                        if attempt > 0:
                            log_msg += f" (attempt {attempt + 1})"
                        logger.info(log_msg)
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
                        'attempts': max_retries + 1,
                        'server_info': 'error',
                        'pod_info': 'error'
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
                        'attempts': max_retries + 1,
                        'server_info': 'timeout',
                        'pod_info': 'timeout'
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
                        'attempts': max_retries + 1,
                        'server_info': 'error',
                        'pod_info': 'error'
                    }

    async def run_load_test(self, load_test: LoadTest) -> Dict[str, Any]:
        """Run a complete load test with pod distribution verification"""
        logger.info(f"🚀 Starting load test: {load_test.name}")
        logger.info(f"📝 Description: {load_test.description}")
        logger.info(f"⚙️  Config: {load_test.concurrent_requests} concurrent, {load_test.total_requests} total requests")
        
        # Resolve service IPs for distribution verification
        service_ips = await self._resolve_service_ips()
        logger.info(f"🎯 Target service IPs: {service_ips}")
        
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

        # Rate-limited request pattern with pod distribution
        async def distributed_request(request_id: int, start_delay: float):
            # Wait for our scheduled start time
            await asyncio.sleep(start_delay)
            async with semaphore:
                return await self.send_process_request(load_test, request_id)

        # Choose pattern based on delay configuration
        if load_test.delay_between_requests > 0:
            logger.info(f"🕒 Using rate-limited pattern: {1/load_test.delay_between_requests:.2f} requests/second with pod distribution")
            # Create tasks with staggered start times for controlled rate
            for i in range(load_test.total_requests):
                start_delay = i * load_test.delay_between_requests
                # Add small random jitter to break connection patterns
                start_delay += random.uniform(0, 0.1)
                task = asyncio.create_task(distributed_request(i + 1, start_delay))
                tasks.append(task)
        else:
            logger.info(f"⚡ Using burst pattern with pod distribution: all requests start immediately")
            # Create all request tasks with small random delays for distribution
            for i in range(load_test.total_requests):
                start_delay = random.uniform(0, 0.5)  # Random delay up to 500ms
                task = asyncio.create_task(distributed_request(i + 1, start_delay))
                tasks.append(task)

        # Wait for all requests to complete
        logger.info(f"⏳ Sending {load_test.total_requests} requests with pod distribution strategy...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        # Process results with pod distribution analysis
        successful_requests = [r for r in results if isinstance(r, dict) and r.get('success', False)]
        failed_requests = [r for r in results if isinstance(r, dict) and not r.get('success', False)]
        exceptions = [r for r in results if not isinstance(r, dict)]
        
        # Analyze pod distribution
        pod_distribution = {}
        session_distribution = {}
        
        for req in successful_requests + failed_requests:
            if isinstance(req, dict):
                pod_info = req.get('pod_info', 'unknown')
                session_id = req.get('session_id', 'unknown')
                
                pod_distribution[pod_info] = pod_distribution.get(pod_info, 0) + 1
                session_distribution[session_id] = session_distribution.get(session_id, 0) + 1
        
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
            'pod_distribution': pod_distribution,
            'session_distribution': session_distribution,
            'unique_pods_hit': len([k for k in pod_distribution.keys() if k not in ['unknown', 'error', 'timeout']]),
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
        logger.info(f"🏗️  Pod distribution: {dict(list(pod_distribution.items())[:5])}{'...' if len(pod_distribution) > 5 else ''}")
        logger.info(f"🎯 Unique pods hit: {test_summary['unique_pods_hit']}")
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
    parser = argparse.ArgumentParser(description="Pod-Distributed Load Generator for HPA Testing")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the consumer service")
    parser.add_argument("--test", default=None,
                        help="Predefined test scenario (minimal, light, medium, heavy, sustained, burst)")
    parser.add_argument("--test-name", default="Custom Load Test", help="Name of the test")
    parser.add_argument("--description", default="Custom load test", help="Test description")
    parser.add_argument("--concurrent", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--total", type=int, default=30, help="Total number of requests")
    parser.add_argument("--async-mode", action="store_true", default=True, help="Use async mode")
    parser.add_argument("--sync-mode", action="store_true", help="Use sync mode (overrides async)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between request STARTS in seconds (rate limiting)")
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

    # Run the load test with pod distribution
    async with PodDistributedLoadGenerator(args.url) as generator:
        # Health check
        if not await generator.health_check():
            logger.error("❌ Service health check failed")
            return

        # Run load test
        test_result = await generator.run_load_test(load_test)

        logger.info(f"✅ Load test '{load_test.name}' completed successfully")
        
        # Final pod distribution summary
        if test_result['unique_pods_hit'] > 1:
            logger.info(f"🎯 SUCCESS: Distributed load across {test_result['unique_pods_hit']} different pods")
        elif test_result['unique_pods_hit'] == 1:
            logger.warning(f"⚠️  WARNING: All requests went to the same pod - HPA testing may be invalid")
        else:
            logger.error(f"❌ ERROR: Could not determine pod distribution")


if __name__ == "__main__":
    asyncio.run(main())
