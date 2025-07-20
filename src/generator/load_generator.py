#!/usr/bin/env python3
"""
Simple Load Generator

A straightforward tool to generate HTTP load against a service.
Focus: Generate load, measure basic metrics, keep it simple.
"""

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, List

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LoadTest:
    name: str
    concurrent_requests: int
    total_requests: int
    delay_between_requests: float = 0.0
    async_mode: bool = True


class SimpleLoadGenerator:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    logger.info("✅ Service is healthy")
                    return True
                else:
                    logger.error(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False

    async def send_request(self, request_id: int, async_mode: bool) -> Dict[str, Any]:
        """Send a single request"""
        start_time = time.time()
        
        try:
            params = {'async_mode': str(async_mode).lower()}
            
            async with self.session.post(
                f"{self.base_url}/process",
                params=params,
                timeout=aiohttp.ClientTimeout(total=25)
            ) as response:
                end_time = time.time()
                
                try:
                    response_data = await response.json()
                except:
                    response_data = {'error': 'json_parse_failed'}

                result = {
                    'request_id': request_id,
                    'status_code': response.status,
                    'response_time': end_time - start_time,
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
                'success': False
            }
        except Exception as e:
            logger.error(f"💥 Request {request_id}: Error - {str(e)[:100]}")
            return {
                'request_id': request_id,
                'status_code': 500,
                'response_time': time.time() - start_time,
                'success': False
            }

    async def run_load_test(self, load_test: LoadTest) -> Dict[str, Any]:
        """Run the load test"""
        logger.info(f"🚀 Starting load test: {load_test.name}")
        logger.info(f"⚙️  Config: {load_test.concurrent_requests} concurrent, {load_test.total_requests} total requests")
        
        start_time = time.time()
        semaphore = asyncio.Semaphore(load_test.concurrent_requests)
        tasks = []

        async def limited_request(request_id: int, start_delay: float):
            await asyncio.sleep(start_delay)
            async with semaphore:
                return await self.send_request(request_id, load_test.async_mode)

        # Create tasks with optional delay between starts
        for i in range(load_test.total_requests):
            start_delay = i * load_test.delay_between_requests
            task = asyncio.create_task(limited_request(i + 1, start_delay))
            tasks.append(task)

        # Wait for all requests to complete
        logger.info(f"⏳ Sending {load_test.total_requests} requests...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate basic metrics
        successful = [r for r in results if isinstance(r, dict) and r.get('success', False)]
        failed = [r for r in results if isinstance(r, dict) and not r.get('success', False)]
        
        avg_response_time = sum(r['response_time'] for r in successful) / len(successful) if successful else 0
        success_rate = len(successful) / load_test.total_requests * 100

        summary = {
            'test_name': load_test.name,
            'total_time': total_time,
            'total_requests': load_test.total_requests,
            'successful_requests': len(successful),
            'failed_requests': len(failed),
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'requests_per_second': load_test.total_requests / total_time,
        }

        logger.info(f"✅ Load test completed in {total_time:.2f}s")
        logger.info(f"📊 Success rate: {success_rate:.1f}% ({len(successful)}/{load_test.total_requests})")
        logger.info(f"⚡ RPS: {summary['requests_per_second']:.2f}")
        logger.info(f"⏱️  Avg response time: {avg_response_time:.2f}s")

        return summary


# Predefined test scenarios
LOAD_TESTS = {
    'light': LoadTest(
        name="Light Load",
        concurrent_requests=5,
        total_requests=30,
        delay_between_requests=1.0
    ),
    'medium': LoadTest(
        name="Medium Load", 
        concurrent_requests=10,
        total_requests=50,
        delay_between_requests=0.5
    ),
    'heavy': LoadTest(
        name="Heavy Load",
        concurrent_requests=20,
        total_requests=100,
        delay_between_requests=0.2
    ),
    'burst': LoadTest(
        name="Burst Load",
        concurrent_requests=30,
        total_requests=60,
        delay_between_requests=0.0
    ),
    'sustained': LoadTest(
        name="Sustained Load",
        concurrent_requests=8,
        total_requests=80,
        delay_between_requests=2.0
    )
}


async def main():
    parser = argparse.ArgumentParser(description="Simple Load Generator")
    parser.add_argument("--url", default="http://localhost:8000", help="Service URL")
    parser.add_argument("--test", default=None, help="Predefined test (light, medium, heavy, burst, sustained)")
    parser.add_argument("--concurrent", type=int, default=5, help="Concurrent requests")
    parser.add_argument("--total", type=int, default=30, help="Total requests")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between starts (seconds)")
    parser.add_argument("--sync-mode", action="store_true", help="Use sync mode instead of async")

    args = parser.parse_args()

    # Use predefined test or create custom
    if args.test and args.test in LOAD_TESTS:
        load_test = LOAD_TESTS[args.test]
        logger.info(f"📋 Using predefined test: {args.test}")
    else:
        load_test = LoadTest(
            name="Custom Load Test",
            concurrent_requests=args.concurrent,
            total_requests=args.total,
            delay_between_requests=args.delay,
            async_mode=not args.sync_mode
        )

    # Run the test
    async with SimpleLoadGenerator(args.url) as generator:
        if not await generator.health_check():
            logger.error("❌ Service not healthy, aborting")
            return

        await generator.run_load_test(load_test)


if __name__ == "__main__":
    asyncio.run(main())