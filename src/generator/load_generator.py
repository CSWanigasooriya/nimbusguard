#!/usr/bin/env python3
"""
NimbusGuard Load Generator

A tool to generate various types of load against the consumer service
to test KEDA autoscaling behavior.
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
    fire_and_forget: bool = False  # Pure fire-and-forget mode - don't wait for any response


class LoadGenerator:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.results = []

    async def __aenter__(self):
        # Configure connector to disable connection pooling for better load distribution
        connector = aiohttp.TCPConnector(
            limit=0,  # Unlimited connections
            limit_per_host=1,  # Max 1 connection per host (forces new connections)
            ttl_dns_cache=1,  # Short DNS cache TTL to get fresh IPs
            use_dns_cache=True,
            force_close=True,  # Force close connections after each request
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
        """Send a single process request"""
        start_time = time.time()

        params = {
            'cpu_intensity': load_test.cpu_intensity,
            'memory_size': load_test.memory_size,
            'duration': load_test.duration,
            'async_mode': load_test.async_mode  # Now already a string
        }

        if load_test.fire_and_forget:
            # Pure fire-and-forget mode - send request but don't wait for response
            try:
                # Create the request but don't await it
                asyncio.create_task(
                    self.session.post(
                        f"{self.base_url}/process",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=1.0)  # Short timeout since we're not waiting
                    )
                )
                end_time = time.time()
                logger.info(f"🚀 Request {request_id}: Fired (fire-and-forget mode)")

                # Return immediately with mock successful result
                return {
                    'request_id': request_id,
                    'status_code': 202,  # Accepted
                    'response_time': end_time - start_time,
                    'response_data': {'status': 'fired', 'mode': 'fire_and_forget'},
                    'success': True
                }
            except Exception as e:
                logger.error(f"💥 Request {request_id}: Failed to fire - {e}")
                return {
                    'request_id': request_id,
                    'status_code': 500,
                    'response_time': time.time() - start_time,
                    'response_data': {'error': str(e)},
                    'success': False
                }

        # Original async/sync mode - wait for response
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
        logger.info(
            f"⚙️  Config: {load_test.concurrent_requests} concurrent, {load_test.total_requests} total requests")

        # NOTE: Removed complex multi-phase logic - use standard timing for all tests
        # The multi-phase logic was causing issues with async/sync mode timing

        # Special handling for baseline comparison test with varying load phases
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
            'results': results
        }

        logger.info(f"✅ Load test completed in {total_time:.2f}s")
        logger.info(
            f"📊 Success rate: {test_summary['success_rate']:.1f}% ({len(successful_requests)}/{load_test.total_requests})")
        logger.info(f"⚡ RPS: {test_summary['requests_per_second']:.2f}")
        logger.info(f"⏱️  Avg response time: {avg_response_time:.2f}s")

        self.results.append(test_summary)
        return test_summary

    async def monitor_service_status(self, interval_seconds: int = 10):
        """Monitor service status continuously during load testing"""
        logger.info(f"📊 Starting continuous monitoring (every {interval_seconds}s)")
        while True:
            try:
                async with self.session.get(f"{self.base_url}/process/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        current_status = data.get('current_status', {})
                        memory_objects = current_status.get('memory_store_objects', 0)
                        memory_mb = current_status.get('memory_store_size_estimate_mb', 0)
                        stats = data.get('statistics', {})
                        total_processed = stats.get('total_processed', 0)
                        logger.info(
                            f"📊 Monitor: {memory_objects} objects, {memory_mb}MB, {total_processed} total processed")
                    else:
                        logger.warning(f"⚠️  Monitor: Status check returned {response.status}")
            except Exception as e:
                logger.warning(f"⚠️  Monitor: Could not get service status: {e}")

            await asyncio.sleep(interval_seconds)

    async def run_baseline_comparison_phases(self, load_test: LoadTest) -> Dict[str, Any]:
        """Run the baseline comparison test with varying load phases for HPA vs KEDA vs DQN"""
        logger.info(f"🎯 Starting Production-Like Baseline Comparison Test")
        logger.info(
            f"📋 Pattern: Morning Rush(6min) → Peak Traffic(8min) → Sustained Business(10min) → Evening Cooldown(6min) = 30min")

        # Log the request mode being used
        if load_test.fire_and_forget:
            logger.info(f"🔥 Using FIRE-AND-FORGET mode - simulating production message queues/event streams")
        else:
            logger.info(f"🔄 Using ASYNC mode for precise timing control across all systems")

        # Production-like load phases (30 minutes total)
        phases = [
            {
                'name': 'Morning Rush Hour',
                'duration_minutes': 6,
                'requests_per_minute': 8,  # Higher initial load - production morning traffic
                'concurrent_requests': 3,
                'cpu_intensity': 6,
                'memory_size': 120,
                'request_duration': 20  # Typical API processing time
            },
            {
                'name': 'Peak Business Hours',
                'duration_minutes': 8,
                'requests_per_minute': 12,  # Peak traffic - lunch/business hours
                'concurrent_requests': 5,
                'cpu_intensity': 8,
                'memory_size': 180,
                'request_duration': 25  # Complex business logic processing
            },
            {
                'name': 'Sustained Operations',
                'duration_minutes': 10,
                'requests_per_minute': 6,  # Steady afternoon operations
                'concurrent_requests': 3,
                'cpu_intensity': 7,
                'memory_size': 150,
                'request_duration': 22  # Regular processing load
            },
            {
                'name': 'Evening Wind-Down',
                'duration_minutes': 6,
                'requests_per_minute': 3,  # End of day, lower traffic
                'concurrent_requests': 2,
                'cpu_intensity': 4,
                'memory_size': 90,
                'request_duration': 18  # Simpler end-of-day operations
            }
        ]

        total_start_time = time.time()
        all_results = []

        for i, phase in enumerate(phases, 1):
            logger.info(f"🏢 Phase {i}/4: {phase['name']} ({phase['duration_minutes']} minutes)")
            logger.info(
                f"   📊 Production Load: {phase['requests_per_minute']} req/min, CPU={phase['cpu_intensity']}, Memory={phase['memory_size']}MB")

            # Calculate timing for this phase
            total_requests = phase['requests_per_minute'] * phase['duration_minutes']

            # Calculate intervals between requests
            delay_between_requests = 60.0 / phase['requests_per_minute'] if phase['requests_per_minute'] > 0 else 60.0

            mode_description = "fire-and-forget" if load_test.fire_and_forget else "async"
            logger.info(f"   ⏱️  Timing: {delay_between_requests:.1f}s between requests ({mode_description} mode)")

            # Create LoadTest for this phase
            phase_test = LoadTest(
                name=f"Phase {i}: {phase['name']}",
                description=f"Production-like phase {i} - {phase['name']}",
                concurrent_requests=phase['concurrent_requests'],
                total_requests=total_requests,
                cpu_intensity=phase['cpu_intensity'],
                memory_size=phase['memory_size'],
                duration=phase['request_duration'],
                async_mode="true",  # Always use async for production-like behavior
                delay_between_requests=delay_between_requests,
                fire_and_forget=load_test.fire_and_forget  # Pass through fire-and-forget mode
            )

            # Run this phase using standard logic
            phase_start_time = time.time()
            semaphore = asyncio.Semaphore(phase_test.concurrent_requests)

            if load_test.fire_and_forget:
                # Fire-and-forget mode: fire all requests immediately with proper timing
                logger.info(
                    f"   🔥 Firing {phase_test.total_requests} requests over {phase['duration_minutes']} minutes (fire-and-forget)...")

                for j in range(phase_test.total_requests):
                    # Calculate exact fire time for this request
                    fire_time = j * phase_test.delay_between_requests

                    async def fire_request_at_time(request_id, delay):
                        await asyncio.sleep(delay)
                        return await self.send_process_request(phase_test, request_id)

                    # Schedule the request to fire at the exact time
                    asyncio.create_task(fire_request_at_time(j + 1, fire_time))

                # Wait for the phase duration, not for all requests to complete
                phase_duration_seconds = phase['duration_minutes'] * 60
                await asyncio.sleep(phase_duration_seconds)

                # Mock results for fire-and-forget mode
                results = [
                    {
                        'request_id': j + 1,
                        'status_code': 202,
                        'response_time': 0.001,  # Immediate
                        'response_data': {'status': 'fired', 'mode': 'fire_and_forget'},
                        'success': True
                    }
                    for j in range(phase_test.total_requests)
                ]
            else:
                # Original async mode logic
                tasks = []

                async def bounded_request(request_id: int):
                    async with semaphore:
                        # Schedule requests at proper intervals: 0s, 5s, 10s, 15s, etc.
                        await asyncio.sleep((request_id - 1) * phase_test.delay_between_requests)
                        return await self.send_process_request(phase_test, request_id)

                # Create all request tasks for this phase
                for j in range(phase_test.total_requests):
                    task = asyncio.create_task(bounded_request(j + 1))
                    tasks.append(task)

                # Wait for all requests in this phase to complete
                logger.info(
                    f"   🚀 Firing {phase_test.total_requests} requests over {phase['duration_minutes']} minutes...")
                results = await asyncio.gather(*tasks, return_exceptions=True)

            phase_end_time = time.time()
            actual_phase_duration = phase_end_time - phase_start_time

            # Process results for this phase
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
                'load_characteristics': {
                    'cpu_intensity': phase['cpu_intensity'],
                    'memory_size_mb': phase['memory_size'],
                    'request_duration': phase['request_duration']
                }
            }

            all_results.append(phase_summary)

            if load_test.fire_and_forget:
                logger.info(
                    f"   🔥 Phase {i} completed: {success_rate:.1f}% fired successfully in {actual_phase_duration / 60:.1f} minutes")
                logger.info(f"   ⚡ All requests fired without waiting - processing happening in background")
            else:
                logger.info(
                    f"   ✅ Phase {i} completed: {success_rate:.1f}% success rate in {actual_phase_duration / 60:.1f} minutes")
                logger.info(f"   📊 Avg response time: {avg_response_time:.1f}s")

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # Create overall summary
        total_requests = sum(r['total_requests'] for r in all_results)
        total_successful = sum(r['successful_requests'] for r in all_results)
        total_failed = sum(r['failed_requests'] for r in all_results)
        overall_success_rate = (total_successful / total_requests) * 100 if total_requests > 0 else 0

        baseline_summary = {
            'test_name': 'Production-Like Baseline Comparison',
            'total_duration_seconds': total_duration,
            'total_duration_minutes': total_duration / 60,
            'phases': all_results,
            'overall_total_requests': total_requests,
            'overall_successful_requests': total_successful,
            'overall_failed_requests': total_failed,
            'overall_success_rate': overall_success_rate,
            'overall_requests_per_second': total_requests / total_duration if total_duration > 0 else 0,
            'pattern_description': 'Production-like load phases: Rush Hour(8req/min) → Peak Business(12req/min) → Sustained Ops(6req/min) → Wind-Down(3req/min)',
            'mode': 'fire_and_forget' if load_test.fire_and_forget else 'async_with_ack'
        }

        logger.info(f"🎉 Production-Like Baseline test completed!")
        logger.info(f"📊 Total duration: {total_duration / 60:.1f} minutes")
        logger.info(f"📈 Overall success rate: {overall_success_rate:.1f}% ({total_successful}/{total_requests})")
        logger.info(f"🏢 Production pattern: Rush Hour → Peak Business → Sustained Operations → Wind-Down")

        if load_test.fire_and_forget:
            logger.info(f"🔥 Fire-and-forget mode: {total_requests} requests fired without waiting")
            logger.info(f"💼 Real-world equivalent: Message queue, event stream, or async API calls")

        self.results.append(baseline_summary)
        return baseline_summary

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
            logger.info(
                f"   📊 Load: {phase['requests_per_minute']} req/min, CPU={phase['cpu_intensity']}, Memory={phase['memory_size']}MB")

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
                    # Schedule requests at proper intervals: 0s, 30s, 60s, 90s, etc.
                    await asyncio.sleep((request_id - 1) * phase_test.delay_between_requests)
                    return await self.send_process_request(phase_test, request_id)

            # Create all request tasks for this phase
            for j in range(phase_test.total_requests):
                task = asyncio.create_task(bounded_request(j + 1))
                tasks.append(task)

            # Wait for all requests in this phase to complete
            logger.info(
                f"   ⏳ Sending {phase_test.total_requests} requests over {phase['duration_minutes']} minutes...")
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
        logger.info(f"📊 Total duration: {total_duration / 60:.1f} minutes")
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
        description="Fixed deterministic pattern for fair HPA vs DQN comparison with proper timing",
        concurrent_requests=6,
        total_requests=90,  # 3 requests per minute for 30 minutes
        cpu_intensity=7,
        memory_size=120,
        duration=20,  # Each request takes 20 seconds
        async_mode="false",  # SYNC mode for proper timing
        delay_between_requests=20.0  # 20 seconds between requests
    ),

    'baseline': LoadTest(
        name="Evaluation-Aligned Baseline Comparison Test",
        description="Load pattern aligned with 15-second evaluation windows for HPA vs KEDA vs DQN comparison",
        concurrent_requests=4,  # Will be handled by custom logic
        total_requests=150,  # Will be handled by custom logic (30 min * 5 avg req/min)
        cpu_intensity=6,  # Will be overridden by phases
        memory_size=150,  # Will be overridden by phases
        duration=26,  # Will be overridden by phases
        async_mode="true",  # Use async for precise timing control
        delay_between_requests=12.0  # Will be overridden by phases (avg 12s between requests)
    )
}


async def main():
    parser = argparse.ArgumentParser(description="Load Generator for Consumer Service")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the consumer service")
    parser.add_argument("--test", default=None,
                        help="Predefined test scenario (light, medium, heavy, burst, sustained, memory_stress, cpu_stress, comparison_pattern, baseline)")
    parser.add_argument("--test-name", default="Evaluation-Aligned Baseline Comparison Test", help="Name of the test")
    parser.add_argument("--description",
                        default="Load pattern aligned with 15-second evaluation windows for HPA vs KEDA vs DQN comparison",
                        help="Test description")
    parser.add_argument("--concurrent", type=int, default=4, help="Number of concurrent requests")
    parser.add_argument("--total", type=int, default=150, help="Total number of requests")
    parser.add_argument("--cpu-intensity", type=int, default=5, help="CPU intensity (1-10)")
    parser.add_argument("--memory-size", type=int, default=100, help="Memory size in MB")
    parser.add_argument("--duration", type=int, default=30, help="Request duration in seconds")
    parser.add_argument("--async-mode", action="store_true", help="Use async mode")
    parser.add_argument("--fire-and-forget", action="store_true",
                        help="Use fire-and-forget mode (pure async, no response waiting)")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between requests in seconds")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for deterministic testing")
    parser.add_argument("--monitor", action="store_true", help="Enable extended monitoring during test")
    parser.add_argument("--cleanup", action="store_true", help="Enable cleanup of service memory after test")

    args = parser.parse_args()

    # Set random seed for deterministic testing
    random.seed(args.seed)
    logger.info(f"🎲 Using deterministic seed: {args.seed}")

    # Create load test configuration
    if args.test and args.test in LOAD_TESTS:
        # Use predefined test scenario
        load_test = LOAD_TESTS[args.test]
        logger.info(f"📋 Using predefined test scenario: {args.test}")
    else:
        # Use custom parameters
        load_test = LoadTest(
            name=args.test_name,
            description=args.description,
            concurrent_requests=args.concurrent,
            total_requests=args.total,
            cpu_intensity=args.cpu_intensity,
            memory_size=args.memory_size,
            duration=args.duration,
            async_mode="true" if args.async_mode else "false",
            delay_between_requests=args.delay,
            fire_and_forget=args.fire_and_forget
        )
        if args.test:
            logger.warning(f"⚠️  Unknown test scenario '{args.test}', using custom parameters")

    # Log the mode being used
    if load_test.fire_and_forget:
        logger.info(f"🔥 Fire-and-forget mode enabled - requests will be fired without waiting for responses")
    elif load_test.async_mode == "true":
        logger.info(f"🔄 Async mode enabled - will wait for quick acknowledgment responses")
    else:
        logger.info(f"🔄 Sync mode enabled - will wait for full processing responses")

    # Run the load test
    async with LoadGenerator(args.url) as generator:
        # Health check
        if not await generator.health_check():
            logger.error("❌ Service health check failed")
            return

        # Cleanup before test (always performed)
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

        # Cleanup after test if requested
        if args.cleanup:
            logger.info("🧹 Post-test cleanup requested")
            await generator.cleanup_service()

        logger.info(f"✅ Load test '{load_test.name}' completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
