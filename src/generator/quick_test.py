#!/usr/bin/env python3
"""
Quick KEDA Test Script

A simple script to quickly trigger KEDA autoscaling for immediate testing.
"""

import asyncio
import time

import aiohttp


async def quick_keda_test():
    """Run a quick test to trigger KEDA scaling"""

    print("🚀 NimbusGuard KEDA Quick Test")
    print("=" * 40)

    url = "http://localhost:8000"

    async with aiohttp.ClientSession() as session:
        # Health check
        print("🔍 Checking service health...")
        try:
            async with session.get(f"{url}/health") as response:
                if response.status == 200:
                    print("✅ Service is healthy!")
                else:
                    print(f"❌ Service health check failed: {response.status}")
                    return
        except Exception as e:
            print(f"❌ Cannot connect to service: {e}")
            print("💡 Make sure to run 'make forward' first!")
            return

        print("\n📊 Starting load generation to trigger KEDA scaling...")
        print("💡 Watch scaling with: kubectl get pods -n nimbusguard -w")
        print("")

        # Generate sustained load
        tasks = []
        for i in range(10):
            task = asyncio.create_task(send_heavy_request(session, url, i + 1))
            tasks.append(task)

        # Wait a bit between starting tasks
        await asyncio.sleep(1)

        print("⏳ Processing 10 concurrent heavy requests...")
        print("🎯 Each request: CPU=9, Memory=150MB, Duration=30s")
        print("")

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # Summary
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
        total_time = end_time - start_time

        print(f"\n🎉 Test completed in {total_time:.1f}s")
        print(f"✅ Successful requests: {successful}/10")
        print(f"📈 This should have triggered KEDA scaling!")
        print(f"🔍 Check: kubectl get pods -n nimbusguard")


async def send_heavy_request(session, base_url, request_id):
    """Send a heavy processing request"""
    params = {
        'cpu_intensity': 9,
        'memory_size': 150,
        'duration': 30,
        'async_mode': True  # Run in background to not block response
    }

    try:
        start = time.time()
        async with session.post(f"{base_url}/process", params=params, timeout=60) as response:
            end = time.time()
            data = await response.json()

            print(f"  ✅ Request {request_id}: {response.status} in {end - start:.1f}s")
            return {
                'request_id': request_id,
                'success': response.status == 200,
                'response_time': end - start,
                'data': data
            }
    except Exception as e:
        print(f"  ❌ Request {request_id}: {e}")
        return {
            'request_id': request_id,
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    print("🎯 Quick KEDA Autoscaling Test")
    print("This will generate heavy load to trigger scaling")
    print("")

    try:
        asyncio.run(quick_keda_test())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")

    print("\n📚 For more advanced testing, use: python load_generator.py")
