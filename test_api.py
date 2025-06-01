#!/usr/bin/env python3
"""
Load test script for NimbusGuard Heavy Processing API.
"""

import requests
import json
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor

# Base URL for the API
BASE_URL = "http://localhost:8000"

def generate_test_data():
    """Generate random test data for heavy processing"""
    return {
        "user_id": random.randint(1, 10000),
        "dataset": list(range(random.randint(10, 100))),
        "text_data": f"Sample text data for processing {random.randint(1, 1000)}",
        "numerical_values": [random.uniform(1.0, 100.0) for _ in range(random.randint(5, 20))],
        "operation_type": random.choice(["heavy_compute", "data_transform", "matrix_ops"]),
        "metadata": {
            "source": "load_test",
            "batch_id": random.randint(1, 100),
            "complexity": random.choice(["high", "very_high", "extreme"])
        },
        "string_array": [f"string_{i}" for i in range(random.randint(10, 50))],
        "nested_data": {
            "level1": {
                "level2": {
                    "values": list(range(random.randint(20, 50)))
                }
            }
        }
    }

def test_heavy_processing():
    """Test the heavy processing endpoint"""
    data = {
        "data": generate_test_data(),
        "request_id": f"test_{int(time.time())}_{random.randint(1000, 9999)}"
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/process", json=data, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request {data['request_id']}: {result['processing_time_ms']:.2f}ms, {result['operations_performed']} ops")
            return True, end_time - start_time
        else:
            print(f"❌ Request {data['request_id']}: HTTP {response.status_code}")
            return False, end_time - start_time
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False, 0

def worker_thread(num_requests, thread_id):
    """Worker thread for load testing"""
    successful = 0
    failed = 0
    total_time = 0
    response_times = []
    
    print(f"🧵 Thread {thread_id} starting {num_requests} requests...")
    
    for i in range(num_requests):
        success, response_time = test_heavy_processing()
        if success:
            successful += 1
            response_times.append(response_time)
        else:
            failed += 1
        
        total_time += response_time
        
        # Small delay between requests
        time.sleep(0.2)
    
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    print(f"🏁 Thread {thread_id} completed: {successful} successful, {failed} failed, avg: {avg_response_time:.2f}s")
    return successful, failed, response_times

def basic_test():
    """Run basic functionality test"""
    print("🧪 Running basic functionality test...")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Error response: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
        return False
    
    # Test heavy processing
    print("\n2. Testing heavy processing...")
    try:
        test_data = {
            "data": {
                "test_user": "load_test_user",
                "test_numbers": [1, 2, 3, 4, 5],
                "test_string": "Hello heavy processing!",
                "test_object": {"nested": "value"}
            },
            "request_id": "basic_test_001"
        }
        
        print("   Sending request (this may take a few seconds)...")
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/process", json=test_data, timeout=30)
        end_time = time.time()
        
        print(f"   Status: {response.status_code}")
        print(f"   Response time: {end_time - start_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Processing time: {result['processing_time_ms']}ms")
            print(f"   Operations performed: {result['operations_performed']}")
            print(f"   Result summary: {result['result']}")
            return True
        else:
            print(f"   Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   Error: {e}")
        return False

def load_test(num_threads=3, requests_per_thread=5):
    """Run load test with multiple threads"""
    print(f"\n🚀 Running heavy load test:")
    print(f"   Threads: {num_threads}")
    print(f"   Requests per thread: {requests_per_thread}")
    print(f"   Total requests: {num_threads * requests_per_thread}")
    print(f"   Expected duration: ~{num_threads * requests_per_thread * 2}+ seconds")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        
        for i in range(num_threads):
            future = executor.submit(worker_thread, requests_per_thread, f"heavy_{i}")
            futures.append(future)
        
        # Collect results
        total_successful = 0
        total_failed = 0
        all_response_times = []
        
        for future in futures:
            successful, failed, response_times = future.result()
            total_successful += successful
            total_failed += failed
            all_response_times.extend(response_times)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Calculate statistics
    if all_response_times:
        avg_response_time = sum(all_response_times) / len(all_response_times)
        min_response_time = min(all_response_times)
        max_response_time = max(all_response_times)
        
        # Calculate percentiles
        sorted_times = sorted(all_response_times)
        p50 = sorted_times[int(0.5 * len(sorted_times))]
        p95 = sorted_times[int(0.95 * len(sorted_times))] if len(sorted_times) > 20 else max_response_time
    else:
        avg_response_time = min_response_time = max_response_time = p50 = p95 = 0
    
    print(f"\n📊 Heavy Load Test Results:")
    print(f"   Total duration: {duration:.2f} seconds")
    print(f"   Total requests: {total_successful + total_failed}")
    print(f"   Successful: {total_successful}")
    print(f"   Failed: {total_failed}")
    print(f"   Success rate: {(total_successful / (total_successful + total_failed) * 100):.1f}%")
    print(f"   Requests per second: {(total_successful + total_failed) / duration:.2f}")
    print(f"\n📈 Response Time Statistics:")
    print(f"   Average: {avg_response_time:.2f}s")
    print(f"   Minimum: {min_response_time:.2f}s")
    print(f"   Maximum: {max_response_time:.2f}s")
    print(f"   50th percentile: {p50:.2f}s")
    print(f"   95th percentile: {p95:.2f}s")

def main():
    print("🎯 NimbusGuard Heavy Processing Load Test")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not responding correctly")
            return
    except:
        print("❌ Cannot connect to API. Make sure it's running on http://localhost:8000")
        print("   Run: python main.py")
        return
    
    print("✅ Heavy Processing API is running")
    
    # Run basic test first
    if not basic_test():
        print("❌ Basic test failed. Check the API.")
        return
    
    print("✅ Basic test passed")
    
    # Ask user for load test
    print("\n" + "="*50)
    choice = input("Run heavy load test? (y/n): ").lower().strip()
    
    if choice == 'y':
        try:
            threads = int(input("Number of threads (default 3, recommended max 5): ") or "3")
            requests = int(input("Requests per thread (default 5, each takes ~2-5 seconds): ") or "5")
            
            if threads > 10:
                print("⚠️  Warning: High thread count may overwhelm the API")
            if requests > 10:
                print("⚠️  Warning: This will take a long time with heavy processing")
                
            load_test(threads, requests)
        except ValueError:
            print("Using default values...")
            load_test()
    
    print("\n📝 Check app.log file for detailed processing logs")
    print("📊 Visit http://localhost:8000/stats for API statistics")
    print("📋 Visit http://localhost:8000/docs for API documentation")

if __name__ == "__main__":
    main()
