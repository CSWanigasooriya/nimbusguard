#!/usr/bin/env python3
"""
Test Prometheus API Connection and Consumer Metrics Discovery
===========================================================

This script tests the connection to Prometheus and shows available
consumer pod metrics to help validate the feature selector setup.
"""

import requests
import json
from typing import List, Dict
import argparse
import sys

def test_prometheus_connection(prometheus_url: str) -> bool:
    """Test basic connection to Prometheus API."""
    print(f"🔍 Testing connection to Prometheus at {prometheus_url}...")
    
    try:
        response = requests.get(f"{prometheus_url}/api/v1/query", 
                              params={"query": "up"}, 
                              timeout=5)
        response.raise_for_status()
        
        data = response.json()
        if data['status'] == 'success':
            print("  ✅ Prometheus API connection successful!")
            return True
        else:
            print(f"  ❌ Prometheus API error: {data.get('error', 'Unknown error')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Failed to connect to Prometheus at {prometheus_url}")
        print("     Make sure Prometheus is running and accessible")
        return False
    except requests.exceptions.Timeout:
        print(f"  ❌ Connection timeout to Prometheus")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def get_all_metrics(prometheus_url: str) -> List[str]:
    """Get all available metric names from Prometheus."""
    print("\n📊 Fetching all available metrics...")
    
    try:
        response = requests.get(f"{prometheus_url}/api/v1/label/__name__/values", 
                              timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data['status'] == 'success':
            metrics = data['data']
            print(f"  ✅ Found {len(metrics)} total metrics")
            return metrics
        else:
            print(f"  ❌ Error fetching metrics: {data.get('error', 'Unknown error')}")
            return []
            
    except Exception as e:
        print(f"  ❌ Error fetching metrics: {e}")
        return []

def filter_consumer_metrics(all_metrics: List[str]) -> Dict[str, List[str]]:
    """Filter for consumer pod relevant metrics."""
    print("\n🎯 Filtering for consumer pod relevant metrics...")
    
    consumer_patterns = [
        'http_request', 'http_response', 'http_duration', 'request_', 'response_',
        'consumer_', 'processing_', 'queue_', 'worker_',
        'process_cpu', 'process_memory', 'go_memstats', 'go_goroutines',
        'up', 'scrape_duration', 'kube_pod_', 'kube_deployment_', 'container_'
    ]
    
    categorized_metrics = {
        'load_metrics': [],
        'resource_metrics': [],
        'health_metrics': [],
        'application_metrics': []
    }
    
    categorization_rules = {
        'load_metrics': ['http_request', 'http_response', 'http_duration', 'request_', 'response_'],
        'resource_metrics': ['process_cpu', 'process_memory', 'go_memstats', 'go_goroutines', 'container_cpu', 'container_memory'],
        'health_metrics': ['up', 'scrape_duration', 'kube_pod_status', 'kube_deployment_status'],
        'application_metrics': ['consumer_', 'processing_', 'queue_', 'worker_', 'job_', 'task_']
    }
    
    # Filter metrics
    for metric in all_metrics:
        metric_lower = metric.lower()
        
        # Check if metric matches consumer patterns
        is_consumer_relevant = any(pattern.lower() in metric_lower for pattern in consumer_patterns)
        
        if is_consumer_relevant:
            # Categorize the metric
            categorized = False
            for category, patterns in categorization_rules.items():
                if any(pattern.lower() in metric_lower for pattern in patterns):
                    categorized_metrics[category].append(metric)
                    categorized = True
                    break
            
            # If not categorized, add to application metrics
            if not categorized:
                categorized_metrics['application_metrics'].append(metric)
    
    return categorized_metrics

def display_results(consumer_metrics: Dict[str, List[str]]):
    """Display the categorized consumer metrics."""
    print("\n📋 Consumer Pod Relevant Metrics by Category:")
    print("=" * 60)
    
    total_count = 0
    for category, metrics in consumer_metrics.items():
        if metrics:
            print(f"\n📊 {category.upper().replace('_', ' ')} ({len(metrics)} metrics):")
            total_count += len(metrics)
            
            # Show first 10 metrics
            for i, metric in enumerate(metrics[:10], 1):
                print(f"   {i:2d}. {metric}")
            
            if len(metrics) > 10:
                print(f"       ... and {len(metrics) - 10} more")
    
    print(f"\n✅ Total consumer-relevant metrics found: {total_count}")
    
    if total_count == 0:
        print("\n⚠️  No consumer-relevant metrics found!")
        print("   This might indicate:")
        print("   1. Consumer pod is not running or not exposing metrics")
        print("   2. Metrics are named differently than expected")
        print("   3. Prometheus is not scraping the consumer pod")

def main():
    parser = argparse.ArgumentParser(description="Test Prometheus connection and discover consumer metrics")
    parser.add_argument("--prometheus-url", type=str, default="http://localhost:9090",
                        help="Prometheus server URL")
    parser.add_argument("--show-all", action="store_true",
                        help="Show all metrics (not just consumer-relevant ones)")
    
    args = parser.parse_args()
    
    print("🧪 Prometheus Consumer Metrics Discovery Test")
    print("=" * 50)
    
    # Test connection
    if not test_prometheus_connection(args.prometheus_url):
        print("\n❌ Cannot proceed - Prometheus connection failed")
        sys.exit(1)
    
    # Get all metrics
    all_metrics = get_all_metrics(args.prometheus_url)
    if not all_metrics:
        print("\n❌ Cannot proceed - No metrics found")
        sys.exit(1)
    
    if args.show_all:
        print(f"\n📋 All {len(all_metrics)} metrics:")
        for i, metric in enumerate(all_metrics, 1):
            print(f"   {i:3d}. {metric}")
    
    # Filter and display consumer metrics
    consumer_metrics = filter_consumer_metrics(all_metrics)
    display_results(consumer_metrics)
    
    print(f"\n🎯 Ready to run feature selector with:")
    print(f"   python scripts/feature_selector.py --prometheus-url {args.prometheus_url}")

if __name__ == "__main__":
    main() 