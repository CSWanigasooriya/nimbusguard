"""
Metrics endpoints for Prometheus integration
"""

import os

import psutil
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

router = APIRouter()

# Prometheus metrics
cpu_usage_gauge = Gauge('nimbusguard_cpu_usage_percent', 'Current CPU usage percentage')
memory_usage_gauge = Gauge('nimbusguard_memory_usage_percent', 'Current memory usage percentage')
workload_requests_total = Counter('nimbusguard_workload_requests_total', 'Total workload requests', ['type'])
workload_duration_seconds = Histogram('nimbusguard_workload_duration_seconds', 'Workload duration', ['type'])
scaling_events_total = Counter('nimbusguard_scaling_events_total', 'Total scaling events received', ['event_type'])


def get_container_cpu_limit():
    """Get CPU limit from cgroup"""
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            quota, period = f.read().strip().split()
            if quota == "max":
                return os.cpu_count() or 1
            return float(quota) / float(period)
    except Exception:
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f1, open("/sys/fs/cgroup/cpu/cpu.cfs_period_us",
                                                                              "r") as f2:
                quota = int(f1.read().strip())
                period = int(f2.read().strip())
                if quota == -1:
                    return os.cpu_count() or 1
                return float(quota) / float(period)
        except Exception:
            return os.cpu_count() or 1


def get_container_memory_limit():
    """Get memory limit from cgroup"""
    try:
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            val = f.read().strip()
            if val.isdigit() and int(val) < 1 << 60:
                return int(val)
    except Exception:
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                val = f.read().strip()
                if val.isdigit() and int(val) < 1 << 60:
                    return int(val)
        except Exception:
            pass
    return psutil.virtual_memory().total


@router.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""

    # Update CPU metrics
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=0.1)
    cpu_limit = get_container_cpu_limit()
    cpu_usage_gauge.set((cpu_percent / 100.0) * cpu_limit * 100)

    # Update memory metrics
    memory_info = process.memory_info()
    memory_limit = get_container_memory_limit()
    memory_percent = (memory_info.rss / memory_limit) * 100
    memory_usage_gauge.set(memory_percent)

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def increment_workload_counter(workload_type: str):
    """Increment workload request counter"""
    workload_requests_total.labels(type=workload_type).inc()


def record_workload_duration(workload_type: str, duration: float):
    """Record workload duration"""
    workload_duration_seconds.labels(type=workload_type).observe(duration)


def increment_scaling_event_counter(event_type: str):
    """Increment scaling event counter"""
    scaling_events_total.labels(event_type=event_type).inc()
