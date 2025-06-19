"""
Enhanced metrics endpoints for Prometheus integration with DQN features
"""

import os
import time
import psutil
import socket
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import Response
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, generate_latest,
    CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
)
from opentelemetry import trace

router = APIRouter()

# Enhanced Prometheus metrics for DQN features

# === SYSTEM METRICS (10 features) ===
cpu_usage_gauge = Gauge('nimbusguard_cpu_usage_percent', 'Current CPU usage percentage')
memory_usage_gauge = Gauge('nimbusguard_memory_usage_percent', 'Current memory usage percentage')

# NEW: Network I/O metrics
network_io_bytes_total = Counter('nimbusguard_network_io_bytes_total', 'Total network I/O bytes', ['direction'])
network_io_rate_gauge = Gauge('nimbusguard_network_io_rate_bytes_per_sec', 'Current network I/O rate')

# NEW: Disk I/O metrics
disk_io_bytes_total = Counter('nimbusguard_disk_io_bytes_total', 'Total disk I/O bytes', ['operation'])
disk_io_rate_gauge = Gauge('nimbusguard_disk_io_rate_bytes_per_sec', 'Current disk I/O rate')

# NEW: Container resource metrics
container_cpu_throttled_seconds_total = Counter('nimbusguard_container_cpu_throttled_seconds_total', 'Total CPU throttled time')
filesystem_usage_gauge = Gauge('nimbusguard_filesystem_usage_percent', 'Filesystem usage percentage')
container_memory_limit_bytes = Gauge('nimbusguard_container_memory_limit_bytes', 'Container memory limit')

# NEW: Process metrics
process_start_time_seconds = Gauge('nimbusguard_process_start_time_seconds', 'Process start time in seconds since epoch')
open_fds_gauge = Gauge('nimbusguard_open_file_descriptors', 'Number of open file descriptors')

# === APPLICATION METRICS (8 features) ===
# Existing metrics
workload_requests_total = Counter('nimbusguard_workload_requests_total', 'Total workload requests', ['type'])
workload_duration_seconds = Histogram('nimbusguard_workload_duration_seconds', 'Workload duration', ['type'])

# NEW: Enhanced HTTP metrics
http_requests_total = Counter('nimbusguard_http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
http_request_duration_seconds = Histogram(
    'nimbusguard_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

# NEW: Application-specific metrics
active_connections_gauge = Gauge('nimbusguard_active_connections', 'Number of active connections')
error_rate_gauge = Gauge('nimbusguard_error_rate_percent', 'Current error rate percentage')
request_rate_gauge = Gauge('nimbusguard_request_rate_per_sec', 'Current request rate per second')
throughput_gauge = Gauge('nimbusguard_throughput_requests_per_sec', 'Current throughput in requests per second')

# NEW: Queue and backlog metrics
queue_size_gauge = Gauge('nimbusguard_queue_size', 'Current queue size')
pending_requests_gauge = Gauge('nimbusguard_pending_requests', 'Number of pending requests')

# === BUSINESS METRICS (5 features) ===
# NEW: Business-level metrics
business_transactions_total = Counter('nimbusguard_business_transactions_total', 'Business transactions', ['type', 'status'])
user_sessions_active = Gauge('nimbusguard_user_sessions_active', 'Number of active user sessions')
workload_complexity_gauge = Gauge('nimbusguard_workload_complexity_score', 'Current workload complexity score')

# === SCALING EVENTS ===
scaling_events_total = Counter('nimbusguard_scaling_events_total', 'Total scaling events received', ['event_type'])

# === TRACE METRICS (for service graph integration) ===
trace_spans_total = Counter('nimbusguard_trace_spans_total', 'Total trace spans generated', ['operation'])
trace_errors_total = Counter('nimbusguard_trace_errors_total', 'Total trace errors', ['error_type'])

# Cache for rate calculations
_last_metrics_time = 0
_last_network_bytes = {'sent': 0, 'recv': 0}
_last_disk_bytes = {'read': 0, 'write': 0}
_last_request_count = 0
_error_count_window = []
_request_count_window = []


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
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f1, \
                 open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f2:
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


def get_cpu_throttling():
    """Get CPU throttling information"""
    try:
        with open("/sys/fs/cgroup/cpu.stat", "r") as f:
            for line in f:
                if line.startswith("throttled_time"):
                    return float(line.split()[1]) / 1_000_000_000  # Convert to seconds
    except Exception:
        try:
            with open("/sys/fs/cgroup/cpu/cpu.stat", "r") as f:
                for line in f:
                    if line.startswith("throttled_time"):
                        return float(line.split()[1]) / 1_000_000_000
        except Exception:
            pass
    return 0.0


def update_system_metrics():
    """Update system-level metrics"""
    global _last_metrics_time, _last_network_bytes, _last_disk_bytes

    current_time = time.time()
    process = psutil.Process(os.getpid())

    # CPU metrics
    cpu_percent = process.cpu_percent(interval=0.1)
    cpu_limit = get_container_cpu_limit()
    cpu_usage_gauge.set((cpu_percent / 100.0) * cpu_limit * 100)

    # Memory metrics
    memory_info = process.memory_info()
    memory_limit = get_container_memory_limit()
    memory_percent = (memory_info.rss / memory_limit) * 100
    memory_usage_gauge.set(memory_percent)
    container_memory_limit_bytes.set(memory_limit)

    # Network I/O metrics
    net_io = psutil.net_io_counters()
    if net_io:
        network_io_bytes_total.labels(direction='sent')._value._value = net_io.bytes_sent
        network_io_bytes_total.labels(direction='recv')._value._value = net_io.bytes_recv

        # Calculate rates
        if _last_metrics_time > 0:
            time_delta = current_time - _last_metrics_time
            if time_delta > 0:
                sent_rate = (net_io.bytes_sent - _last_network_bytes['sent']) / time_delta
                recv_rate = (net_io.bytes_recv - _last_network_bytes['recv']) / time_delta
                network_io_rate_gauge.set(sent_rate + recv_rate)

        _last_network_bytes = {'sent': net_io.bytes_sent, 'recv': net_io.bytes_recv}

    # Disk I/O metrics
    disk_io = psutil.disk_io_counters()
    if disk_io:
        disk_io_bytes_total.labels(operation='read')._value._value = disk_io.read_bytes
        disk_io_bytes_total.labels(operation='write')._value._value = disk_io.write_bytes

        # Calculate rates
        if _last_metrics_time > 0:
            time_delta = current_time - _last_metrics_time
            if time_delta > 0:
                read_rate = (disk_io.read_bytes - _last_disk_bytes['read']) / time_delta
                write_rate = (disk_io.write_bytes - _last_disk_bytes['write']) / time_delta
                disk_io_rate_gauge.set(read_rate + write_rate)

        _last_disk_bytes = {'read': disk_io.read_bytes, 'write': disk_io.write_bytes}

    # Filesystem usage
    try:
        disk_usage = psutil.disk_usage('/')
        filesystem_usage_gauge.set((disk_usage.used / disk_usage.total) * 100)
    except Exception:
        pass

    # CPU throttling
    throttled_time = get_cpu_throttling()
    container_cpu_throttled_seconds_total._value._value = throttled_time

    # File descriptors
    try:
        open_fds_gauge.set(process.num_fds())
    except Exception:
        pass

    # Process start time
    try:
        process_start_time_seconds.set(process.create_time())
    except Exception:
        pass

    _last_metrics_time = current_time


def update_application_metrics():
    """Update application-specific metrics"""
    global _error_count_window, _request_count_window

    current_time = time.time()

    # Clean old entries (keep last 60 seconds)
    cutoff_time = current_time - 60
    _error_count_window = [t for t in _error_count_window if t > cutoff_time]
    _request_count_window = [t for t in _request_count_window if t > cutoff_time]

    # Calculate rates
    error_rate = len(_error_count_window) / max(len(_request_count_window), 1) * 100
    request_rate = len(_request_count_window) / 60  # requests per second over last minute

    error_rate_gauge.set(error_rate)
    request_rate_gauge.set(request_rate)
    throughput_gauge.set(request_rate)  # Same as request rate for this app

    # Active connections (approximation)
    try:
        connections = len([conn for conn in psutil.net_connections() if conn.status == 'ESTABLISHED'])
        active_connections_gauge.set(connections)
    except Exception:
        active_connections_gauge.set(0)

    # Queue size (placeholder - would be actual queue in real app)
    queue_size_gauge.set(0)  # No actual queue in this demo app
    pending_requests_gauge.set(0)


@router.get("/metrics")
async def get_metrics():
    """Enhanced Prometheus metrics endpoint with DQN features"""

    # Update all metrics
    update_system_metrics()
    update_application_metrics()

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# === INSTRUMENTATION HELPERS ===

def record_http_request(method: str, endpoint: str, status_code: int, duration: float):
    """Record HTTP request metrics for DQN"""
    global _request_count_window, _error_count_window

    current_time = time.time()

    # Record in Prometheus
    http_requests_total.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

    # Record for rate calculations
    _request_count_window.append(current_time)
    if status_code >= 400:
        _error_count_window.append(current_time)

    # Business transaction tracking
    if endpoint.startswith('/api/v1/workload'):
        status = 'success' if status_code < 400 else 'error'
        business_transactions_total.labels(type='workload', status=status).inc()


def record_workload_metrics(workload_type: str, duration: float, complexity: float = 1.0):
    """Record workload-specific metrics"""
    workload_requests_total.labels(type=workload_type).inc()
    workload_duration_seconds.labels(type=workload_type).observe(duration)
    workload_complexity_gauge.set(complexity)


def record_scaling_event(event_type: str):
    """Record scaling event"""
    scaling_events_total.labels(event_type=event_type).inc()


def record_trace_metrics(operation: str, error_type: str = None):
    """Record trace-related metrics"""
    trace_spans_total.labels(operation=operation).inc()
    if error_type:
        trace_errors_total.labels(error_type=error_type).inc()


# === MIDDLEWARE FOR AUTOMATIC HTTP INSTRUMENTATION ===

async def metrics_middleware(request: Request, call_next):
    """Middleware to automatically instrument HTTP requests"""
    start_time = time.time()

    # Get current span for trace correlation
    span = trace.get_current_span()
    if span and span.is_recording():
        record_trace_metrics(f"{request.method} {request.url.path}")

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        # Record metrics
        record_http_request(
            method=request.method,
            endpoint=str(request.url.path),
            status_code=response.status_code,
            duration=duration
        )

        return response

    except Exception as e:
        duration = time.time() - start_time

        # Record error metrics
        record_http_request(
            method=request.method,
            endpoint=str(request.url.path),
            status_code=500,
            duration=duration
        )

        # Record trace error
        if span and span.is_recording():
            record_trace_metrics(f"{request.method} {request.url.path}", error_type=type(e).__name__)

        raise


# === METRIC HELPER FUNCTIONS ===

def increment_workload_counter(workload_type: str):
    """Legacy function for backward compatibility"""
    workload_requests_total.labels(type=workload_type).inc()


def record_workload_duration(workload_type: str, duration: float):
    """Legacy function for backward compatibility"""
    workload_duration_seconds.labels(type=workload_type).observe(duration)


def increment_scaling_event_counter(event_type: str):
    """Legacy function for backward compatibility"""
    scaling_events_total.labels(event_type=event_type).inc()