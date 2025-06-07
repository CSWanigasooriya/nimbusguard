import time
import threading
import psutil
import logging
import os
import numpy as np
import math
from typing import Optional, List

logger = logging.getLogger(__name__)

def get_container_memory_usage() -> int:
    """Get the current memory usage of the container in bytes"""
    try:
        # Try cgroup v2 first
        with open("/sys/fs/cgroup/memory.current", "r") as f:
            return int(f.read().strip())
    except Exception:
        try:
            # Try cgroup v1
            with open("/sys/fs/cgroup/memory/memory.usage_in_bytes", "r") as f:
                return int(f.read().strip())
        except Exception:
            # Fallback to process memory
            return psutil.Process(os.getpid()).memory_info().rss

def get_container_memory_limit() -> int:
    # Try cgroup v2 first
    try:
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            val = f.read().strip()
            if val.isdigit() and int(val) < 1 << 60:  # Ignore 'max' or very large values
                return int(val)
    except Exception:
        pass
    # Try cgroup v1
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
            val = f.read().strip()
            if val.isdigit() and int(val) < 1 << 60:
                return int(val)
    except Exception:
        pass
    # Fallback to system total
    return psutil.virtual_memory().total

class MemoryWorkloadGenerator:
    def __init__(self):
        self._is_running = False
        self._thread = None
        self._target_usage = 0.0
        self._duration = 0
        self._start_time = 0.0
        self._allocated = []
        self._total_memory = get_container_memory_limit()
        
        logger.info("MemoryWorkloadGenerator initialized", extra={
            "total_memory": self._total_memory,
            "pid": os.getpid()
        })

    def _log_memory_usage(self):
        """Log the memory usage of the current process"""
        current_usage = get_container_memory_usage()
        memory_percent = (current_usage / self._total_memory) * 100
        
        logger.info("Memory usage metrics", extra={
            "memory_percent": memory_percent,
            "memory_usage_bytes": current_usage,
            "memory_limit_bytes": self._total_memory,
            "num_threads": psutil.Process(os.getpid()).num_threads(),
            "pid": os.getpid()
        })

    def _workload(self, target_usage, duration):
        # _start_time is now set in generate_load, so use it for end_time calculation  
        end_time = self._start_time + duration
        self._is_running = True
        
        while self._is_running and time.time() < end_time:
            elapsed = time.time() - self._start_time
            # Calculate current target using exponential growth
            current_target = (target_usage) * (1 - math.exp(-3 * elapsed / duration))
            target_bytes = int((current_target / 100.0) * self._total_memory)
            
            # Calculate how much memory to allocate/deallocate
            current_bytes = sum(len(chunk) for chunk in self._allocated)
            bytes_to_allocate = target_bytes - current_bytes
            
            if bytes_to_allocate > 0:
                # Allocate more memory
                try:
                    new_chunks = [bytearray(1024*1024) for _ in range(bytes_to_allocate // (1024*1024))]
                    self._allocated.extend(new_chunks)
                except MemoryError:
                    pass
            elif bytes_to_allocate < 0:
                # Deallocate memory
                bytes_to_remove = abs(bytes_to_allocate)
                chunks_to_remove = bytes_to_remove // (1024*1024)
                self._allocated = self._allocated[:-chunks_to_remove] if chunks_to_remove > 0 else self._allocated
            
            time.sleep(0.5)
            
        self._allocated = []  # Release memory
        self._is_running = False

    def generate_load(self, target_usage: float, duration: int, initial_usage: float = 0.0) -> None:
        if self._is_running:
            self.stop()
        self._target_usage = target_usage
        self._duration = duration
        self._start_time = time.time()  # Set start time immediately
        self._thread = threading.Thread(target=self._workload, args=(target_usage, duration))
        self._thread.start()

    def stop(self):
        self._is_running = False
        if self._thread:
            self._thread.join()
            self._thread = None

    @property
    def is_running(self):
        return self._is_running

    @property
    def current_usage(self):
        """Get current memory usage as a percentage of container limit"""
        current_usage = get_container_memory_usage()
        return (current_usage / self._total_memory) * 100

    @property
    def remaining_time(self):
        if not self._is_running:
            return 0.0
        elapsed = time.time() - self._start_time
        return max(0, self._duration - elapsed)

    @property
    def target_usage(self):
        return self._target_usage

    @property
    def duration(self):
        return self._duration

    def _cleanup(self) -> None:
        """Clean up allocated memory"""
        logger.info("Cleaning up memory workload resources", extra={
            "allocated_chunks": len(self._allocated)
        })
        self._allocated.clear()

    @property
    def total_memory(self) -> int:
        """Get total memory limit"""
        return self._total_memory

    def _get_container_memory_limit(self):
        # Try cgroup v2 first
        try:
            with open("/sys/fs/cgroup/memory.max", "r") as f:
                val = f.read().strip()
                if val.isdigit() and int(val) < 1 << 60:
                    return int(val)
        except Exception:
            pass
        # Try cgroup v1
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                val = f.read().strip()
                if val.isdigit() and int(val) < 1 << 60:
                    return int(val)
        except Exception:
            pass
        return psutil.virtual_memory().total

    def _get_container_memory_usage(self):
        # Try cgroup v2 first
        try:
            with open("/sys/fs/cgroup/memory.max", "r") as f:
                val = f.read().strip()
                if val.isdigit() and int(val) < 1 << 60:
                    return int(val)
        except Exception:
            pass
        # Try cgroup v1
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                val = f.read().strip()
                if val.isdigit() and int(val) < 1 << 60:
                    return int(val)
        except Exception:
            pass
        return psutil.virtual_memory().total

    def _get_container_memory_usage_percent(self):
        # Try cgroup v2 first
        try:
            with open("/sys/fs/cgroup/memory.max", "r") as f:
                val = f.read().strip()
                if val.isdigit() and int(val) < 1 << 60:
                    return int(val)
        except Exception:
            pass
        # Try cgroup v1
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                val = f.read().strip()
                if val.isdigit() and int(val) < 1 << 60:
                    return int(val)
        except Exception:
            pass
        return psutil.virtual_memory().percent

    def _get_container_memory_usage_rss(self):
        # Try cgroup v2 first
        try:
            with open("/sys/fs/cgroup/memory.max", "r") as f:
                val = f.read().strip()
                if val.isdigit() and int(val) < 1 << 60:
                    return int(val)
        except Exception:
            pass
        # Try cgroup v1
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                val = f.read().strip()
                if val.isdigit() and int(val) < 1 << 60:
                    return int(val)
        except Exception:
            pass
        return psutil.virtual_memory().rss

    def _get_container_memory_usage_vms(self):
        # Try cgroup v2 first
        try:
            with open("/sys/fs/cgroup/memory.max", "r") as f:
                val = f.read().strip()
                if val.isdigit() and int(val) < 1 << 60:
                    return int(val)
        except Exception:
            pass
        # Try cgroup v1
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                val = f.read().strip()
                if val.isdigit() and int(val) < 1 << 60:
                    return int(val)
        except Exception:
            pass
        return psutil.virtual_memory().vms 