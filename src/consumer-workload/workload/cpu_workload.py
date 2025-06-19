import logging
import math
import os
import threading
import time

import psutil

logger = logging.getLogger(__name__)


def get_container_cpu_limit():
    # Try cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            quota, period = f.read().strip().split()
            if quota == "max":
                return os.cpu_count() or 1
            return float(quota) / float(period)
    except Exception:
        pass
    # Try cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f1, open("/sys/fs/cgroup/cpu/cpu.cfs_period_us",
                                                                          "r") as f2:
            quota = int(f1.read().strip())
            period = int(f2.read().strip())
            if quota == -1:
                return os.cpu_count() or 1
            return float(quota) / float(period)
    except Exception:
        pass
    # Fallback
    return os.cpu_count() or 1


class CPUWorkloadGenerator:
    def __init__(self):
        self._is_running = False
        self._thread = None
        self._target_usage = 0.0
        self._duration = 0
        self._start_time = 0.0
        self._initial_usage = 0.0

    def _workload(self, target_usage, duration):
        # _start_time is now set in generate_load, so use it for end_time calculation
        end_time = self._start_time + duration
        self._is_running = True

        while self._is_running and time.time() < end_time:
            elapsed = time.time() - self._start_time
            # Calculate current target using exponential growth
            current_target = self._initial_usage + (target_usage - self._initial_usage) * (
                    1 - math.exp(-3 * elapsed / duration))

            busy_time = current_target / 100.0
            idle_time = 1.0 - busy_time
            start = time.time()
            # Busy loop
            while (time.time() - start) < busy_time:
                pass
            # Sleep for the rest of the second
            if idle_time > 0:
                time.sleep(idle_time)
        self._is_running = False

    def generate_load(self, target_usage: float, duration: int, initial_usage: float = 0.0) -> None:
        if self._is_running:
            self.stop()
        self._target_usage = target_usage
        self._duration = duration
        self._initial_usage = initial_usage
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
        process_percent = psutil.Process(os.getpid()).cpu_percent(interval=0.1)
        num_cpus = get_container_cpu_limit()
        # Report as a percentage of all CPUs available to the container
        return (process_percent / 100.0) * num_cpus * 100

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

    @property
    def max_cpu_percentage(self):
        return get_container_cpu_limit() * 100
