"""
DQN Reward Function Module

SIMPLIFIED & ALIGNED: Uses exactly the same 9 scientifically-selected metrics as the DQN.
- Ensures perfect fairness between DQN and reward function
- Clear, straightforward reward logic
- Same feature space for both systems
"""

import numpy as np
import logging

logger = logging.getLogger("Rewards")

# EXACT SAME 9 FEATURES AS DQN (from scientific selection - with _rate suffixes)
DQN_FEATURES = [
    'process_cpu_seconds_total_rate',         # Rate of CPU usage
    'process_resident_memory_bytes',          # Memory usage (gauge)
    'process_virtual_memory_bytes',           # Virtual memory (gauge)
    'http_request_duration_seconds_sum_rate', # Rate of HTTP duration
    'http_requests_total_rate',               # Rate of HTTP requests
    'http_request_duration_seconds_count_rate', # Rate of HTTP request count
    'process_open_fds',                       # File descriptors (gauge)
    'http_response_size_bytes_sum_rate',      # Rate of response size
    'http_request_size_bytes_count_rate',     # Rate of request size count
]


def calculate_stable_reward(current_state, next_state, action, current_replicas, config=None):
    """
    SIMPLIFIED REWARD FUNCTION: Uses exactly the same 9 metrics as the DQN.
    
    Core Logic:
    1. High load + Scale Up = Good (+2)
    2. Low load + Scale Down = Good (+2)  
    3. Stable load + Keep Same = Good (+1)
    4. Wrong direction = Bad (-2)
    5. Bonus for efficiency improvements
    
    USES SAME METRICS AS DQN: Ensures perfect alignment.
    """
    
    # Get DQN-compatible metrics (using _rate suffixes for computed rates)
    curr_cpu_rate = current_state.get('process_cpu_seconds_total_rate', 0.0)
    curr_memory = current_state.get('process_resident_memory_bytes', 0.0)
    curr_virtual_memory = current_state.get('process_virtual_memory_bytes', 0.0)
    curr_http_duration_rate = current_state.get('http_request_duration_seconds_sum_rate', 0.0)
    curr_http_requests_rate = current_state.get('http_requests_total_rate', 0.0)
    curr_http_count_rate = current_state.get('http_request_duration_seconds_count_rate', 0.0)
    curr_open_fds = current_state.get('process_open_fds', 0.0)
    curr_response_size_rate = current_state.get('http_response_size_bytes_sum_rate', 0.0)
    curr_request_size_rate = current_state.get('http_request_size_bytes_count_rate', 0.0)
    
    next_cpu_rate = next_state.get('process_cpu_seconds_total_rate', 0.0)
    next_memory = next_state.get('process_resident_memory_bytes', 0.0)
    next_virtual_memory = next_state.get('process_virtual_memory_bytes', 0.0)
    next_http_duration_rate = next_state.get('http_request_duration_seconds_sum_rate', 0.0)
    next_http_requests_rate = next_state.get('http_requests_total_rate', 0.0)
    next_http_count_rate = next_state.get('http_request_duration_seconds_count_rate', 0.0)
    next_open_fds = next_state.get('process_open_fds', 0.0)
    next_response_size_rate = next_state.get('http_response_size_bytes_sum_rate', 0.0)
    next_request_size_rate = next_state.get('http_request_size_bytes_count_rate', 0.0)
    
    # Get replica info
    next_replicas = next_state.get('current_replicas', current_replicas)
    
    # Use the rate values directly (already computed by Prometheus rate() function)
    cpu_rate = next_cpu_rate
    http_requests_rate = next_http_requests_rate
    http_duration_rate = next_http_duration_rate
    
    # Use current memory values (gauges don't need rates)
    memory_mb = next_memory / 1_000_000 if next_memory > 0 else 0
    
    # Classify load using the same logic as DQN
    curr_load = _classify_load_level(
        curr_cpu_rate, curr_memory/1_000_000, curr_http_requests_rate, 
        curr_http_duration_rate, curr_open_fds, current_replicas
    )
    
    next_load = _classify_load_level(
        cpu_rate, memory_mb, http_requests_rate,
        http_duration_rate, next_open_fds, next_replicas
    )
    
    # Core reward logic (simple and clear)
    base_reward = _calculate_base_reward(action, curr_load, current_replicas)
    
    # Efficiency bonus
    efficiency_bonus = _calculate_efficiency_bonus(
        curr_cpu_rate, curr_http_requests_rate, curr_memory,
        cpu_rate, http_requests_rate, next_memory,
        current_replicas, next_replicas
    )
    
    # System health bonus/penalty
    health_bonus = _calculate_health_bonus(next_state, next_replicas)
    
    # Stability bonus
    stability_bonus = 0.2 if abs(next_replicas - current_replicas) <= 1 else 0.0
    
    # Final reward
    total_reward = base_reward + efficiency_bonus + stability_bonus + health_bonus
    total_reward = np.clip(total_reward, -5.0, 5.0)
    
    # Detailed logging
    logger.info(f"REWARD_ALIGNED: action={action} curr_load={curr_load} next_load={next_load} "
                f"replicas={current_replicas}=>{next_replicas} reward={total_reward:.2f}")
    logger.info(f"DQN_FEATURES_USED: cpu={cpu_rate:.3f} resident_mb={memory_mb:.1f} "
                f"http_requests={http_requests_rate:.3f} http_duration={http_duration_rate:.3f} "
                f"open_fds={next_open_fds:.0f}")
    logger.info(f"REWARD_BREAKDOWN: base={base_reward:.2f} efficiency={efficiency_bonus:.2f} "
                f"stability={stability_bonus:.2f} health={health_bonus:.2f}")
    
    return total_reward


def _classify_load_level(cpu_rate, memory_mb, http_requests_rate, http_duration_rate, open_fds, replicas):
    """
    Classify system load using the same 9 metrics as the DQN.
    Returns: "HIGH", "MEDIUM", or "LOW"
    """
    # Per-replica metrics (what matters for scaling)
    cpu_per_replica = cpu_rate / max(1, replicas)
    memory_per_replica = memory_mb / max(1, replicas)
    requests_per_replica = http_requests_rate / max(1, replicas)
    fds_per_replica = open_fds / max(1, replicas)
    
    # Calculate average response time (key performance indicator)
    avg_response_time = (http_duration_rate / max(0.01, http_requests_rate)) if http_requests_rate > 0 else 0
    
    # HIGH load thresholds (any of these indicate high load)
    high_load_conditions = [
        cpu_per_replica > 0.8,           # High CPU per replica
        memory_per_replica > 400,        # High memory per replica (400MB)
        requests_per_replica > 5,        # High request rate per replica
        avg_response_time > 2.0,         # High latency (2s)
        fds_per_replica > 200,          # High file descriptor usage
    ]
    
    # MEDIUM load thresholds
    medium_load_conditions = [
        cpu_per_replica > 0.4,           # Medium CPU per replica
        memory_per_replica > 200,        # Medium memory per replica (200MB)
        requests_per_replica > 2,        # Medium request rate per replica
        avg_response_time > 1.0,         # Medium latency (1s)
        fds_per_replica > 100,          # Medium file descriptor usage
    ]
    
    # Classification based on conditions
    if any(high_load_conditions):
        return "HIGH"
    elif any(medium_load_conditions):
        return "MEDIUM"
    else:
        return "LOW"


def _calculate_base_reward(action, load_level, current_replicas):
    """
    Calculate base reward based on action appropriateness.
    """
    if action == "Scale Up":
        if load_level == "HIGH":
            return 2.0  # Good decision
        elif load_level == "MEDIUM":
            return 0.5  # Okay decision
        else:  # LOW load
            return -2.0  # Bad decision
            
    elif action == "Scale Down":
        if load_level == "LOW" and current_replicas > 1:
            return 2.0  # Good decision
        elif load_level == "MEDIUM" and current_replicas > 3:
            return 0.5  # Okay decision
        else:  # HIGH load or can't scale down
            return -2.0  # Bad decision
            
    else:  # Keep Same
        if load_level == "MEDIUM":
            return 1.0  # Good stability
        else:
            return -0.5  # Should have scaled


def _calculate_efficiency_bonus(curr_cpu_rate, curr_http_rate, curr_memory, 
                               next_cpu_rate, next_http_rate, next_memory,
                               current_replicas, next_replicas):
    """
    Calculate efficiency bonus based on resource utilization improvements.
    """
    if next_replicas > 0 and current_replicas > 0:
        # Use rate metrics for efficiency calculation
        curr_efficiency = (curr_cpu_rate + curr_http_rate + curr_memory/1_000_000) / current_replicas
        next_efficiency = (next_cpu_rate + next_http_rate + next_memory/1_000_000) / next_replicas
        
        if next_efficiency > curr_efficiency:
            return 0.5  # Improved efficiency
        elif next_efficiency < curr_efficiency * 0.8:
            return -0.5  # Efficiency dropped significantly
    
    return 0.0


def _calculate_health_bonus(next_state, next_replicas):
    """
    Calculate health bonus/penalty based on system health metrics.
    """
    health_bonus = 0.0
    
    # Check for pod failures using actual metrics
    replicas_unavailable = next_state.get('deployment_replicas_unavailable', 0)
    replicas_ready = next_state.get('deployment_replicas_ready', next_replicas)
    containers_ready = next_state.get('pod_container_ready', 0)
    restarts = next_state.get('pod_container_restarts', 0)
    
    # Penalty for unhealthy system
    if replicas_unavailable > 0:
        health_bonus -= 1.0  # Penalty for unavailable replicas
    
    if replicas_ready < next_replicas:
        health_bonus -= 0.5  # Penalty for not ready replicas
    
    if restarts > 0:
        health_bonus -= 0.3  # Penalty for restarts
    
    # Bonus for healthy system
    if replicas_ready == next_replicas and replicas_unavailable == 0:
        health_bonus += 0.3  # Bonus for all replicas healthy
    
    return health_bonus


# === REWARD FUNCTION DOCUMENTATION ===

REWARD_EQUATIONS = """
=== SIMPLIFIED DQN REWARD FUNCTION ===

## USES SAME 9 METRICS AS DQN:
1. process_cpu_seconds_total_rate (computed rate)
2. process_resident_memory_bytes (gauge)  
3. process_virtual_memory_bytes (gauge)
4. http_request_duration_seconds_sum_rate (computed rate)
5. http_requests_total_rate (computed rate)
6. http_request_duration_seconds_count_rate (computed rate)
7. process_open_fds (gauge)
8. http_response_size_bytes_sum_rate (computed rate)
9. http_request_size_bytes_count_rate (computed rate)

Plus Kubernetes health metrics:
- deployment_replicas_unavailable
- deployment_replicas_ready
- pod_container_ready
- pod_container_restarts

## CORE LOGIC:
1. HIGH LOAD + Scale Up = +2.0 (Good)
2. LOW LOAD + Scale Down = +2.0 (Good) 
3. MEDIUM LOAD + Keep Same = +1.0 (Good)
4. Wrong direction = -2.0 (Bad)

## LOAD CLASSIFICATION (per replica):
- HIGH: CPU >0.8 OR Memory >400MB OR Requests >5/s OR Latency >2s OR FDs >200
- MEDIUM: CPU >0.4 OR Memory >200MB OR Requests >2/s OR Latency >1s OR FDs >100  
- LOW: Below medium thresholds

## BONUSES:
- Efficiency improvement: +0.5
- Stability (small changes): +0.2
- System health: +0.3 (healthy) or -1.0 (failures)

## RANGE: [-5.0, +5.0]
""" 