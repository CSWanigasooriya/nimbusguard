"""
DQN Reward Function Module

Consumer-aligned reward functions that use the same metrics the DQN sees for decisions.
Implements balanced incentives for appropriate scaling behavior.

Key Features:
- Uses consumer pod metrics (same as DQN input)
- Rewards appropriate scaling decisions
- Penalizes over/under-provisioning
- Enables proactive scaling
- Balanced component weights
"""

import numpy as np
import logging

logger = logging.getLogger("Rewards")


def calculate_stable_reward(current_state, next_state, action, current_replicas, config=None):
    """
    CONSUMER-ALIGNED REWARD FUNCTION: Uses the same metrics DQN sees for decisions.
    
    Rewards:
    - Scaling up when consumer pods show high load/pressure
    - Scaling down when consumer pods show low load/efficiency
    - Keeping same when load is stable and appropriate
    
    Penalizes:
    - Scaling up when load is low (over-provisioning)
    - Scaling down when load is high (under-provisioning)  
    - Keeping same when scaling is clearly needed
    """
    
    # === EXTRACT CONSUMER POD METRICS (same as DQN input) ===
    # Current state consumer metrics
    curr_cpu_rate = current_state.get('process_cpu_seconds_total_rate', 0.0)
    curr_resident_memory = current_state.get('process_resident_memory_bytes', 0.0)
    curr_virtual_memory = current_state.get('process_virtual_memory_bytes', 0.0)
    curr_http_duration = current_state.get('http_request_duration_seconds_sum_rate', 0.0)
    curr_http_requests = current_state.get('http_requests_total_rate', 0.0)
    curr_http_count = current_state.get('http_request_duration_seconds_count_rate', 0.0)
    curr_open_fds = current_state.get('process_open_fds', 0.0)
    curr_response_size = current_state.get('http_response_size_bytes_sum_rate', 0.0)
    curr_request_size = current_state.get('http_request_size_bytes_count_rate', 0.0)
    
    # Next state consumer metrics
    next_cpu_rate = next_state.get('process_cpu_seconds_total_rate', 0.0)
    next_resident_memory = next_state.get('process_resident_memory_bytes', 0.0)
    next_virtual_memory = next_state.get('process_virtual_memory_bytes', 0.0)
    next_http_duration = next_state.get('http_request_duration_seconds_sum_rate', 0.0)
    next_http_requests = next_state.get('http_requests_total_rate', 0.0)
    next_http_count = next_state.get('http_request_duration_seconds_count_rate', 0.0)
    next_open_fds = next_state.get('process_open_fds', 0.0)
    next_response_size = next_state.get('http_response_size_bytes_sum_rate', 0.0)
    next_request_size = next_state.get('http_request_size_bytes_count_rate', 0.0)
    
    # Calculate replica changes
    next_replicas = next_state.get('current_replicas', current_replicas)
    replica_change = next_replicas - current_replicas
    
    # === LOAD INTENSITY ANALYSIS ===
    def calculate_load_intensity(cpu_rate, resident_memory, virtual_memory, http_requests, http_duration, open_fds, response_size):
        """Calculate overall load intensity from consumer pod metrics with ADAPTIVE thresholds."""
        
        # ADAPTIVE THRESHOLDS: Learn from current system baseline instead of hardcoded values
        # Calculate dynamic thresholds based on current system state and historical patterns
        
        # CPU pressure (adaptive based on current system capacity)
        # Use current_replicas to estimate expected CPU baseline
        expected_cpu_baseline = current_replicas * 0.1  # 0.1 CPU per replica as baseline
        cpu_threshold = max(expected_cpu_baseline * 2.0, 1.0)  # At least 1.0 as minimum
        cpu_pressure = min(1.0, cpu_rate / cpu_threshold)
        
        # Memory pressure (adaptive based on actual memory usage patterns)
        # Convert bytes to MB for easier threshold management
        resident_memory_mb = resident_memory / 1000000 if resident_memory > 0 else 0
        virtual_memory_mb = virtual_memory / 1000000 if virtual_memory > 0 else 0
        
        # Use replica count to normalize - more replicas = more expected memory usage
        expected_memory_per_replica = 100  # 100MB per replica baseline
        memory_threshold = max(expected_memory_per_replica * current_replicas * 2.0, 200.0)  # At least 200MB minimum
        memory_pressure = min(1.0, resident_memory_mb / memory_threshold)
        
        # Virtual memory pressure (separate threshold)
        expected_virtual_per_replica = 200  # 200MB virtual per replica baseline
        virtual_threshold = max(expected_virtual_per_replica * current_replicas * 2.0, 400.0)  # At least 400MB minimum
        virtual_pressure = min(1.0, virtual_memory_mb / virtual_threshold)
        
        # HTTP load pressure (adaptive based on current throughput capacity)
        # Calculate expected capacity based on current replica count
        expected_req_capacity = current_replicas * 10.0  # 10 req/sec per replica baseline
        http_threshold = max(expected_req_capacity * 1.5, 20.0)  # At least 20 req/sec minimum
        http_load = min(1.0, http_requests / http_threshold)
        
        # Latency pressure (adaptive based on current performance)
        # Calculate average response time and use adaptive threshold
        avg_response_time = (http_duration / max(0.1, http_requests)) if http_requests > 0 else 0
        # Adaptive latency threshold: higher replicas should handle load better
        latency_threshold = max(1.0 / current_replicas, 0.5)  # Better performance expected with more replicas
        latency_pressure = min(1.0, avg_response_time / latency_threshold)
        
        # System resource pressure (adaptive based on replica count)
        # More replicas = more expected file descriptors
        expected_fd_baseline = current_replicas * 50  # 50 FDs per replica baseline
        fd_threshold = max(expected_fd_baseline * 2.0, 200.0)  # At least 200 FDs minimum
        fd_pressure = min(1.0, open_fds / fd_threshold)
        
        # Network pressure (adaptive based on throughput and replica count)
        # More replicas should handle more network traffic
        expected_network_capacity = current_replicas * 100000.0  # 100KB/s per replica baseline
        network_threshold = max(expected_network_capacity * 2.0, 500000.0)  # At least 500KB/s minimum
        network_pressure = min(1.0, response_size / network_threshold)
        
        # Weighted combination (updated weights for memory metrics)
        load_intensity = (
            cpu_pressure * 0.25 +         # CPU usage (adaptive)
            memory_pressure * 0.15 +      # Resident memory pressure (adaptive)
            virtual_pressure * 0.10 +     # Virtual memory pressure (adaptive)
            http_load * 0.25 +            # Request volume (adaptive)
            latency_pressure * 0.15 +     # Response time (adaptive)
            fd_pressure * 0.05 +          # System resources (adaptive)
            network_pressure * 0.05       # Network throughput (adaptive)
        )
        
        # Log adaptive thresholds for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"ADAPTIVE_THRESHOLDS: cpu={cpu_threshold:.2f} memory={memory_threshold:.2f}MB "
                        f"virtual={virtual_threshold:.2f}MB http={http_threshold:.2f} latency={latency_threshold:.3f} "
                        f"fd={fd_threshold:.0f} network={network_threshold:.0f}")
            logger.debug(f"PRESSURE_SCORES: cpu={cpu_pressure:.3f} memory={memory_pressure:.3f} "
                        f"virtual={virtual_pressure:.3f} http={http_load:.3f} latency={latency_pressure:.3f} "
                        f"fd={fd_pressure:.3f} network={network_pressure:.3f}")
        
        return np.clip(load_intensity, 0.0, 1.0)
    
    # Calculate load intensities
    curr_load_intensity = calculate_load_intensity(
        curr_cpu_rate, curr_resident_memory, curr_virtual_memory, 
        curr_http_requests, curr_http_duration, curr_open_fds, curr_response_size
    )
    
    next_load_intensity = calculate_load_intensity(
        next_cpu_rate, next_resident_memory, next_virtual_memory,
        next_http_requests, next_http_duration, next_open_fds, next_response_size
    )
    
    # === LOAD CLASSIFICATION ===
    def classify_load(load_intensity):
        """Classify load into categories for scaling decisions."""
        if load_intensity >= 0.7:
            return "HIGH_LOAD"
        elif load_intensity >= 0.4:
            return "MEDIUM_LOAD"
        elif load_intensity >= 0.1:
            return "LOW_LOAD"
        else:
            return "NO_LOAD"
    
    curr_load_class = classify_load(curr_load_intensity)
    next_load_class = classify_load(next_load_intensity)
    
    # === PERFORMANCE IMPROVEMENT SCORE (35%) ===
    # Reward load reduction after scaling up, penalize load increase after scaling down
    load_change = next_load_intensity - curr_load_intensity
    
    if action == "Scale Up":
        # Reward if load decreased or stayed manageable after scaling up
        if load_change <= 0:
            performance_score = 2.0 - abs(load_change)  # Reward load reduction
        else:
            performance_score = 1.0 - load_change  # Light penalty if load still increased
    elif action == "Scale Down":
        # Penalize if load increased significantly after scaling down
        if load_change <= 0.1:
            performance_score = 1.5  # Reward successful scale down
        else:
            performance_score = 1.0 - (load_change * 2.0)  # Penalty for load spike
    else:  # Keep Same
        # Reward stable load, penalize if load changed significantly
        if abs(load_change) <= 0.1:
            performance_score = 1.0  # Reward stability
        else:
            performance_score = 0.5 - abs(load_change)  # Penalty for not responding to load changes
    
    performance_score = np.clip(performance_score, -3.0, 3.0)
    
    # === SCALING APPROPRIATENESS SCORE (40%) ===
    # This is the key component - reward appropriate scaling decisions based on OUTCOME
    
    def calculate_scaling_appropriateness(curr_load_class, next_load_class, action, curr_replicas, next_replicas):
        """Calculate how appropriate the scaling decision was given the load and outcome."""
        
        # Define scaling thresholds
        min_replicas = 1
        max_replicas = 50
        
        # Base score based on initial load classification
        if curr_load_class == "HIGH_LOAD":
            if action == "Scale Up" and curr_replicas < max_replicas:
                base_score = 3.0  # Excellent: Scale up under high load
            elif action == "Keep Same" and curr_replicas >= 8:
                base_score = 1.0  # OK: Already have many replicas
            elif action == "Scale Down":
                base_score = -2.0  # Bad: Scale down under high load
            else:
                base_score = 0.0
                
        elif curr_load_class == "MEDIUM_LOAD":
            if action == "Keep Same":
                base_score = 2.0  # Excellent: Maintain stable state
            elif action == "Scale Up" and curr_replicas < 8:
                base_score = 1.0  # OK: Conservative scaling up
            elif action == "Scale Down" and curr_replicas > 3:
                base_score = 0.5  # OK: Conservative scaling down
            else:
                base_score = -0.5  # Suboptimal
                
        elif curr_load_class == "LOW_LOAD":
            if action == "Scale Down" and curr_replicas > min_replicas:
                base_score = 3.0  # Excellent: Scale down under low load
            elif action == "Keep Same" and curr_replicas <= 3:
                base_score = 1.5  # Good: Maintain minimal resources
            elif action == "Scale Up":
                base_score = -2.0  # Bad: Scale up under low load
            else:
                base_score = 0.0
                
        else:  # NO_LOAD
            if action == "Scale Down" and curr_replicas > min_replicas:
                base_score = 3.5  # Excellent: Scale down when no load
            elif action == "Keep Same" and curr_replicas == min_replicas:
                base_score = 2.0  # Good: Maintain minimum
            elif action == "Scale Up":
                base_score = -3.0  # Very bad: Scale up with no load
            else:
                base_score = -1.0
        
        # OUTCOME BONUS: Reward if the action actually improved the situation
        outcome_bonus = 0.0
        if action == "Scale Up":
            # Did scaling up actually help with load?
            if curr_load_class in ["HIGH_LOAD", "MEDIUM_LOAD"] and next_load_class in ["LOW_LOAD", "NO_LOAD"]:
                outcome_bonus = 1.0  # Excellent outcome
            elif curr_load_class == "HIGH_LOAD" and next_load_class == "MEDIUM_LOAD":
                outcome_bonus = 0.5  # Good outcome
        elif action == "Scale Down":
            # Did scaling down maintain performance?
            if curr_load_class in ["LOW_LOAD", "NO_LOAD"] and next_load_class in ["LOW_LOAD", "NO_LOAD"]:
                outcome_bonus = 1.0  # Excellent outcome - maintained low load
            elif curr_load_class == "LOW_LOAD" and next_load_class == "MEDIUM_LOAD":
                outcome_bonus = 0.0  # Neutral - slight increase is acceptable
            elif next_load_class == "HIGH_LOAD":
                outcome_bonus = -1.0  # Bad outcome - caused overload
        elif action == "Keep Same":
            # Did keeping same maintain stability?
            if curr_load_class == next_load_class:
                outcome_bonus = 0.5  # Good outcome - maintained stability
            elif abs(curr_load_intensity - next_load_intensity) <= 0.1:
                outcome_bonus = 0.3  # Acceptable outcome - minor changes
        
        return base_score + outcome_bonus
    
    appropriateness_score = calculate_scaling_appropriateness(curr_load_class, next_load_class, action, current_replicas, next_replicas)
    
    # === RESOURCE EFFICIENCY SCORE (15%) ===
    # Reward efficient per-replica resource utilization
    
    def calculate_per_replica_efficiency(load_intensity, replicas):
        """Calculate how efficiently each replica is being used."""
        if replicas <= 0:
            return 0.0
            
        # Ideal load per replica
        ideal_load_per_replica = 0.6  # 60% utilization is ideal
        actual_load_per_replica = load_intensity / replicas
        
        # Efficiency score based on how close to ideal
        efficiency = 1.0 - abs(actual_load_per_replica - ideal_load_per_replica)
        return np.clip(efficiency, 0.0, 1.0)
    
    curr_efficiency = calculate_per_replica_efficiency(curr_load_intensity, current_replicas)
    next_efficiency = calculate_per_replica_efficiency(next_load_intensity, next_replicas)
    
    resource_score = (next_efficiency - curr_efficiency) * 2.0  # Amplify efficiency improvements
    
    # === COST OPTIMIZATION SCORE (10%) ===
    # Reward cost reduction while maintaining performance
    
    def calculate_cost_efficiency(replicas, load_intensity):
        """Calculate cost efficiency - lower replicas with adequate performance."""
        if load_intensity <= 0.3:  # Low load
            if replicas <= 2:
                return 2.0  # Excellent cost efficiency
            elif replicas <= 5:
                return 1.0  # Good cost efficiency
            else:
                return -0.5  # Over-provisioned
        elif load_intensity <= 0.6:  # Medium load
            if replicas <= 5:
                return 1.0  # Good balance
            elif replicas <= 8:
                return 0.5  # Acceptable
            else:
                return -0.2  # Slightly over-provisioned
        else:  # High load
            if replicas >= 5:
                return 0.5  # Necessary resources
            else:
                return -1.0  # Under-provisioned
    
    cost_score = calculate_cost_efficiency(next_replicas, next_load_intensity)
    
    # === PROACTIVE SCALING BONUS (5%) ===
    # Reward early scaling before problems occur
    proactive_bonus = calculate_proactive_bonus(action, current_state, next_state)
    
    # === FINAL REWARD CALCULATION ===
    # Get weights from config or use defaults
    if config and hasattr(config, 'reward'):
        perf_weight = config.reward.performance_weight
        resource_weight = config.reward.resource_weight
        health_weight = config.reward.health_weight
        cost_weight = config.reward.cost_weight
    else:
        perf_weight = 0.35
        resource_weight = 0.15
        health_weight = 0.10
        cost_weight = 0.10
    
    # Calculate final reward
    total_reward = (
        performance_score * perf_weight +           # 35% - Load management
        appropriateness_score * 0.40 +             # 40% - Decision appropriateness  
        resource_score * resource_weight +          # 15% - Resource efficiency
        cost_score * cost_weight +                  # 10% - Cost optimization
        proactive_bonus * 0.05                      # 5% - Proactive scaling bonus
    )
    
    # Clip to reasonable range
    total_reward = np.clip(total_reward, -10.0, 10.0)
    
    # === DETAILED LOGGING ===
    logger.info(f"CONSUMER_REWARD_ANALYSIS: action={action} curr_replicas={current_replicas} next_replicas={next_replicas}")
    logger.info(f"LOAD_ANALYSIS: curr_load={curr_load_intensity:.3f}({curr_load_class}) next_load={next_load_intensity:.3f}({next_load_class})")
    logger.info(f"CONSUMER_METRICS: cpu_rate={curr_cpu_rate:.3f}=>{next_cpu_rate:.3f} "
                f"http_rate={curr_http_requests:.3f}=>{next_http_requests:.3f} "
                f"resident_memory={curr_resident_memory/1000000:.1f}MB=>{next_resident_memory/1000000:.1f}MB "
                f"virtual_memory={curr_virtual_memory/1000000:.1f}MB=>{next_virtual_memory/1000000:.1f}MB")
    logger.info(f"REWARD_BREAKDOWN: performance={performance_score:.2f}({perf_weight:.0%}) "
                f"appropriateness={appropriateness_score:.2f}(40%) "
                f"resource={resource_score:.2f}({resource_weight:.0%}) "
                f"cost={cost_score:.2f}({cost_weight:.0%}) "
                f"proactive={proactive_bonus:.2f}(5%) "
                f"total={total_reward:.2f}")
    
    # Log scaling decision evaluation
    if appropriateness_score >= 2.0:
        logger.info(f"SCALING_DECISION: EXCELLENT - {action} was highly appropriate for {curr_load_class}")
    elif appropriateness_score >= 1.0:
        logger.info(f"SCALING_DECISION: GOOD - {action} was appropriate for {curr_load_class}")
    elif appropriateness_score >= 0:
        logger.info(f"SCALING_DECISION: NEUTRAL - {action} was acceptable for {curr_load_class}")
    else:
        logger.warning(f"SCALING_DECISION: POOR - {action} was inappropriate for {curr_load_class}")
    
    return total_reward


def calculate_proactive_bonus(action, current_state, next_state):
    """Calculate proactive scaling bonus based on load trends."""
    
    # Extract current state metrics
    curr_cpu_rate = current_state.get('process_cpu_seconds_total_rate', 0.0)
    curr_resident_memory = current_state.get('process_resident_memory_bytes', 0.0)
    curr_virtual_memory = current_state.get('process_virtual_memory_bytes', 0.0)
    curr_http_requests = current_state.get('http_requests_total_rate', 0.0)
    curr_http_duration = current_state.get('http_request_duration_seconds_sum_rate', 0.0)
    curr_open_fds = current_state.get('process_open_fds', 0.0)
    curr_response_size = current_state.get('http_response_size_bytes_sum_rate', 0.0)
    
    # Extract next state metrics
    next_cpu_rate = next_state.get('process_cpu_seconds_total_rate', 0.0)
    next_resident_memory = next_state.get('process_resident_memory_bytes', 0.0)
    next_virtual_memory = next_state.get('process_virtual_memory_bytes', 0.0)
    next_http_requests = next_state.get('http_requests_total_rate', 0.0)
    next_http_duration = next_state.get('http_request_duration_seconds_sum_rate', 0.0)
    next_open_fds = next_state.get('process_open_fds', 0.0)
    next_response_size = next_state.get('http_response_size_bytes_sum_rate', 0.0)
    
    # Extract replica counts from state
    current_replicas = current_state.get('current_replicas', 1)
    next_replicas = next_state.get('current_replicas', current_replicas)  # FIXED: Extract from next_state
    
    # Calculate load trends (normalized per replica to account for scaling)
    cpu_trend = (next_cpu_rate / max(1, next_replicas)) - (curr_cpu_rate / max(1, current_replicas))
    memory_trend = (next_resident_memory / max(1, next_replicas)) - (curr_resident_memory / max(1, current_replicas))
    http_trend = (next_http_requests / max(1, next_replicas)) - (curr_http_requests / max(1, current_replicas))
    
    # Calculate overall load trend (weighted combination)
    load_trend = (
        cpu_trend * 0.4 +           # CPU trend weight
        memory_trend * 0.3 +        # Memory trend weight  
        http_trend * 0.3            # HTTP trend weight
    )
    
    # Calculate current load intensity for context
    curr_load_intensity = min(1.0, (
        min(1.0, curr_cpu_rate / 2.0) * 0.3 +
        min(1.0, (curr_resident_memory / 1000000) / 200.0) * 0.2 +  # Convert to MB and normalize
        min(1.0, curr_http_requests / 30.0) * 0.3 +
        min(1.0, curr_open_fds / 400.0) * 0.2
    ))
    
    # Proactive scaling logic
    if action == "Scale Up":
        # Reward scaling up when load is trending upward
        if load_trend > 0.1 and curr_load_intensity >= 0.3:
            return 1.5  # Strong proactive bonus
        elif load_trend > 0.05 and curr_load_intensity >= 0.2:
            return 1.0  # Good proactive bonus
        elif load_trend > 0.02 and curr_load_intensity >= 0.4:
            return 0.5  # Light proactive bonus
        elif curr_load_intensity >= 0.8:
            return 0.8  # Reactive scaling bonus (high load)
        else:
            return 0.0  # No bonus
    
    elif action == "Scale Down":
        # Reward scaling down when load is trending downward
        if load_trend < -0.1 and curr_load_intensity <= 0.4:
            return 1.5  # Strong proactive bonus
        elif load_trend < -0.05 and curr_load_intensity <= 0.5:
            return 1.0  # Good proactive bonus
        elif load_trend < -0.02 and curr_load_intensity <= 0.3:
            return 0.5  # Light proactive bonus
        elif curr_load_intensity <= 0.2:
            return 0.8  # Reactive scaling bonus (low load)
        else:
            return 0.0  # No bonus
    
    elif action == "Keep Same":
        # Reward stability when load is stable
        if abs(load_trend) <= 0.05 and 0.3 <= curr_load_intensity <= 0.6:
            return 0.5  # Stability bonus
        else:
            return 0.0  # No bonus
    
    return 0.0


def calculate_response_efficiency(http_bucket_count, response_size_sum):
    """Calculate response efficiency score (higher is better)."""
    # Normalize metrics and combine for efficiency score
    bucket_score = max(0, 5.0 - http_bucket_count * 0.002)  # Lower bucket count = better
    size_score = max(0, 3.0 - response_size_sum * 0.00001)  # Reasonable response size
    return (bucket_score + size_score) / 2.0


def calculate_cpu_efficiency(current_cpu, next_cpu, replicas):
    """Calculate CPU utilization efficiency."""
    cpu_usage_rate = max(0.1, next_cpu - current_cpu)  # CPU usage in time period
    cpu_per_replica = cpu_usage_rate / max(1, replicas)

    # Optimal CPU usage per replica (not too low, not too high)
    if 0.5 <= cpu_per_replica <= 2.0:  # Sweet spot
        return 2.0
    elif cpu_per_replica < 0.5:  # Under-utilized
        return 1.0 - (0.5 - cpu_per_replica)
    else:  # Over-utilized
        return max(0, 2.0 - (cpu_per_replica - 2.0) * 0.5)


def calculate_health_score(current_scrape, next_scrape, memory_mb, http_buckets):
    """Calculate overall system health score."""
    # Scrape sample health (monitoring effectiveness)
    scrape_health = min(3.0, next_scrape / 100.0)  # 300+ samples = full score

    # System stress indicators
    memory_stress = max(0, 2.0 - max(0, memory_mb - 500) * 0.002)  # Penalty for >500MB
    traffic_stress = max(0, 2.0 - max(0, http_buckets - 100) * 0.001)  # Penalty for high traffic

    return (scrape_health + memory_stress + traffic_stress) / 3.0


def calculate_action_penalty(action, current_replicas, current_state=None, next_state=None):
    """
    BALANCED: Much reduced penalties, focus on rewarding good decisions.
    Addresses: Reduced harsh penalties + appropriate scale-down rewards + load context.
    FIXED: Uses actual selected features (gauges), not cumulative counters.
    """
    penalty = 0.0

    # FIXED: Get load context using actual available metrics from our feature set
    if current_state:
        unavailable_replicas = current_state.get('kube_deployment_status_replicas_unavailable', 0)
        pod_readiness = current_state.get('kube_pod_container_status_ready', 1.0)
        running_containers = current_state.get('kube_pod_container_status_running', 0)
        cpu_limits = current_state.get('kube_pod_container_resource_limits_cpu', 0)
        memory_limits = current_state.get('kube_pod_container_resource_limits_memory', 0)

        memory_mb = memory_limits / 1000000 if memory_limits > 0 else 0

        # Define load context using actual Kubernetes state metrics
        is_no_load = (
                unavailable_replicas == 0 and  # No failing pods
                pod_readiness >= 0.95 and  # High readiness
                cpu_limits <= 1.0 and  # Low CPU allocation
                memory_mb <= 300  # Low memory allocation
        )

        is_low_load = (
                unavailable_replicas == 0 and  # No failing pods
                pod_readiness >= 0.90 and  # Good readiness
                cpu_limits <= 2.0 and  # Moderate CPU
                memory_mb <= 600  # Moderate memory
        )
    else:
        is_no_load = False
        is_low_load = False

    # Scale Down penalties/rewards
    if action == "Scale Down":
        if current_replicas <= 1:
            penalty = -1.0  # REDUCED: was -2.0, still prevent scaling below 1
        elif current_state is not None:
            is_underutilized = check_system_underutilized(current_state, next_state, current_replicas)

            if is_underutilized or is_no_load or is_low_load:
                # REWARD appropriate scale-down - this was the key missing piece!
                if is_no_load:
                    penalty = +2.5  # Strong reward for scaling down with no load
                elif is_low_load:
                    penalty = +2.0  # Good reward for scaling down with low load
                elif current_replicas >= 8:
                    penalty = +1.5  # Reward for reducing over-provisioning
                elif current_replicas >= 5:
                    penalty = +1.0  # Reward for right-sizing
                else:
                    penalty = +0.5  # Light reward for any appropriate scale-down
            else:
                # System is NOT underutilized - light penalty
                penalty = -0.5  # REDUCED: was -1.0, less harsh

    # Scale Up penalties
    elif action == "Scale Up":
        if current_state is not None:
            is_overloaded = check_system_overloaded(current_state, next_state, current_replicas)

            if is_overloaded and not is_no_load:
                # System needs resources and has actual load - minimal penalty
                if current_replicas >= 50:
                    penalty = -1.0  # REDUCED: was -2.0
                else:
                    penalty = 0.0  # Neutral - system needs resources
            else:
                # Unnecessary scaling up - moderate penalties
                if is_no_load:
                    penalty = -1.5  # REDUCED: was much higher, but still discourage waste
                elif current_replicas >= 10:
                    penalty = -1.0  # REDUCED: was -3.0
                elif current_replicas >= 5:
                    penalty = -0.5  # REDUCED: was -1.0
                else:
                    penalty = -0.2  # REDUCED: was -0.5

    # Keep Same - very light penalty for any action (encourage slight bias toward stability)
    elif action != "Keep Same":
        penalty = -0.1  # REDUCED: was -0.2, minimal stability bias

    return penalty


def check_system_underutilized(current_state, next_state, current_replicas):
    """
    FIXED: Use BOTH current and next state to evaluate if scaling action was appropriate.
    Check if system remained stable after action and has low utilization.
    """
    if not current_state or not next_state:
        return False

    # Use NEXT state (after scaling action) to evaluate final system state
    cpu_limits_total = next_state.get('kube_pod_container_resource_limits_cpu', 0)  # total cores
    memory_bytes_total = next_state.get('kube_pod_container_resource_limits_memory', 0)
    memory_mb_total = memory_bytes_total / 1000000 if memory_bytes_total > 0 else 0

    # Get actual replica count from next state
    next_replicas = next_state.get('current_replicas', current_replicas)

    # Per-replica resource usage (this is what matters for scaling decisions)
    cpu_per_replica = cpu_limits_total / max(1, next_replicas)
    memory_per_replica = memory_mb_total / max(1, next_replicas)

    # Kubernetes health indicators from NEXT state (after action)
    unavailable_replicas = next_state.get('kube_deployment_status_replicas_unavailable', 0)
    pod_readiness = next_state.get('kube_pod_container_status_ready', 1.0)
    running_containers = next_state.get('kube_pod_container_status_running', 0)

    # Check if system maintained stability after scaling action
    underutilized_conditions = [
        cpu_per_replica < 1.0,  # < 1 CPU core per replica = low utilization
        memory_per_replica < 300,  # < 300MB per replica = low memory usage
        unavailable_replicas == 0,  # No failing pods (absolute requirement)
        pod_readiness >= 0.95,  # 95% readiness = healthy
        running_containers > 0,  # At least some containers running
        next_replicas > 1  # Don't scale below 1 replica (minimum viable)
    ]

    is_underutilized = all(underutilized_conditions)

    # Enhanced logging to show the fix
    if next_replicas >= 2:
        logger.info(f"UNDERUTILIZATION_CHECK: {sum(underutilized_conditions)}/6 conditions_met "
                    f"cpu_per_replica={cpu_per_replica:.1f}cores memory_per_replica={memory_per_replica:.0f}MB "
                    f"(total: cpu={cpu_limits_total:.1f} memory={memory_mb_total:.0f}MB across {next_replicas} replicas) "
                    f"unavailable={unavailable_replicas} readiness={pod_readiness:.2f} "
                    f"running={running_containers} result={'SAFE_TO_SCALE_DOWN' if is_underutilized else 'NOT_SAFE'}")

    return is_underutilized


def check_system_overloaded(current_state, next_state, current_replicas):
    """
    FIXED: Use BOTH current and next state to evaluate if scaling action resolved overload.
    Check if system improved after scaling action.
    """
    if not current_state or not next_state:
        return False

    # Compare BEFORE (current) and AFTER (next) to see if scaling helped
    curr_cpu_limits = current_state.get('kube_pod_container_resource_limits_cpu', 0)
    curr_memory_bytes = current_state.get('kube_pod_container_resource_limits_memory', 0)
    curr_memory_mb = curr_memory_bytes / 1000000 if curr_memory_bytes > 0 else 0
    curr_unavailable = current_state.get('kube_deployment_status_replicas_unavailable', 0)
    curr_readiness = current_state.get('kube_pod_container_status_ready', 1.0)

    # Use NEXT state (after scaling action) to evaluate final system state
    next_cpu_limits = next_state.get('kube_pod_container_resource_limits_cpu', 0)
    next_memory_bytes = next_state.get('kube_pod_container_resource_limits_memory', 0)
    next_memory_mb = next_memory_bytes / 1000000 if next_memory_bytes > 0 else 0
    next_unavailable = next_state.get('kube_deployment_status_replicas_unavailable', 0)
    next_readiness = next_state.get('kube_pod_container_status_ready', 1.0)

    # Check if system is STILL overloaded after scaling action
    critical_conditions = [
        next_unavailable > 0,  # Pods still failing = action didn't help
        next_readiness < 0.8,  # Still low readiness = still stressed
    ]

    # High utilization conditions (after scaling)
    high_utilization_conditions = [
        next_cpu_limits > 3.0,  # > 3 CPU cores = high load
        next_memory_mb > 800,  # > 800MB = memory pressure
    ]

    # System is overloaded if critical issues persist OR high utilization continues
    is_overloaded = any(critical_conditions) or sum(high_utilization_conditions) >= 2

    # Get actual replica count from next state
    next_replicas = next_state.get('current_replicas', current_replicas)

    if next_replicas <= 50:
        logger.info(
            f"OVERLOAD_CHECK: critical={sum(critical_conditions)}/2 utilization={sum(high_utilization_conditions)}/2 "
            f"cpu_limits={curr_cpu_limits:.1f}=>{next_cpu_limits:.1f}cores memory={curr_memory_mb:.0f}=>{next_memory_mb:.0f}MB "
            f"unavailable={curr_unavailable}=>{next_unavailable} readiness={curr_readiness:.2f}=>{next_readiness:.2f} "
            f"result={'STILL_NEEDS_RESOURCES' if is_overloaded else 'RESOLVED_OR_ADEQUATE'}")

    return is_overloaded

# === REWARD FUNCTION EQUATIONS AND WEIGHTS ===

REWARD_EQUATIONS = """
=== DQN REWARD FUNCTION EQUATIONS ===

## FINAL REWARD CALCULATION:
total_reward = (
    performance_score * 35% +           # Load management effectiveness
    appropriateness_score * 40% +       # Decision appropriateness for load state  
    resource_score * 15% +              # Resource efficiency improvement
    cost_score * 10% +                  # Cost optimization
    proactive_bonus * 5%                # Proactive scaling bonus
)

## COMPONENT EQUATIONS:

### 1. LOAD INTENSITY CALCULATION:
load_intensity = (
    cpu_pressure * 25% +                # CPU usage normalized to 2.0 as high
    gc_pressure * 20% +                 # Memory pressure (GC activity)
    http_load * 25% +                   # Request volume normalized to 50 req/s
    latency_pressure * 15% +            # Response time normalized to 2s
    fd_pressure * 10% +                 # File descriptors normalized to 500
    network_pressure * 5%               # Network throughput normalized to 1MB/s
)

### 2. PERFORMANCE SCORE:
if action == "Scale Up":
    if load_decreased:
        performance_score = 2.0 - abs(load_change)      # Reward load reduction
    else:
        performance_score = 1.0 - load_change           # Light penalty if load increased
        
elif action == "Scale Down":
    if load_increase <= 0.1:
        performance_score = 1.5                         # Reward successful scale down
    else:
        performance_score = 1.0 - (load_change * 2.0)   # Penalty for load spike
        
else: # Keep Same
    if abs(load_change) <= 0.1:
        performance_score = 1.0                         # Reward stability
    else:
        performance_score = 0.5 - abs(load_change)      # Penalty for not responding

### 3. APPROPRIATENESS SCORE:
HIGH_LOAD (≥0.7):
    Scale Up: +3.0 (excellent)
    Keep Same: +1.0 (if replicas ≥8)
    Scale Down: -2.0 (bad)

MEDIUM_LOAD (0.4-0.7):
    Keep Same: +2.0 (excellent)
    Scale Up: +1.0 (if replicas <8)
    Scale Down: +0.5 (if replicas >3)

LOW_LOAD (0.1-0.4):
    Scale Down: +3.0 (excellent)
    Keep Same: +1.5 (if replicas ≤3)
    Scale Up: -2.0 (bad)

NO_LOAD (<0.1):
    Scale Down: +3.5 (excellent)
    Keep Same: +2.0 (if replicas =1)
    Scale Up: -3.0 (very bad)

### 4. RESOURCE EFFICIENCY:
ideal_load_per_replica = 0.6  # 60% utilization target
efficiency = 1.0 - abs(actual_load_per_replica - ideal_load_per_replica)
resource_score = (next_efficiency - curr_efficiency) * 2.0

### 5. COST EFFICIENCY:
LOW_LOAD (≤0.3):
    ≤2 replicas: +2.0 (excellent)
    ≤5 replicas: +1.0 (good)
    >5 replicas: -0.5 (over-provisioned)

MEDIUM_LOAD (0.3-0.6):
    ≤5 replicas: +1.0 (good balance)
    ≤8 replicas: +0.5 (acceptable)
    >8 replicas: -0.2 (over-provisioned)

HIGH_LOAD (>0.6):
    ≥5 replicas: +0.5 (necessary)
    <5 replicas: -1.0 (under-provisioned)

### 6. PROACTIVE BONUS:
load_trend = weighted combination of metric trends
if Scale Up and trending_up and current_load ≥0.3: +1.5
if Scale Down and trending_down and current_load ≤0.4: +1.5
if Keep Same and stable_load and 0.3 ≤ load ≤ 0.6: +0.5

## LOAD CLASSIFICATION THRESHOLDS:
- HIGH_LOAD: ≥0.7 load intensity
- MEDIUM_LOAD: 0.4-0.7 load intensity  
- LOW_LOAD: 0.1-0.4 load intensity
- NO_LOAD: <0.1 load intensity

## REWARD RANGE: [-10.0, +10.0] (clipped)
"""

REWARD_WEIGHTS = {
    "performance_weight": 0.35,      # 35% - Load management effectiveness
    "appropriateness_weight": 0.40,  # 40% - Decision appropriateness for load state
    "resource_weight": 0.15,         # 15% - Resource efficiency improvement  
    "cost_weight": 0.10,             # 10% - Cost optimization
    "proactive_weight": 0.05         # 5% - Proactive scaling bonus
} 