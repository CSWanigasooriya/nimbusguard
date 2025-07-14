# Simplified Consumer Pod Feature Selection Results

## Methodology
- **Focus**: CPU and Memory metrics only
- **Auto-discovered**: 2 relevant consumer pod metrics
- **Auto-analyzed**: 3 metric characteristics  
- **Auto-calculated**: CPU rate from counter
- **Resource Targets**: 0.5 CPU cores, 1024MB memory

## Dataset Summary
- **Samples**: 45
- **Features**: 2
- **Scaling Actions**: {'scale_down': 40, 'keep_same': 3, 'scale_up': 2}

## Selected Features
1. **`process_cpu_seconds_total_rate`** (category: cpu, type: gauge, relevance: 100.0)
2. **`process_resident_memory_bytes`** (category: memory, type: gauge, relevance: 100.0)

## Resource Configuration
- **CPU Limit**: 0.5 cores (500m)
- **Memory Limit**: 1024MB (1Gi)
- **CPU Target**: 70.0% utilization
- **Memory Target**: 80.0% utilization

## Scaling Logic
- **Scale Up**: When pressure score > 120% of target capacity
- **Scale Down**: When pressure score < 50% of target capacity
- **Keep Same**: When pressure score is between 50-120% of target
