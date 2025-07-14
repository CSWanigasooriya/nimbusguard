# Intelligent Consumer Pod Feature Selection Results

## Methodology
- **Auto-discovered**: 31 consumer pod metrics
- **Auto-analyzed**: 41 metric characteristics  
- **Auto-calculated**: Rate metrics from counters
- **Auto-ensured**: Category diversity

## Dataset Summary
- **Samples**: 45
- **Features**: 9
- **Scaling Actions**: {'scale_up': 44, 'keep_same': 1}

## Selected Features (Top 9)
1. **`http_request_duration_seconds_sum_rate`** (category: network, type: gauge, relevance: 100.0)
2. **`http_request_duration_seconds_count_rate`** (category: network, type: gauge, relevance: 95.0)
3. **`process_cpu_seconds_total_rate`** (category: cpu, type: gauge, relevance: 115.0)
4. **`process_resident_memory_bytes`** (category: memory, type: gauge, relevance: 120.0)
5. **`http_requests_total_rate`** (category: network, type: gauge, relevance: 105.0)
6. **`http_response_size_bytes_sum_rate`** (category: network, type: gauge, relevance: 90.0)
7. **`process_open_fds`** (category: io, type: gauge, relevance: 90.0)
8. **`http_requests_total`** (category: network, type: counter, relevance: 85.0)
9. **`process_virtual_memory_bytes`** (category: memory, type: gauge, relevance: 115.0)

## Category Distribution
- **NETWORK**: 5 features
- **CPU**: 1 features
- **MEMORY**: 2 features
- **IO**: 1 features
