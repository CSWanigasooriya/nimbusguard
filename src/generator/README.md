# NimbusGuard Load Generator

A comprehensive load testing tool designed to test KEDA autoscaling behavior by generating realistic CPU and memory load against the NimbusGuard consumer service.

## Features

- 🚀 **Multiple Load Patterns**: 7 predefined scenarios from light to heavy load
- ⚡ **Async HTTP Requests**: High-performance concurrent request handling
- 📊 **Real-time Monitoring**: Track success rates, response times, and RPS
- 🧹 **Service Management**: Health checks and memory cleanup
- 🎯 **KEDA Testing**: Designed to trigger CPU and memory-based scaling

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# List available load tests
python load_generator.py --list

# Run medium load test (default)
python load_generator.py --test medium

# Run all tests with monitoring
python load_generator.py --test all --monitor --cleanup
```

### With Docker
```bash
# Build the generator image
docker build -t nimbusguard-generator .

# Run medium load test
docker run --rm --network host nimbusguard-generator --test medium

# Run heavy load with cleanup
docker run --rm --network host nimbusguard-generator --test heavy --cleanup
```

## Available Load Tests

| Test | Description | Concurrent | Total Requests | CPU | Memory | Duration |
|------|-------------|------------|----------------|-----|--------|----------|
| `light` | Baseline load | 2 | 10 | 2 | 20MB | 5s |
| `medium` | Moderate load | 5 | 20 | 6 | 80MB | 15s |
| `heavy` | High load | 10 | 30 | 9 | 150MB | 20s |
| `sustained` | Long-running background | 8 | 25 | 8 | 120MB | 60s |
| `burst` | Quick intense burst | 15 | 40 | 10 | 100MB | 10s |
| `memory_stress` | Memory-focused | 6 | 15 | 3 | 250MB | 30s |
| `cpu_stress` | CPU-focused | 8 | 20 | 10 | 50MB | 25s |

## Command Line Options

```bash
python load_generator.py [OPTIONS]

Options:
  --url TEXT              Base URL of consumer service [default: http://localhost:8000]
  --test {light,medium,heavy,sustained,burst,memory_stress,cpu_stress,all}
                         Load test to run [default: medium]
  --cleanup              Cleanup service memory before starting
  --monitor              Monitor service status during tests
  --list                 List available load tests
```

## Testing KEDA Autoscaling

### 1. Deploy Your Application with KEDA
```bash
cd /path/to/nimbusguard
make keda-install
make dev
make forward
```

### 2. Monitor Scaling in Another Terminal
```bash
# Watch pods scale
kubectl get pods -n nimbusguard -w

# Watch KEDA ScaledObject
kubectl get scaledobjects -n nimbusguard -w

# Monitor resource usage
kubectl top pods -n nimbusguard
```

### 3. Generate Load to Trigger Scaling

**Quick Test (should trigger scaling):**
```bash
python load_generator.py --test heavy --monitor
```

**Sustained Load (maintains scaling):**
```bash
python load_generator.py --test sustained --monitor
```

**Complete Test Suite:**
```bash
python load_generator.py --test all --cleanup --monitor
```

### 4. Expected Scaling Behavior

- **CPU Threshold**: 70% - heavy/cpu_stress tests should trigger this
- **Memory Threshold**: 80% - memory_stress/sustained tests should trigger this
- **Scale Up**: Should see pods increase from 1 to 2-15 based on load
- **Scale Down**: After load ends, pods should gradually scale back down

## Load Test Output

```
2024-01-15 10:30:15,123 - INFO - ✅ Service is healthy: {'status': 'healthy'}
2024-01-15 10:30:15,124 - INFO - 🚀 Starting load test: Heavy Load
2024-01-15 10:30:15,124 - INFO - 📝 Description: High load that should definitely trigger KEDA scaling
2024-01-15 10:30:15,124 - INFO - ⚙️  Config: 10 concurrent, 30 total requests
2024-01-15 10:30:15,125 - INFO - ⏳ Sending 30 requests...
2024-01-15 10:30:35,456 - INFO - ✅ Load test completed in 20.33s
2024-01-15 10:30:35,456 - INFO - 📊 Success rate: 100.0% (30/30)
2024-01-15 10:30:35,456 - INFO - ⚡ RPS: 1.48
2024-01-15 10:30:35,456 - INFO - ⏱️  Avg response time: 20.15s
```

## Integration with Monitoring

The load generator works seamlessly with your existing monitoring stack:

- **Prometheus**: HTTP metrics captured automatically
- **Grafana**: View load patterns and scaling behavior
- **Tempo**: Trace individual requests through the system
- **Loki**: Aggregate logs from load generation

## Troubleshooting

**Service not responding:**
```bash
# Check if consumer is running
kubectl get pods -n nimbusguard

# Check service health
curl http://localhost:8000/health
```

**KEDA not scaling:**
```bash
# Check KEDA operator logs
kubectl logs -n keda -l app.kubernetes.io/name=keda-operator

# Check ScaledObject status
kubectl describe scaledobject consumer-scaledobject -n nimbusguard
```

**High memory usage:**
```bash
# Cleanup service memory
python load_generator.py --cleanup
# Or via API
curl -X DELETE http://localhost:8000/process/cleanup
``` 