# KEDA Scaling Fix and Troubleshooting

This directory contains scripts to fix and troubleshoot KEDA scaling issues.

## Issues Fixed

1. **Kafka Bootstrap Servers**: Fixed KEDA scaler to use only the single Kafka instance (`kafka-0`)
2. **Added TriggerAuthentication**: Added proper Kafka authentication for KEDA
3. **Improved Monitoring**: Enhanced monitoring script with better error handling
4. **Setup Automation**: Added script to automatically install and configure KEDA

## Quick Fix

To quickly fix the KEDA scaling issue:

```bash
# 1. Apply the fixed KEDA configuration
kubectl apply -f kubernetes-manifests/base/keda-scaler.yaml

# 2. Run the setup script to ensure everything is configured
python3 scripts/fix_keda_setup.py

# 3. Test the scaling
python3 scripts/test_keda_scaling.py

# 4. Monitor the scaling
python3 scripts/monitor_keda.py
```

## Scripts

### `fix_keda_setup.py`
Automatically installs KEDA (if needed) and configures all components:
- Checks KEDA installation
- Installs KEDA if missing
- Verifies Kafka connectivity
- Creates required Kafka topics
- Applies ScaledObject configuration

### `monitor_keda.py` (Enhanced)
Improved monitoring script that shows:
- Current pod count
- Kafka consumer lag
- KEDA operator status
- ScaledObject status
- Recent scaling events

### `test_keda_scaling.py`
Produces test messages to Kafka to trigger scaling:
- Sends 50 test messages to `scaling-events` topic
- Monitors consumer group lag
- Watches pod scaling for 2 minutes

## Configuration Changes

### KEDA ScaledObject (`keda-scaler.yaml`)
- **Fixed**: Use single Kafka instance (`kafka-0` only)
- **Added**: TriggerAuthentication for proper Kafka connection
- **Improved**: Better polling intervals and thresholds

## Troubleshooting

If scaling still doesn't work:

1. **Check KEDA operator logs**:
   ```bash
   kubectl logs -n keda-system -l app=keda-operator
   ```

2. **Check ScaledObject status**:
   ```bash
   kubectl describe scaledobject consumer-workload-scaler -n nimbusguard
   ```

3. **Check HPA status**:
   ```bash
   kubectl get hpa -n nimbusguard
   kubectl describe hpa -n nimbusguard
   ```

4. **Verify Kafka connectivity**:
   ```bash
   kubectl exec -n nimbusguard kafka-0 -- kafka-topics.sh --bootstrap-server localhost:9092 --list
   ```

5. **Check consumer group**:
   ```bash
   kubectl exec -n nimbusguard kafka-0 -- kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group background-consumer
   ```

## Expected Behavior

After applying the fixes:
1. KEDA should successfully connect to Kafka
2. Consumer lag should be properly monitored
3. Pods should scale up when lag increases
4. Pods should scale down when lag decreases

The external metrics API errors should be resolved, and you should see successful scaling events instead of the error messages.
