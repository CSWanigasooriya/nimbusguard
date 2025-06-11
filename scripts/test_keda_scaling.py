#!/usr/bin/env python3
"""
Test Kafka message production to trigger KEDA scaling
"""

import subprocess
import time
import json

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def get_kafka_pod():
    """Get the Kafka pod name"""
    stdout, stderr, returncode = run_command("kubectl get pods -n nimbusguard -l app=kafka --no-headers -o custom-columns=':metadata.name'")
    if returncode == 0 and stdout.strip():
        return stdout.strip().split('\n')[0]
    return None

def produce_test_messages(count=50):
    """Produce test messages to trigger scaling"""
    kafka_pod = get_kafka_pod()
    if not kafka_pod:
        print("❌ Kafka pod not found")
        return False
    
    print(f"📤 Producing {count} test messages to scaling-events topic...")
    
    # Create a temporary file with test messages
    test_messages = []
    for i in range(count):
        message = {
            "event_id": f"test-{i}",
            "timestamp": int(time.time() * 1000),
            "source": "test-script",
            "message": f"Test scaling event {i}",
            "processing_time_ms": 100 + (i % 500)  # Simulate variable processing times
        }
        test_messages.append(json.dumps(message))
    
    # Send messages in batches
    batch_size = 10
    for i in range(0, len(test_messages), batch_size):
        batch = test_messages[i:i+batch_size]
        messages_str = "\n".join(batch)
        
        # Use kubectl exec to produce messages
        cmd = f'kubectl exec -n nimbusguard {kafka_pod} -- bash -c \'echo "{messages_str}" | kafka-console-producer.sh --bootstrap-server localhost:9092 --topic scaling-events\''
        
        stdout, stderr, returncode = run_command(cmd)
        if returncode != 0:
            print(f"❌ Failed to send batch {i//batch_size + 1}: {stderr}")
            return False
        
        print(f"✅ Sent batch {i//batch_size + 1}/{(len(test_messages) + batch_size - 1)//batch_size}")
        time.sleep(1)  # Small delay between batches
    
    print(f"✅ Successfully produced {count} test messages")
    return True

def check_consumer_group():
    """Check consumer group lag"""
    kafka_pod = get_kafka_pod()
    if not kafka_pod:
        return
    
    print("🔍 Checking consumer group lag...")
    cmd = f"kubectl exec -n nimbusguard {kafka_pod} -- kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group background-consumer"
    stdout, stderr, returncode = run_command(cmd)
    
    if returncode == 0:
        print("📊 Consumer Group Status:")
        print(stdout)
    else:
        print(f"❌ Failed to get consumer group info: {stderr}")

def monitor_scaling():
    """Monitor pod scaling for a short period"""
    print("👀 Monitoring pod count for 2 minutes...")
    
    for i in range(24):  # 2 minutes with 5-second intervals
        stdout, stderr, returncode = run_command("kubectl get pods -n nimbusguard -l app=consumer-workload --no-headers | wc -l")
        if returncode == 0:
            pod_count = stdout.strip()
            print(f"⏰ {i*5:3d}s: {pod_count} pods")
        else:
            print(f"⏰ {i*5:3d}s: Error getting pod count")
        
        time.sleep(5)

def main():
    print("🧪 KEDA Scaling Test Script")
    print("=" * 30)
    
    # Check initial state
    print("🔍 Checking initial state...")
    check_consumer_group()
    
    # Produce test messages
    if not produce_test_messages(50):
        print("❌ Failed to produce test messages")
        return
    
    print("\n⏳ Waiting 30 seconds for KEDA to detect lag...")
    time.sleep(30)
    
    # Check consumer group again
    check_consumer_group()
    
    # Monitor scaling
    monitor_scaling()
    
    print("\n🎉 Test complete!")
    print("💡 If scaling didn't occur, check:")
    print("  1. KEDA operator logs: kubectl logs -n keda-system -l app=keda-operator")
    print("  2. ScaledObject status: kubectl describe scaledobject -n nimbusguard")
    print("  3. HPA status: kubectl get hpa -n nimbusguard")

if __name__ == "__main__":
    main()
