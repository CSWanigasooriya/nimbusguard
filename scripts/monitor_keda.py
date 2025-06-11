#!/usr/bin/env python3
import subprocess
import time
from datetime import datetime
import json
import os

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return None, "Command timed out", 1
    except Exception as e:
        return None, f"Error running command: {e}", 1

def get_pod_count():
    cmd = "kubectl get pods -n nimbusguard -l app=consumer-workload --no-headers | wc -l"
    stdout, stderr, returncode = run_command(cmd)
    if returncode == 0:
        return stdout.strip()
    return f"Error: {stderr}"

def get_kafka_lag():
    """Get Kafka consumer lag using kubectl exec into Kafka pod"""
    # First, try to get the Kafka pod name
    get_pod_cmd = "kubectl get pods -n nimbusguard -l app=kafka --no-headers -o custom-columns=':metadata.name'"
    stdout, stderr, returncode = run_command(get_pod_cmd)
    
    if returncode != 0 or not stdout.strip():
        return "Kafka pod not found"
    
    kafka_pod = stdout.strip().split('\n')[0]
    
    # Try to get consumer group lag
    lag_cmd = f"kubectl exec -n nimbusguard {kafka_pod} -- kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group background-consumer"
    stdout, stderr, returncode = run_command(lag_cmd)
    
    if returncode == 0 and stdout:
        lines = stdout.strip().split('\n')
        # Look for lag information in the output
        total_lag = 0
        found_topic = False
        for line in lines:
            if 'scaling-events' in line and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        lag = int(parts[5])  # LAG column
                        total_lag += lag
                        found_topic = True
                    except (IndexError, ValueError):
                        continue
        
        if found_topic:
            return f"Topic: scaling-events, Lag: {total_lag}"
        return "No lag data found"
    
    # Fallback: try to get ScaledObject status
    scaled_obj_cmd = "kubectl get scaledobject consumer-workload-scaler -n nimbusguard -o jsonpath='{.status}'"
    stdout, stderr, returncode = run_command(scaled_obj_cmd)
    
    if returncode == 0 and stdout:
        try:
            status = json.loads(stdout)
            if 'externalMetricNames' in status:
                return f"ScaledObject active: {status.get('conditions', [{}])[-1].get('status', 'Unknown') if status.get('conditions') else 'Unknown'}"
        except json.JSONDecodeError:
            pass
    
    return "Unable to determine lag"

def get_scaling_events():
    cmd = "kubectl get events -n nimbusguard --sort-by='.lastTimestamp' | grep -E '(scaled|scaling|ScaledObject|HPA|KEDA)' | tail -n 5"
    stdout, stderr, returncode = run_command(cmd)
    if returncode == 0:
        return stdout if stdout else "No recent scaling events"
    return f"Error getting events: {stderr}"

def get_keda_status():
    """Check KEDA operator status in multiple namespaces with detailed diagnostics"""
    # First check keda namespace (our setup)
    cmd = "kubectl get pods -n keda -l app=keda-operator --no-headers"
    stdout, stderr, returncode = run_command(cmd)
    if returncode == 0 and stdout:
        lines = stdout.strip().split('\n')
        running_pods = [line for line in lines if 'Running' in line]
        if running_pods:
            # Get pod details
            pod_name = running_pods[0].split()[0]
            cmd_details = f"kubectl get pod {pod_name} -n keda -o jsonpath='{{.status.conditions}}'"
            stdout_details, stderr_details, _ = run_command(cmd_details)
            try:
                conditions = json.loads(stdout_details)
                for condition in conditions:
                    if condition['type'] == 'Ready' and condition['status'] != 'True':
                        return f"⚠️ KEDA operator pod not ready: {condition.get('message', 'No message')}"
            except json.JSONDecodeError:
                pass
            return f"✅ KEDA operator running ({len(running_pods)} pods in 'keda' namespace)"
    
    # Check metrics server
    cmd_metrics = "kubectl get pods -n keda -l app=keda-operator-metrics-apiserver --no-headers"
    stdout_metrics, stderr_metrics, _ = run_command(cmd_metrics)
    if stdout_metrics:
        lines = stdout_metrics.strip().split('\n')
        running_pods = [line for line in lines if 'Running' in line]
        if not running_pods:
            return "⚠️ KEDA metrics server not running"
    
    return "❌ KEDA operator not found"

def get_scaledobject_status():
    """Get ScaledObject status with detailed diagnostics"""
    cmd = "kubectl get scaledobject -n nimbusguard -o jsonpath='{.items[0].status}'"
    stdout, stderr, returncode = run_command(cmd)
    if returncode == 0 and stdout:
        try:
            status = json.loads(stdout)
            conditions = status.get('conditions', [])
            if conditions:
                latest_condition = conditions[-1]
                if latest_condition['type'] == 'Ready' and latest_condition['status'] != 'True':
                    return f"⚠️ ScaledObject not ready: {latest_condition.get('message', 'No message')}"
            return f"✅ ScaledObject ready: {status.get('externalMetricNames', ['No metrics'])[0]}"
        except json.JSONDecodeError:
            pass
    
    cmd_basic = "kubectl get scaledobject -n nimbusguard -o wide --no-headers"
    stdout_basic, stderr_basic, returncode = run_command(cmd_basic)
    if returncode == 0:
        return stdout_basic if stdout_basic else "No ScaledObjects found"
    return f"Error: {stderr_basic}"

def get_external_metrics_api_status():
    """Check if external metrics API is available with detailed diagnostics"""
    # Check API service status
    cmd = "kubectl get apiservice v1beta1.external.metrics.k8s.io -o jsonpath='{.status.conditions[?(@.type==\"Available\")].status}'"
    stdout, stderr, returncode = run_command(cmd)
    
    if returncode == 0:
        if stdout.strip() == "True":
            return "✅ External Metrics API Available"
        else:
            # Get detailed status
            cmd_details = "kubectl get apiservice v1beta1.external.metrics.k8s.io -o jsonpath='{.status.conditions}'"
            stdout_details, stderr_details, _ = run_command(cmd_details)
            try:
                conditions = json.loads(stdout_details)
                for condition in conditions:
                    if condition['type'] == 'Available':
                        return f"❌ External Metrics API Unavailable: {condition.get('message', 'No message')}"
            except json.JSONDecodeError:
                pass
            
            # Check KEDA metrics server pod
            cmd_pod = "kubectl get pods -n keda -l app=keda-operator-metrics-apiserver -o jsonpath='{.items[0].status.conditions}'"
            stdout_pod, stderr_pod, _ = run_command(cmd_pod)
            try:
                pod_conditions = json.loads(stdout_pod)
                for condition in pod_conditions:
                    if condition['type'] == 'Ready' and condition['status'] != 'True':
                        return f"❌ Metrics Server Pod Not Ready: {condition.get('message', 'No message')}"
            except json.JSONDecodeError:
                pass
            
            return "❌ External Metrics API Unavailable (check KEDA metrics server)"
    return "❓ External Metrics API Status Unknown"

def get_hpa_status():
    """Get HPA status"""
    cmd = "kubectl get hpa -n nimbusguard --no-headers"
    stdout, stderr, returncode = run_command(cmd)
    if returncode == 0:
        if stdout.strip():
            return f"HPA found: {len(stdout.strip().split('\n'))} objects"
        else:
            return "No HPA objects found"
    return f"Error getting HPA: {stderr}"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("KEDA Scaling Monitor - Enhanced Version")
    print("======================================")
    print("Press Ctrl+C to exit")
    print()

    try:
        while True:
            clear_screen()
            print("KEDA Scaling Monitor - Enhanced Version")
            print("======================================")
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"⏰ Time: {current_time}")
            print()
            
            # Basic metrics
            pod_count = get_pod_count()
            print(f"🔢 Current Pod Count: {pod_count}")
            
            kafka_lag = get_kafka_lag()
            print(f"📊 Kafka Lag: {kafka_lag}")
            
            # KEDA status
            keda_status = get_keda_status()
            print(f"🚀 KEDA Status: {keda_status}")
            
            # External Metrics API status
            api_status = get_external_metrics_api_status()
            print(f"🔌 {api_status}")
            
            # HPA status
            hpa_status = get_hpa_status()
            print(f"📈 HPA Status: {hpa_status}")
            
            # ScaledObject status
            scaledobject_status = get_scaledobject_status()
            print(f"⚖️  ScaledObject Status: {scaledobject_status}")
            print()
            
            # Events
            scaling_events = get_scaling_events()
            print("📝 Recent Scaling Events:")
            print("-" * 50)
            print(scaling_events)
            print()
            
            print("🔄 Monitoring... (Press Ctrl+C to exit)")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped.")

if __name__ == "__main__":
    main()
