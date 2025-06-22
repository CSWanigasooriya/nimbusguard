#!/usr/bin/env python3
"""
Manual Training Trigger Script
Use this to manually trigger model training for testing
"""

import subprocess
import sys
import time
from datetime import datetime

def run_training_job():
    """Create and run a one-time training job"""
    
    job_yaml = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: manual-training-{int(time.time())}
  namespace: nimbusguard-serving
spec:
  template:
    metadata:
      annotations:
        sidecar.istio.io/inject: "false"
    spec:
      restartPolicy: Never
      containers:
      - name: manual-trainer
        image: nimbusguard/kubeflow:latest
        command:
        - python
        - /app/continuous_training.py
        env:
        - name: PROMETHEUS_ENDPOINT
          value: "http://prometheus.monitoring.svc.cluster.local:9090"
        - name: COLLECTION_HOURS
          value: "1"  # Collect last 1 hour for quick testing
        - name: MODEL_PATH
          value: "/models"
        - name: KSERVE_MODEL_NAME
          value: "nimbusguard-dqn"
        - name: TRAINING_MODE
          value: "manual"
        - name: MIN_SAMPLES
          value: "10"  # Lower threshold for testing
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1
            memory: 2Gi
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: training-logs
          mountPath: /logs
      volumes:
      - name: model-storage
        hostPath:
          path: /Users/chamathwanigasooriya/Documents/FYP/nimbusguard/models
          type: DirectoryOrCreate
      - name: training-logs
        hostPath:
          path: /Users/chamathwanigasooriya/Documents/FYP/nimbusguard/logs
          type: DirectoryOrCreate
"""
    
    print(f"🚀 Starting manual training job at {datetime.now()}")
    
    # Apply the job
    try:
        result = subprocess.run(
            ['kubectl', 'apply', '-f', '-'],
            input=job_yaml,
            text=True,
            capture_output=True,
            check=True
        )
        print(f"✅ Training job created: {result.stdout.strip()}")
        
        # Get the job name (remove "job.batch/" prefix if present)
        job_name = result.stdout.strip().split()[0].replace('job.batch/', '')
        print(f"📋 Job name: {job_name}")
        
        # Wait for job to complete
        print("⏳ Waiting for training to complete...")
        
        while True:
            # Check job status - first check if job exists and get its status
            status_result = subprocess.run(
                ['kubectl', 'get', 'job', job_name, '-n', 'nimbusguard-serving', '-o', 'jsonpath={.status.conditions[?(@.type=="Complete")].status}'],
                capture_output=True,
                text=True
            )
            
            failed_result = subprocess.run(
                ['kubectl', 'get', 'job', job_name, '-n', 'nimbusguard-serving', '-o', 'jsonpath={.status.conditions[?(@.type=="Failed")].status}'],
                capture_output=True,
                text=True
            )
            
            if status_result.returncode == 0 and status_result.stdout.strip() == "True":
                print("✅ Training completed successfully!")
                break
            elif failed_result.returncode == 0 and failed_result.stdout.strip() == "True":
                print("❌ Training job failed!")
                break
            
            print("⏳ Training in progress...")
            time.sleep(10)
        
        # Show logs
        print("\n📜 Training logs:")
        logs_result = subprocess.run(
            ['kubectl', 'logs', f'job/{job_name}', '-n', 'nimbusguard-serving'],
            capture_output=True,
            text=True
        )
        
        if logs_result.returncode == 0:
            print(logs_result.stdout)
        else:
            print(f"Failed to get logs: {logs_result.stderr}")
        
        # Clean up job
        print(f"\n🗑️ Cleaning up job {job_name}")
        subprocess.run(['kubectl', 'delete', 'job', job_name, '-n', 'nimbusguard-serving'])
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create training job: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    return True

def check_model_update():
    """Check if the model was updated"""
    print("\n🔍 Checking for model updates...")
    
    try:
        # Check if KServe pod exists
        result = subprocess.run(
            ['kubectl', 'get', 'pods', '-n', 'nimbusguard-serving', '-l', 'serving.kserve.io/inferenceservice=nimbusguard-dqn-model', '-o', 'name'],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout.strip():
            pod_name = result.stdout.strip().split('/')[-1]
            print(f"📦 Found KServe pod: {pod_name}")
            
            # Check model version in logs
            logs_result = subprocess.run(
                ['kubectl', 'logs', pod_name, '-n', 'nimbusguard-serving', '--tail=20'],
                capture_output=True,
                text=True
            )
            
            if logs_result.returncode == 0:
                print("📜 Recent KServe logs:")
                print(logs_result.stdout)
            
            # Test prediction endpoint
            print("\n🧪 Testing prediction endpoint...")
            test_result = subprocess.run([
                'kubectl', 'exec', '-n', 'nimbusguard', 'deployment/nimbusguard-operator', '--',
                'curl', '-s', '-X', 'POST',
                'http://nimbusguard-dqn-model-predictor.nimbusguard-serving.svc.cluster.local/predict',
                '-H', 'Content-Type: application/json',
                '-d', '{"state": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}'
            ], capture_output=True, text=True)
            
            if test_result.returncode == 0:
                print("✅ Prediction endpoint test successful:")
                print(test_result.stdout)
            else:
                print(f"❌ Prediction endpoint test failed: {test_result.stderr}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to check model: {e}")

def main():
    """Main function"""
    print("🤖 NimbusGuard Manual Training Trigger")
    print("=====================================")
    
    # Check if we're in the right directory
    try:
        subprocess.run(['kubectl', 'get', 'ns', 'nimbusguard-serving'], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ nimbusguard-serving namespace not found!")
        print("Make sure you've deployed the Kubeflow components first.")
        sys.exit(1)
    
    # Run training
    if run_training_job():
        print("\n✅ Training completed!")
        
        # Wait a bit for model to be loaded
        print("⏳ Waiting for model to be loaded by KServe...")
        time.sleep(30)
        
        # Check model update
        check_model_update()
        
        print("\n🎉 Manual training process completed!")
        print("💡 The model should now be updated with new training data.")
        print("💡 Check the operator logs to see if it's using the new model.")
        
    else:
        print("❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 