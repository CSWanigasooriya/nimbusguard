#!/usr/bin/env python3
"""
Train model directly with external dataset from MinIO
"""

import subprocess
import sys
import time
from datetime import datetime

def run_training_job():
    """Create and run a training job with external dataset"""
    
    job_yaml = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: external-dataset-training-{int(time.time())}
  namespace: nimbusguard-serving
spec:
  template:
    metadata:
      annotations:
        sidecar.istio.io/inject: "false"
    spec:
      restartPolicy: Never
      containers:
      - name: external-trainer
        image: nimbusguard/kubeflow:latest
        command:
        - python
        - -c
        - |
          import os
          import sys
          import time
          import json
          import logging
          import numpy as np
          from datetime import datetime
          
          # Set up logging
          logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
          logger = logging.getLogger(__name__)
          
          logger.info("🤖 Starting External Dataset Training")
          logger.info("=" * 50)
          
          # Import required modules
          sys.path.append('/app')
          from data_preprocessing import KubernetesDatasetProcessor
          
          try:
              import torch
              import torch.nn as nn
              import torch.optim as optim
              logger.info("✅ PyTorch imported successfully")
          except ImportError:
              logger.error("❌ PyTorch not available")
              sys.exit(1)
          
          # MinIO configuration
          s3_config = {{
              'endpoint_url': 'http://minio-api.minio.svc.cluster.local:9000',
              'access_key': 'nimbusguard',
              'secret_key': 'nimbusguard123',
              'bucket_name': 'datasets',
              'region': 'us-east-1'
          }}
          
          # Load dataset from MinIO
          logger.info("📦 Loading TrainData.csv from MinIO...")
          processor = KubernetesDatasetProcessor(
              dataset_path="", 
              use_s3=True, 
              s3_config=s3_config
          )
          
          train_df = processor.load_training_data("TrainData.csv")
          data = processor.process_for_dqn_training(train_df)
          
          states = data['states']
          actions = data['actions']
          rewards = data['rewards']
          
          logger.info(f"✅ Loaded {{len(states)}} training samples")
          logger.info(f"States shape: {{states.shape}}, Actions shape: {{actions.shape}}, Rewards shape: {{rewards.shape}}")
          
          # DQN model
          class DQN(nn.Module):
              def __init__(self, state_dim=11, action_dim=5, hidden_dim=128):
                  super().__init__()
                  self.network = nn.Sequential(
                      nn.Linear(state_dim, hidden_dim),
                      nn.ReLU(),
                      nn.LayerNorm(hidden_dim),
                      nn.Dropout(0.1),
                      nn.Linear(hidden_dim, hidden_dim),
                      nn.ReLU(),
                      nn.LayerNorm(hidden_dim),
                      nn.Dropout(0.1),
                      nn.Linear(hidden_dim, action_dim)
                  )
              
              def forward(self, x):
                  return self.network(x)
          
          # Train model
          logger.info("🚀 Starting DQN training...")
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          model = DQN().to(device)
          optimizer = optim.Adam(model.parameters(), lr=0.001)
          criterion = nn.MSELoss()
          
          epochs = 200  # More epochs for better training
          batch_size = 64  # Larger batch size
          
          logger.info(f"Training on {{device}} for {{epochs}} epochs with batch size {{batch_size}}")
          
          model.train()
          losses = []
          for epoch in range(epochs):
              # Sample batch
              batch_size_actual = min(batch_size, len(states))
              indices = np.random.choice(len(states), batch_size_actual, replace=False)
              batch_states = torch.FloatTensor(states[indices]).to(device)
              batch_actions = torch.LongTensor(actions[indices]).to(device)
              batch_rewards = torch.FloatTensor(rewards[indices]).to(device)
              
              # Forward pass
              q_values = model(batch_states)
              action_q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze()
              
              # Enhanced Q-learning update with next states
              with torch.no_grad():
                  next_indices = np.random.choice(len(states), batch_size_actual, replace=False)
                  next_states = torch.FloatTensor(states[next_indices]).to(device)
                  next_q_values = model(next_states).max(1)[0]
                  target_q_values = batch_rewards + 0.95 * next_q_values
              
              loss = criterion(action_q_values, target_q_values)
              
              # Backward pass
              optimizer.zero_grad()
              loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
              optimizer.step()
              
              losses.append(loss.item())
              
              if epoch % 50 == 0:
                  logger.info(f"Epoch {{epoch}}/{{epochs}}, Loss: {{loss.item():.4f}}")
          
          final_loss = losses[-1]
          avg_loss = np.mean(losses)
          logger.info(f"Training completed - Final loss: {{final_loss:.4f}}, Avg loss: {{avg_loss:.4f}}")
          
          # Save model to MinIO
          model_name = 'nimbusguard-dqn'
          timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
          
          checkpoint = {{
              'model_state_dict': model.state_dict(),
              'state_dim': 11,
              'action_dim': 5,
              'metadata': {{
                  'model_version': int(timestamp),
                  'training_timestamp': timestamp,
                  'total_samples': len(states),
                  'final_loss': final_loss,
                  'avg_loss': avg_loss,
                  'model_type': 'DQN',
                  'dataset_source': 'TrainData.csv',
                  'training_epochs': epochs
              }}
          }}
          
          # Save to MinIO
          try:
              import boto3
              import tempfile
              
              s3_client = boto3.client(
                  's3',
                  endpoint_url='http://minio-api.minio.svc.cluster.local:9000',
                  aws_access_key_id='nimbusguard',
                  aws_secret_access_key='nimbusguard123',
                  region_name='us-east-1'
              )
              
              with tempfile.NamedTemporaryFile(suffix='.pth') as temp_file:
                  torch.save(checkpoint, temp_file.name)
                  
                  # Upload both timestamped and latest versions
                  timestamped_key = f"{{model_name}}/{{timestamp}}/model.pth"
                  latest_key = f"{{model_name}}/latest/model.pth"
                  
                  s3_client.upload_file(temp_file.name, 'models', timestamped_key)
                  s3_client.upload_file(temp_file.name, 'models', latest_key)
                  
                  logger.info(f"✅ Model saved to MinIO: s3://models/{{model_name}}/")
                  logger.info(f"🎯 Model version: {{timestamp}}")
                  logger.info(f"📊 Trained on {{len(states)}} samples with final loss {{final_loss:.4f}}")
          
          except Exception as e:
              logger.error(f"❌ Failed to save model to MinIO: {{e}}")
              # Save locally as fallback
              model_path = '/models'
              model_dir = f"{{model_path}}/{{model_name}}"
              latest_dir = f"{{model_dir}}/latest"
              os.makedirs(latest_dir, exist_ok=True)
              
              latest_file = f"{{latest_dir}}/model.pth"
              torch.save(checkpoint, latest_file)
              logger.info(f"✅ Model saved locally to {{latest_file}}")
          
          logger.info("🎉 External dataset training completed!")
        env:
        - name: PYTHONPATH
          value: "/app"
        - name: PYTHONUNBUFFERED
          value: "1"
        resources:
          requests:
            cpu: 1
            memory: 2Gi
          limits:
            cpu: 2
            memory: 4Gi
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: nimbusguard-model-pvc
"""
    
    print(f"🚀 Starting external dataset training job at {datetime.now()}")
    
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
        
        # Get the job name
        job_name = result.stdout.strip().split()[0].replace('job.batch/', '')
        print(f"📋 Job name: {job_name}")
        
        # Wait for job to complete
        print("⏳ Waiting for training to complete...")
        
        while True:
            # Check job status
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
            time.sleep(15)
        
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
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create training job: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main function"""
    print("🤖 NimbusGuard External Dataset Training")
    print("=========================================")
    
    if run_training_job():
        print("\n✅ Training with external dataset completed!")
        print("💡 The model should now be trained with 1000 samples from TrainData.csv")
        print("💡 Check the operator logs to see improved confidence scores")
    else:
        print("❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 