#!/usr/bin/env python3
"""
Backup DQN models from MinIO to local filesystem.
Run this before restarting your Kind cluster to preserve trained models.
"""

import os
import sys
import requests
import json
from datetime import datetime
from pathlib import Path

def backup_models():
    """Backup DQN models from MinIO to local directory."""
    
    # Create backup directory
    backup_dir = Path("./model_backups")
    backup_dir.mkdir(exist_ok=True)
    
    # Timestamp for this backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_backup_dir = backup_dir / f"backup_{timestamp}"
    current_backup_dir.mkdir(exist_ok=True)
    
    print(f"🔄 Starting model backup to {current_backup_dir}")
    
    try:
        # Port forward to MinIO (you'll need to run this first)
        # kubectl port-forward -n nimbusguard svc/minio 9000:9000
        
        # Check if MinIO is accessible
        minio_url = "http://localhost:9000"
        try:
            response = requests.get(f"{minio_url}/minio/health/live", timeout=5)
            if response.status_code != 200:
                print("❌ MinIO not accessible. Please run: kubectl port-forward -n nimbusguard svc/minio 9000:9000")
                return False
        except requests.exceptions.RequestException:
            print("❌ MinIO not accessible. Please run: kubectl port-forward -n nimbusguard svc/minio 9000:9000")
            return False
        
        # Note: For simplicity, we'll use kubectl to copy files
        # You could also use the MinIO Python client for direct API access
        
        # Create a simple backup script
        backup_script = f"""#!/bin/bash
# Backup DQN models from MinIO pod
TIMESTAMP={timestamp}
BACKUP_DIR="{current_backup_dir}"

echo "🔄 Backing up DQN models..."

# Get MinIO pod name
MINIO_POD=$(kubectl get pods -n nimbusguard -l app=minio -o jsonpath="{{.items[0].metadata.name}}")

if [ -z "$MINIO_POD" ]; then
    echo "❌ MinIO pod not found"
    exit 1
fi

echo "📦 Found MinIO pod: $MINIO_POD"

# Copy models bucket from MinIO pod
kubectl exec -n nimbusguard $MINIO_POD -- ls -la /data/models/ 2>/dev/null || echo "No models found"
kubectl cp -n nimbusguard $MINIO_POD:/data/models "$BACKUP_DIR/models" 2>/dev/null || echo "No models to backup"

# Copy any other important data
kubectl exec -n nimbusguard $MINIO_POD -- ls -la /data/ > "$BACKUP_DIR/minio_contents.txt"

echo "✅ Backup completed: $BACKUP_DIR"
echo "📁 To restore: ./scripts/restore_models.py $BACKUP_DIR"
        """
        
        script_path = current_backup_dir / "backup.sh"
        with open(script_path, 'w') as f:
            f.write(backup_script)
        
        os.chmod(script_path, 0o755)
        
        # Execute the backup
        import subprocess
        result = subprocess.run(['bash', str(script_path)], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model backup completed successfully!")
            print(f"📁 Backup location: {current_backup_dir}")
            
            # Create a symlink to latest backup
            latest_link = backup_dir / "latest"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(current_backup_dir.name)
            
            return True
        else:
            print(f"❌ Backup failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Backup error: {e}")
        return False

if __name__ == "__main__":
    if not backup_models():
        sys.exit(1) 