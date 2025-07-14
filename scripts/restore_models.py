#!/usr/bin/env python3
"""
Restore DQN models to MinIO from local filesystem backup.
Run this after restarting your Kind cluster to restore trained models.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def restore_models(backup_path=None):
    """Restore DQN models from backup to MinIO."""
    
    if backup_path is None:
        # Use latest backup
        backup_dir = Path("./model_backups")
        latest_link = backup_dir / "latest"
        
        if not latest_link.exists():
            print("❌ No backups found. Please run backup_models.py first.")
            return False
        
        backup_path = backup_dir / latest_link.readlink()
    else:
        backup_path = Path(backup_path)
    
    if not backup_path.exists():
        print(f"❌ Backup path not found: {backup_path}")
        return False
    
    models_dir = backup_path / "models"
    if not models_dir.exists():
        print(f"❌ No models directory found in backup: {models_dir}")
        return False
    
    print(f"🔄 Restoring models from {backup_path}")
    
    try:
        # Create restore script
        restore_script = f"""#!/bin/bash
# Restore DQN models to MinIO pod
BACKUP_PATH="{backup_path}"
MODELS_DIR="{models_dir}"

echo "🔄 Restoring DQN models from $BACKUP_PATH..."

# Wait for MinIO pod to be ready
echo "⏳ Waiting for MinIO pod..."
kubectl wait --for=condition=ready pod -l app=minio -n nimbusguard --timeout=300s

# Get MinIO pod name
MINIO_POD=$(kubectl get pods -n nimbusguard -l app=minio -o jsonpath="{{.items[0].metadata.name}}")

if [ -z "$MINIO_POD" ]; then
    echo "❌ MinIO pod not found"
    exit 1
fi

echo "📦 Found MinIO pod: $MINIO_POD"

# Create models directory in MinIO if it doesn't exist
kubectl exec -n nimbusguard $MINIO_POD -- mkdir -p /data/models

# Copy models back to MinIO pod
if [ -d "$MODELS_DIR" ]; then
    echo "📁 Restoring models..."
    kubectl cp "$MODELS_DIR/." -n nimbusguard $MINIO_POD:/data/models/
    echo "✅ Models restored successfully"
else
    echo "⚠️ No models directory found in backup"
fi

# List contents to verify
echo "📋 MinIO contents after restore:"
kubectl exec -n nimbusguard $MINIO_POD -- ls -la /data/models/ 2>/dev/null || echo "Models directory empty"

echo "✅ Restore completed!"
        """
        
        script_path = backup_path / "restore.sh"
        with open(script_path, 'w') as f:
            f.write(restore_script)
        
        os.chmod(script_path, 0o755)
        
        # Execute the restore
        result = subprocess.run(['bash', str(script_path)], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model restore completed successfully!")
            print("🚀 Your DQN models are now available in MinIO")
            return True
        else:
            print(f"❌ Restore failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Restore error: {e}")
        return False

if __name__ == "__main__":
    backup_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not restore_models(backup_path):
        sys.exit(1) 