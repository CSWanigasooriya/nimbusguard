#!/bin/bash
# Backup DQN models from MinIO pod
TIMESTAMP=20250715_203352
BACKUP_DIR="model_backups/backup_20250715_203352"

echo "🔄 Backing up DQN models..."

# Get MinIO pod name
MINIO_POD=$(kubectl get pods -n nimbusguard -l app=minio -o jsonpath="{.items[0].metadata.name}")

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
        