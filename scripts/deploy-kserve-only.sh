#!/bin/bash
# Deploy NimbusGuard in KServe-Only Mode

KSERVE_ENDPOINT="${1:-http://nimbusguard-dqn-model.nimbusguard-serving.svc.cluster.local/v1/models/nimbusguard-dqn:predict}"

echo "🚀 Deploying NimbusGuard in KServe-Only Mode"
echo "📍 KServe endpoint: $KSERVE_ENDPOINT"
echo ""

# Validate that KServe endpoint is provided
if [ -z "$KSERVE_ENDPOINT" ]; then
    echo "❌ KServe endpoint is required for KServe-only mode"
    echo "Usage: $0 [kserve-endpoint]"
    exit 1
fi

echo "🔧 Step 1: Apply KServe-only CRD..."
kubectl apply -f kubernetes-manifests/base/crd.yaml

echo ""
echo "🔧 Step 2: Update operator configuration..."
kubectl patch configmap operator-config -n nimbusguard --patch "
data:
  KSERVE_ENDPOINT: \"$KSERVE_ENDPOINT\"
  KSERVE_MODEL_NAME: \"nimbusguard-dqn\"
  DQN_CONFIDENCE_THRESHOLD: \"0.7\"
" 2>/dev/null || echo "⚠️  ConfigMap will be created during deployment"

echo ""
echo "🔧 Step 3: Deploy KServe-only operator..."
kubectl apply -f kubernetes-manifests/components/operator/operator.yaml

echo ""
echo "🔧 Step 4: Wait for operator to be ready..."
kubectl wait --for=condition=available deployment/nimbusguard-operator -n nimbusguard --timeout=120s

if [ $? -eq 0 ]; then
    echo "✅ Operator deployed successfully"
    
    echo ""
    echo "🔧 Step 5: Deploy KServe-only IntelligentScaling resource..."
    
    # Update the resource with the provided endpoint
    sed "s|kserve_endpoint: \".*\"|kserve_endpoint: \"$KSERVE_ENDPOINT\"|" \
        kubernetes-manifests/components/operator/intelligent-scaling-kserve-only.yaml | \
        kubectl apply -f -
    
    echo ""
    echo "🔍 Step 6: Verify KServe integration..."
    sleep 5
    
    # Check operator logs for KServe initialization
    echo "📜 Checking operator logs..."
    kubectl logs deployment/nimbusguard-operator -n nimbusguard --tail=10 | grep -i kserve || echo "⚠️  KServe logs not found yet"
    
    echo ""
    echo "✅ KServe-only deployment completed!"
    echo ""
    echo "📊 Status:"
    echo "   kubectl get intelligentscaling -n nimbusguard"
    echo ""
    echo "📜 Logs:"
    echo "   kubectl logs deployment/nimbusguard-operator -n nimbusguard -f"
    echo ""
    echo "🔍 Verify KServe connection:"
    echo "   kubectl logs deployment/nimbusguard-operator -n nimbusguard | grep 'KServe-only agent'"
else
    echo "❌ Operator deployment failed"
    exit 1
fi
