#!/bin/bash

# ============================================================================
# QUICK HPA STATUS CHECK
# One-shot HPA status and scaling information
# ============================================================================

NAMESPACE="nimbusguard"
DEPLOYMENT="consumer"

echo "🎯 QUICK HPA STATUS CHECK"
echo "=========================="
echo "Namespace: $NAMESPACE"
echo "Deployment: $DEPLOYMENT"
echo "Time: $(date)"
echo ""

echo "📊 HPA STATUS:"
if kubectl get hpa -n "$NAMESPACE" >/dev/null 2>&1; then
    # Get HPAs targeting our deployment
    hpas_targeting_deployment=$(kubectl get hpa -n "$NAMESPACE" -o json | jq -r --arg deployment "$DEPLOYMENT" '.items[] | select(.spec.scaleTargetRef.name == $deployment) | .metadata.name' 2>/dev/null || echo "")
    
    if [[ -n "$hpas_targeting_deployment" ]]; then
        echo "🎯 HPAs targeting deployment '$DEPLOYMENT':"
        kubectl get hpa -n "$NAMESPACE" -o wide | head -1  # Header
        for hpa in $hpas_targeting_deployment; do
            kubectl get hpa "$hpa" -n "$NAMESPACE" -o wide
            hpa_type=$(kubectl get hpa "$hpa" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.evaluation}' 2>/dev/null || echo "unknown")
            echo "   └── Type: $hpa_type"
        done
        
        # Check for conflicts
        active_count=$(echo "$hpas_targeting_deployment" | wc -w)
        if [[ "$active_count" -gt 1 ]]; then
            echo "⚠️  WARNING: Multiple HPAs ($active_count) targeting the same deployment!"
        fi
    else
        echo "📋 All HPAs in namespace (none targeting '$DEPLOYMENT'):"
        kubectl get hpa -n "$NAMESPACE" -o wide
    fi
else
    echo "❌ No HPAs found"
fi
echo ""

echo "🚀 DEPLOYMENT STATUS:"
kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" -o wide 2>/dev/null || echo "❌ Deployment not found"
echo ""

echo "📦 PODS:"
kubectl get pods -n "$NAMESPACE" -l app="$DEPLOYMENT" 2>/dev/null || echo "❌ No pods found"
echo ""

echo "📅 RECENT HPA EVENTS (Last 5):"
kubectl get events -n "$NAMESPACE" \
    --field-selector involvedObject.kind=HorizontalPodAutoscaler \
    --sort-by='.lastTimestamp' \
    -o custom-columns="TIME:.lastTimestamp,HPA:.involvedObject.name,REASON:.reason,MESSAGE:.message" \
    2>/dev/null | tail -5 || echo "No HPA events found"
echo ""

echo "🔧 CUSTOM METRICS API:"
if kubectl get --raw '/apis/custom.metrics.k8s.io/v1beta1' >/dev/null 2>&1; then
    echo "✅ Custom metrics API available"
else
    echo "❌ Custom metrics API not available"
fi
echo ""

echo "📈 SCALING ANALYSIS:"
current=$(kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" -o jsonpath='{.status.replicas}' 2>/dev/null || echo "0")
desired=$(kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
echo "Current replicas: $current"
echo "Desired replicas: $desired"

if [[ "$current" -gt "$desired" ]]; then
    echo "Status: 📈 SCALING UP"
elif [[ "$current" -lt "$desired" ]]; then
    echo "Status: 📉 SCALING DOWN"
else
    echo "Status: ⚖️ STABLE"
fi

# Show which HPAs are controlling this deployment
hpas_targeting_deployment=$(kubectl get hpa -n "$NAMESPACE" -o json | jq -r --arg deployment "$DEPLOYMENT" '.items[] | select(.spec.scaleTargetRef.name == $deployment) | .metadata.name' 2>/dev/null || echo "")

if [[ -n "$hpas_targeting_deployment" ]]; then
    active_count=$(echo "$hpas_targeting_deployment" | wc -w)
    echo "Active HPAs: $active_count"
    
    for hpa in $hpas_targeting_deployment; do
        hpa_type=$(kubectl get hpa "$hpa" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.evaluation}' 2>/dev/null || echo "unknown")
        hpa_status=$(kubectl get hpa "$hpa" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="AbleToScale")].status}' 2>/dev/null || echo "Unknown")
        
        if [[ "$hpa_status" == "True" ]]; then
            echo "  ✅ $hpa ($hpa_type) - Active"
        else
            echo "  ❌ $hpa ($hpa_type) - Inactive/Error"
        fi
    done
    
    if [[ "$active_count" -gt 1 ]]; then
        echo "⚠️  CONFLICT: Multiple HPAs may cause scaling issues!"
    fi
else
    echo "🚫 No HPAs controlling this deployment"
fi

echo ""
echo "💡 For continuous monitoring, run: ./scripts/monitor_hpa_scaling.sh" 