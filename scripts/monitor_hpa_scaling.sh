#!/bin/bash

# ============================================================================
# HPA SCALING MONITORING SCRIPT
# Real-time monitoring of HPA scaling decisions and metrics
# ============================================================================

set -euo pipefail

# Configuration
NAMESPACE="nimbusguard"
DEPLOYMENT="consumer"
REFRESH_INTERVAL=5
LOG_FILE="hpa_scaling_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to display header
show_header() {
    clear
    echo -e "${BLUE}=====================================================================${NC}"
    echo -e "${BLUE}🎯 NIMBUSGUARD HPA SCALING MONITOR (v2.0)${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
    echo -e "${CYAN}Namespace:${NC} $NAMESPACE"
    echo -e "${CYAN}Deployment:${NC} $DEPLOYMENT"
    echo -e "${CYAN}Refresh Interval:${NC} ${REFRESH_INTERVAL}s"
    echo -e "${CYAN}Log File:${NC} $LOG_FILE"
    echo -e "${PURPLE}Monitoring:${NC} All HPAs (baseline + enhanced)"
    echo -e "${YELLOW}Features:${NC} Smart event filtering, status detection, conflict warnings"
    echo -e "${BLUE}=====================================================================${NC}"
    echo ""
}

# Function to check HPA status
check_hpa_status() {
    echo -e "${GREEN}📊 HPA STATUS${NC}"
    echo "---------------------------------------------------------------------"
    
    if kubectl get hpa -n "$NAMESPACE" >/dev/null 2>&1; then
        # Get all HPAs targeting our deployment
        local hpas_targeting_deployment
        hpas_targeting_deployment=$(kubectl get hpa -n "$NAMESPACE" -o json | jq -r --arg deployment "$DEPLOYMENT" '.items[] | select(.spec.scaleTargetRef.name == $deployment) | .metadata.name' 2>/dev/null || echo "")
        
        if [[ -n "$hpas_targeting_deployment" ]]; then
            echo -e "${CYAN}🎯 HPAs targeting deployment '$DEPLOYMENT':${NC}"
            kubectl get hpa -n "$NAMESPACE" -o wide | head -1  # Header
            for hpa in $hpas_targeting_deployment; do
                kubectl get hpa "$hpa" -n "$NAMESPACE" -o wide 2>/dev/null || echo "  ❌ Failed to get HPA: $hpa"
            done
            
            # Count active HPAs
            local active_count
            active_count=$(echo "$hpas_targeting_deployment" | wc -w)
            if [[ "$active_count" -gt 1 ]]; then
                echo -e "${RED}⚠️  WARNING: Multiple HPAs ($active_count) targeting the same deployment!${NC}"
                echo -e "${YELLOW}💡 Only one HPA should control a deployment at a time${NC}"
            fi
        else
            # Show all HPAs if none target our deployment
            echo -e "${YELLOW}📋 All HPAs in namespace (none targeting '$DEPLOYMENT'):${NC}"
            kubectl get hpa -n "$NAMESPACE" -o wide
        fi
    else
        echo -e "${RED}❌ No HPAs found in namespace $NAMESPACE${NC}"
    fi
    echo ""
}

# Function to show detailed HPA information
show_hpa_details() {
    echo -e "${PURPLE}🔍 HPA DETAILED STATUS${NC}"
    echo "---------------------------------------------------------------------"
    
    # Get HPAs targeting our deployment first
    local hpas_targeting_deployment
    hpas_targeting_deployment=$(kubectl get hpa -n "$NAMESPACE" -o json | jq -r --arg deployment "$DEPLOYMENT" '.items[] | select(.spec.scaleTargetRef.name == $deployment) | .metadata.name' 2>/dev/null || echo "")
    
    if [[ -n "$hpas_targeting_deployment" ]]; then
        echo -e "${CYAN}🎯 HPAs targeting deployment '$DEPLOYMENT':${NC}"
        for hpa in $hpas_targeting_deployment; do
            echo -e "${CYAN}📋 HPA: $hpa${NC}"
            
            # Get HPA type (baseline/enhanced) from labels
            local hpa_type
            hpa_type=$(kubectl get hpa "$hpa" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.evaluation}' 2>/dev/null || echo "unknown")
            echo -e "${PURPLE}   Type: $hpa_type${NC}"
            
            # Show key metrics
            kubectl describe hpa "$hpa" -n "$NAMESPACE" | grep -E "(Current|Target|Conditions|Min replicas|Max replicas|Deployment pods)" || echo "   No details available"
            
            # Show recent events for this HPA
            echo -e "${YELLOW}   Recent events:${NC}"
            kubectl get events -n "$NAMESPACE" --field-selector involvedObject.name="$hpa" --sort-by='.lastTimestamp' -o custom-columns="TIME:.lastTimestamp,REASON:.reason,MESSAGE:.message" 2>/dev/null | tail -2 || echo "   No events"
            echo ""
        done
    else
        # Show all HPAs if none target our deployment
        local all_hpas
        all_hpas=$(kubectl get hpa -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
        
        if [[ -n "$all_hpas" ]]; then
            echo -e "${YELLOW}📋 All HPAs in namespace (none targeting '$DEPLOYMENT'):${NC}"
            for hpa in $all_hpas; do
                echo -e "${CYAN}📋 HPA: $hpa${NC}"
                local target_deployment
                target_deployment=$(kubectl get hpa "$hpa" -n "$NAMESPACE" -o jsonpath='{.spec.scaleTargetRef.name}' 2>/dev/null || echo "unknown")
                echo -e "${PURPLE}   Target: $target_deployment${NC}"
                kubectl describe hpa "$hpa" -n "$NAMESPACE" | grep -E "(Current|Target|Conditions)" || echo "   No details available"
                echo ""
            done
        else
            echo -e "${RED}❌ No HPAs found${NC}"
        fi
    fi
    echo ""
}

# Function to show deployment status
show_deployment_status() {
    echo -e "${YELLOW}🚀 DEPLOYMENT STATUS${NC}"
    echo "---------------------------------------------------------------------"
    
    if kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" >/dev/null 2>&1; then
        kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" -o wide || echo -e "${RED}❌ Failed to get deployment status${NC}"
        echo ""
        
        # Show pod status
        echo -e "${YELLOW}📦 POD STATUS${NC}"
        kubectl get pods -n "$NAMESPACE" -l app="$DEPLOYMENT" -o wide 2>/dev/null || echo -e "${RED}❌ No pods found for deployment $DEPLOYMENT${NC}"
    else
        echo -e "${RED}❌ Deployment $DEPLOYMENT not found in namespace $NAMESPACE${NC}"
    fi
    echo ""
}

# Function to show recent scaling events
show_scaling_events() {
    echo -e "${GREEN}📅 RECENT SCALING EVENTS${NC}"
    echo "---------------------------------------------------------------------"
    
    # Get current time for age calculation
    local current_time
    current_time=$(date -u +%s)
    
    # Get HPAs targeting our deployment
    local hpas_targeting_deployment
    hpas_targeting_deployment=$(kubectl get hpa -n "$NAMESPACE" -o json | jq -r --arg deployment "$DEPLOYMENT" '.items[] | select(.spec.scaleTargetRef.name == $deployment) | .metadata.name' 2>/dev/null || echo "")
    
    if [[ -n "$hpas_targeting_deployment" ]]; then
        # Show events for each HPA targeting our deployment
        for hpa in $hpas_targeting_deployment; do
            local hpa_type
            hpa_type=$(kubectl get hpa "$hpa" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.evaluation}' 2>/dev/null || echo "unknown")
            
            # Check current HPA status first
            local hpa_status
            hpa_status=$(kubectl get hpa "$hpa" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="ScalingActive")].status}' 2>/dev/null || echo "Unknown")
            local scaling_active_reason
            scaling_active_reason=$(kubectl get hpa "$hpa" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="ScalingActive")].reason}' 2>/dev/null || echo "Unknown")
            
            echo -e "${CYAN}🎯 HPA Status ($hpa - $hpa_type):${NC}"
            if [[ "$hpa_status" == "True" && "$scaling_active_reason" == "ValidMetricFound" ]]; then
                echo -e "${GREEN}  ✅ Status: HEALTHY - Metrics are working${NC}"
                
                # Show current metric values
                local current_metrics
                current_metrics=$(kubectl get hpa "$hpa" -n "$NAMESPACE" -o jsonpath='{.status.currentMetrics}' 2>/dev/null || echo "")
                if [[ -n "$current_metrics" ]]; then
                    local cpu_current
                    cpu_current=$(kubectl get hpa "$hpa" -n "$NAMESPACE" -o jsonpath='{.status.currentMetrics[?(@.type=="Resource" && @.resource.name=="cpu")].resource.current.averageUtilization}' 2>/dev/null || echo "N/A")
                    local memory_current
                    memory_current=$(kubectl get hpa "$hpa" -n "$NAMESPACE" -o jsonpath='{.status.currentMetrics[?(@.type=="Resource" && @.resource.name=="memory")].resource.current.averageUtilization}' 2>/dev/null || echo "N/A")
                    
                    if [[ "$cpu_current" != "N/A" ]]; then
                        echo -e "${GREEN}  📊 Current CPU: ${cpu_current}%${NC}"
                    fi
                    if [[ "$memory_current" != "N/A" ]]; then
                        echo -e "${GREEN}  📊 Current Memory: ${memory_current}%${NC}"
                    fi
                fi
            elif [[ "$hpa_status" == "False" ]]; then
                echo -e "${RED}  ❌ Status: ERROR - Cannot get metrics${NC}"
            else
                echo -e "${YELLOW}  ⚠️  Status: INITIALIZING - Waiting for metrics${NC}"
            fi
            
            # Show only recent events (last 10 minutes)
            local recent_events
            recent_events=$(kubectl get events -n "$NAMESPACE" \
                --field-selector involvedObject.name="$hpa" \
                --sort-by='.lastTimestamp' \
                -o json 2>/dev/null | jq -r --arg min_time "$((current_time - 600))" '
                .items[] | 
                select(.lastTimestamp | fromdateiso8601 > ($min_time | tonumber)) |
                "\(.lastTimestamp)|\(.reason)|\(.message)"' 2>/dev/null || echo "")
            
            if [[ -n "$recent_events" ]]; then
                echo -e "${YELLOW}  📋 Recent events (last 10 minutes):${NC}"
                                 echo "$recent_events" | while IFS='|' read -r timestamp reason message; do
                     local event_time
                     # Handle both GNU date and BSD date (macOS)
                     if command -v gdate >/dev/null 2>&1; then
                         event_time=$(gdate -d "$timestamp" +%s 2>/dev/null || echo "$current_time")
                     else
                         event_time=$(date -d "$timestamp" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%SZ" "$timestamp" +%s 2>/dev/null || echo "$current_time")
                     fi
                     local age_minutes=$(( (current_time - event_time) / 60 ))
                    
                    if [[ "$reason" == "FailedGetResourceMetric" || "$reason" == "FailedComputeMetricsReplicas" ]]; then
                        if [[ "$age_minutes" -gt 5 ]]; then
                            echo -e "${GRAY}    🕐 ${age_minutes}m ago - $reason (RESOLVED)${NC}"
                        else
                            echo -e "${RED}    🔥 ${age_minutes}m ago - $reason (ACTIVE)${NC}"
                        fi
                    elif [[ "$reason" == "SuccessfulRescale" ]]; then
                        echo -e "${GREEN}    📈 ${age_minutes}m ago - $reason${NC}"
                    else
                        echo -e "${CYAN}    ℹ️  ${age_minutes}m ago - $reason${NC}"
                    fi
                done
            else
                echo -e "${GREEN}  ✅ No recent events (system is stable)${NC}"
            fi
            echo ""
        done
    else
        # Show all HPA events if none target our deployment
        echo -e "${CYAN}🎯 All HPA Events:${NC}"
        kubectl get events -n "$NAMESPACE" \
            --field-selector involvedObject.kind=HorizontalPodAutoscaler \
            --sort-by='.lastTimestamp' \
            -o custom-columns="TIME:.lastTimestamp,OBJECT:.involvedObject.name,REASON:.reason,MESSAGE:.message" \
            2>/dev/null | tail -5 || echo "No HPA events found"
        echo ""
    fi
    
    # Deployment scaling events (only recent ones)
    echo -e "${CYAN}🚀 Recent Deployment Scaling Events ($DEPLOYMENT):${NC}"
    local recent_deployment_events
    recent_deployment_events=$(kubectl get events -n "$NAMESPACE" \
        --field-selector involvedObject.name="$DEPLOYMENT" \
        --sort-by='.lastTimestamp' \
        -o json 2>/dev/null | jq -r --arg min_time "$((current_time - 1800))" '
        .items[] | 
        select(.lastTimestamp | fromdateiso8601 > ($min_time | tonumber)) |
        select(.reason == "ScalingReplicaSet") |
        "\(.lastTimestamp)|\(.reason)|\(.message)"' 2>/dev/null || echo "")
    
    if [[ -n "$recent_deployment_events" ]]; then
                 echo "$recent_deployment_events" | while IFS='|' read -r timestamp reason message; do
             local event_time
             # Handle both GNU date and BSD date (macOS)
             if command -v gdate >/dev/null 2>&1; then
                 event_time=$(gdate -d "$timestamp" +%s 2>/dev/null || echo "$current_time")
             else
                 event_time=$(date -d "$timestamp" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%SZ" "$timestamp" +%s 2>/dev/null || echo "$current_time")
             fi
             local age_minutes=$(( (current_time - event_time) / 60 ))
            echo -e "${GREEN}  📈 ${age_minutes}m ago - $message${NC}"
        done
    else
        echo -e "${BLUE}  ⚖️  No recent scaling events (deployment is stable)${NC}"
    fi
    
    echo ""
}

# Function to check custom metrics
check_custom_metrics() {
    echo -e "${PURPLE}📈 CUSTOM METRICS STATUS${NC}"
    echo "---------------------------------------------------------------------"
    
    # Check if custom metrics API is available
    if kubectl get --raw '/apis/custom.metrics.k8s.io/v1beta1' >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Custom metrics API is available${NC}"
        
        # Try to query some common custom metrics
        local metrics=("unavailable_replicas" "container_ready_count" "container_running_count")
        for metric in "${metrics[@]}"; do
            echo -e "${CYAN}📊 Checking metric: $metric${NC}"
            kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/$NAMESPACE/deployments/$DEPLOYMENT/$metric" 2>/dev/null | jq -r '.value // "N/A"' || echo "Metric not available"
        done
    else
        echo -e "${RED}❌ Custom metrics API not available${NC}"
        echo -e "${YELLOW}💡 Install prometheus-adapter for custom metrics${NC}"
    fi
    echo ""
}

# Function to show resource utilization
show_resource_usage() {
    echo -e "${BLUE}💻 RESOURCE UTILIZATION${NC}"
    echo "---------------------------------------------------------------------"
    
    # Show resource usage if metrics-server is available
    if kubectl top pods -n "$NAMESPACE" >/dev/null 2>&1; then
        echo -e "${CYAN}📊 Pod Resource Usage:${NC}"
        kubectl top pods -n "$NAMESPACE" -l app="$DEPLOYMENT" 2>/dev/null || echo "No resource data available"
    else
        echo -e "${RED}❌ Metrics server not available${NC}"
    fi
    echo ""
}

# Function to show scaling analysis
show_scaling_analysis() {
    echo -e "${YELLOW}📈 SCALING ANALYSIS${NC}"
    echo "---------------------------------------------------------------------"
    
    local current_replicas
    current_replicas=$(kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" -o jsonpath='{.status.replicas}' 2>/dev/null || echo "0")
    
    local desired_replicas
    desired_replicas=$(kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    
    echo -e "${CYAN}Current Replicas:${NC} $current_replicas"
    echo -e "${CYAN}Desired Replicas:${NC} $desired_replicas"
    
    if [[ "$current_replicas" -gt "$desired_replicas" ]]; then
        echo -e "${GREEN}📈 Status: Scaling UP${NC}"
    elif [[ "$current_replicas" -lt "$desired_replicas" ]]; then
        echo -e "${RED}📉 Status: Scaling DOWN${NC}"
    else
        echo -e "${BLUE}⚖️  Status: STABLE${NC}"
    fi
    
    # Show which HPAs are controlling this deployment
    local hpas_targeting_deployment
    hpas_targeting_deployment=$(kubectl get hpa -n "$NAMESPACE" -o json | jq -r --arg deployment "$DEPLOYMENT" '.items[] | select(.spec.scaleTargetRef.name == $deployment) | .metadata.name' 2>/dev/null || echo "")
    
    if [[ -n "$hpas_targeting_deployment" ]]; then
        local active_count
        active_count=$(echo "$hpas_targeting_deployment" | wc -w)
        
        echo -e "${CYAN}Active HPAs:${NC} $active_count"
        for hpa in $hpas_targeting_deployment; do
            local hpa_type
            hpa_type=$(kubectl get hpa "$hpa" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.evaluation}' 2>/dev/null || echo "unknown")
            
            # Get HPA status
            local hpa_status
            hpa_status=$(kubectl get hpa "$hpa" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="AbleToScale")].status}' 2>/dev/null || echo "Unknown")
            
            if [[ "$hpa_status" == "True" ]]; then
                echo -e "${GREEN}  ✅ $hpa ($hpa_type) - Active${NC}"
            else
                echo -e "${RED}  ❌ $hpa ($hpa_type) - Inactive/Error${NC}"
            fi
        done
        
        if [[ "$active_count" -gt 1 ]]; then
            echo -e "${RED}⚠️  CONFLICT: Multiple HPAs may cause scaling issues!${NC}"
        fi
    else
        echo -e "${YELLOW}🚫 No HPAs controlling this deployment${NC}"
    fi
    
    # Log the current state
    log "Replicas: Current=$current_replicas, Desired=$desired_replicas, HPAs=$hpas_targeting_deployment"
    echo ""
}

# Function to show help
show_help() {
    echo -e "${BLUE}📖 HELP${NC}"
    echo "---------------------------------------------------------------------"
    echo "Controls:"
    echo "  q, Ctrl+C  - Quit monitoring"
    echo "  h          - Show this help"
    echo "  r          - Refresh now"
    echo "  e          - Show recent events only"
    echo "  m          - Show metrics only"
    echo "  s          - Show status only"
    echo ""
    echo "Files:"
    echo "  Log file: $LOG_FILE"
    echo ""
}

# Function to handle user input
handle_input() {
    if read -t 1 -n 1 key 2>/dev/null; then
        case $key in
            'q'|'Q')
                echo -e "\n${GREEN}👋 Monitoring stopped${NC}"
                log "Monitoring stopped by user"
                exit 0
                ;;
            'h'|'H')
                show_help
                read -p "Press Enter to continue..."
                ;;
            'r'|'R')
                echo -e "${GREEN}🔄 Refreshing...${NC}"
                ;;
            'e'|'E')
                clear
                show_header
                show_scaling_events
                read -p "Press Enter to continue..."
                ;;
            'm'|'M')
                clear
                show_header
                check_custom_metrics
                show_resource_usage
                read -p "Press Enter to continue..."
                ;;
            's'|'S')
                clear
                show_header
                check_hpa_status
                show_deployment_status
                read -p "Press Enter to continue..."
                ;;
        esac
    fi
}

# Main monitoring loop
main() {
    log "Starting HPA monitoring for $DEPLOYMENT in $NAMESPACE"
    echo -e "${GREEN}🚀 Starting HPA monitoring...${NC}"
    echo -e "${YELLOW}💡 Press 'h' for help, 'q' to quit${NC}"
    sleep 2
    
    while true; do
        show_header
        check_hpa_status
        show_deployment_status
        show_scaling_analysis
        show_scaling_events
        check_custom_metrics
        show_resource_usage
        
        echo -e "${BLUE}=====================================================================${NC}"
        echo -e "${YELLOW}Next refresh in ${REFRESH_INTERVAL}s... (Press 'h' for help, 'q' to quit)${NC}"
        
        # Handle user input during the wait period
        for ((i=0; i<REFRESH_INTERVAL; i++)); do
            handle_input
            sleep 1
        done
    done
}

# Cleanup function
cleanup() {
    echo -e "\n${GREEN}👋 Monitoring stopped${NC}"
    log "Monitoring stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check prerequisites
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

if ! command -v kubectl >/dev/null 2>&1; then
    echo -e "${RED}❌ kubectl not found. Please install kubectl.${NC}"
    exit 1
fi

if ! kubectl cluster-info >/dev/null 2>&1; then
    echo -e "${RED}❌ Unable to connect to Kubernetes cluster.${NC}"
    exit 1
fi

if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
    echo -e "${RED}❌ Namespace $NAMESPACE not found.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Check for jq (optional but recommended)
if ! command -v jq >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  jq not found. Some features may be limited.${NC}"
fi

# Start monitoring
main 