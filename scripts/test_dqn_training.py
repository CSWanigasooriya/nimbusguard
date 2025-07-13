#!/usr/bin/env python3
"""
DQN Training Health Test Script

This script tests the current health of the DQN training system and can trigger
emergency diagnostic and fixes if needed.
"""

import requests
import json
import time
import sys
from datetime import datetime


def get_dqn_adapter_url():
    """Get the DQN adapter service URL."""
    # Try different possible URLs
    possible_urls = [
        "http://localhost:8080",  # Local development or port-forward
        "http://dqn-adapter.nimbusguard.svc.cluster.local:8080",  # In-cluster
        "http://dqn-adapter:8080"  # Docker compose
    ]
    
    for url in possible_urls:
        try:
            response = requests.get(f"{url}/healthz", timeout=5)
            if response.status_code == 200:
                print(f"✅ Found DQN adapter at: {url}")
                return url
        except Exception as e:
            print(f"❌ Failed to connect to {url}: {e}")
            continue
    
    print("❌ Could not find DQN adapter service")
    return None


def check_training_status(base_url):
    """Check current training status."""
    try:
        response = requests.get(f"{base_url}/diagnostic/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Status check failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return None


def trigger_emergency_diagnostic(base_url):
    """Trigger emergency diagnostic and fixes."""
    try:
        print("🚑 Triggering emergency diagnostic...")
        response = requests.post(f"{base_url}/diagnostic/emergency", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Emergency diagnostic failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Emergency diagnostic error: {e}")
        return None


def analyze_training_health(status):
    """Analyze training health and provide recommendations."""
    if not status:
        return False, "No status data available"
    
    issues = []
    recommendations = []
    
    training_health = status.get('training_health', {})
    model_health = status.get('model_health', {})
    
    # Check training losses
    recent_losses = training_health.get('recent_losses', [])
    if recent_losses:
        avg_loss = training_health.get('average_loss', 0)
        max_loss = training_health.get('max_loss', 0)
        
        if avg_loss > 100:
            issues.append(f"🚨 CRITICAL: Average loss extremely high ({avg_loss:.2f})")
            recommendations.append("Immediate emergency diagnostic required")
        elif avg_loss > 10:
            issues.append(f"⚠️  WARNING: Average loss high ({avg_loss:.2f})")
            recommendations.append("Consider reducing learning rate")
        
        if max_loss > 500:
            issues.append(f"🔥 CRITICAL: Maximum loss explosive ({max_loss:.2f})")
            recommendations.append("Model likely corrupted - emergency recovery needed")
    
    # Check Q-values
    sample_q_values = model_health.get('sample_q_values', [])
    if sample_q_values:
        q_range = model_health.get('q_value_range', [0, 0])
        extreme_q = model_health.get('extreme_q_values', False)
        
        if extreme_q:
            issues.append("⚡ CRITICAL: Q-values exploding (>50)")
            recommendations.append("Model reinitialization required")
        
        if len(q_range) == 2 and abs(q_range[1] - q_range[0]) > 100:
            issues.append(f"📈 WARNING: Large Q-value range ({q_range[0]:.2f} to {q_range[1]:.2f})")
            recommendations.append("Check reward scaling and feature normalization")
    
    # Check learning rate
    learning_rate = training_health.get('learning_rate', 0)
    if learning_rate > 0.001:
        issues.append(f"🎯 WARNING: Learning rate may be too high ({learning_rate:.6f})")
        recommendations.append("Consider reducing learning rate to 1e-4 or lower")
    
    # Check buffer size
    buffer_size = training_health.get('buffer_size', 0)
    if buffer_size < 10:
        issues.append("📊 WARNING: Very small replay buffer")
        recommendations.append("System may need more time to accumulate experiences")
    
    return len(issues) > 0, issues, recommendations


def print_status_report(status):
    """Print a formatted status report."""
    print("\n" + "="*60)
    print("🔍 DQN TRAINING HEALTH REPORT")
    print("="*60)
    
    if not status:
        print("❌ No status data available")
        return
    
    timestamp = status.get('timestamp', 'Unknown')
    print(f"📅 Report Time: {timestamp}")
    
    # Training Health
    training_health = status.get('training_health', {})
    print(f"\n📊 TRAINING METRICS:")
    print(f"   • Batches Trained: {training_health.get('batches_trained', 'Unknown')}")
    print(f"   • Buffer Size: {training_health.get('buffer_size', 'Unknown')}")
    print(f"   • Learning Rate: {training_health.get('learning_rate', 'Unknown'):.6f}")
    
    recent_losses = training_health.get('recent_losses', [])
    if recent_losses:
        print(f"   • Recent Losses: {recent_losses[-5:]}")  # Last 5
        print(f"   • Average Loss: {training_health.get('average_loss', 0):.4f}")
        print(f"   • Max Loss: {training_health.get('max_loss', 0):.4f}")
    else:
        print("   • No recent loss data")
    
    # Model Health
    model_health = status.get('model_health', {})
    print(f"\n🧠 MODEL HEALTH:")
    sample_q_values = model_health.get('sample_q_values', [])
    if sample_q_values:
        print(f"   • Sample Q-values: {[f'{q:.2f}' for q in sample_q_values]}")
        print(f"   • Q-value Range: {model_health.get('q_value_range', 'Unknown')}")
        print(f"   • Extreme Q-values: {'Yes' if model_health.get('extreme_q_values') else 'No'}")
    else:
        print("   • No Q-value data")
    
    # System Health
    system_health = status.get('system_health', {})
    print(f"\n⚙️  SYSTEM STATE:")
    print(f"   • Epsilon: {system_health.get('epsilon', 'Unknown'):.4f}")
    print(f"   • Decision Count: {system_health.get('decision_count', 'Unknown')}")


def main():
    """Main function."""
    print("🔍 DQN Training Health Checker")
    print("=" * 40)
    
    # Find DQN adapter
    base_url = get_dqn_adapter_url()
    if not base_url:
        sys.exit(1)
    
    # Check current status
    print("\n📊 Checking training status...")
    status = check_training_status(base_url)
    
    if not status:
        print("❌ Could not get training status")
        sys.exit(1)
    
    # Print status report
    print_status_report(status)
    
    # Analyze health
    has_issues, issues, recommendations = analyze_training_health(status)
    
    if has_issues:
        print(f"\n🚨 ISSUES DETECTED ({len(issues)}):")
        for issue in issues:
            print(f"   • {issue}")
        
        print(f"\n💡 RECOMMENDATIONS ({len(recommendations)}):")
        for rec in recommendations:
            print(f"   • {rec}")
        
        # Ask if user wants to trigger emergency diagnostic
        if any("CRITICAL" in issue for issue in issues):
            print(f"\n🚑 CRITICAL ISSUES DETECTED!")
            answer = input("Do you want to trigger emergency diagnostic and fixes? (y/N): ")
            
            if answer.lower() in ['y', 'yes']:
                result = trigger_emergency_diagnostic(base_url)
                if result:
                    print(f"\n✅ Emergency diagnostic completed!")
                    print(f"Status: {result.get('status')}")
                    print(f"Message: {result.get('message')}")
                    
                    fixes = result.get('fixes_applied', {})
                    if fixes:
                        print(f"\n🔧 FIXES APPLIED:")
                        for fix, applied in fixes.items():
                            status_icon = "✅" if applied else "❌"
                            print(f"   {status_icon} {fix.replace('_', ' ').title()}")
                    
                    # Check status again
                    print(f"\n🔄 Checking status after fixes...")
                    time.sleep(2)
                    new_status = check_training_status(base_url)
                    if new_status:
                        print_status_report(new_status)
                else:
                    print(f"❌ Emergency diagnostic failed")
    else:
        print(f"\n✅ No critical issues detected - training appears healthy!")
    
    print(f"\n" + "="*60)


if __name__ == "__main__":
    main() 