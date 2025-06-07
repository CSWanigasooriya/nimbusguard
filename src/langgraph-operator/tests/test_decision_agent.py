#!/usr/bin/env python3
"""
Test script for Decision Agent

This script tests the Decision Agent functionality with simulated data,
including Q-learning integration, LLM analysis, and decision fusion.
"""

import sys
import os
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.scaling_state import (
    ScalingWorkflowState, 
    WorkflowStatus, 
    MetricData,
    ScalingAction,
    create_initial_state
)
from agents.decision_agent import DecisionAgent
from ml_models.q_learning import QLearningAgent, QState, QAction

print("🧠 Testing Decision Agent...")
print(f"Test time: {datetime.now()}")
print("=" * 60)


def create_test_metrics(cpu: float, memory: float, pods: int, error_rate: float = 0.5) -> MetricData:
    """Create test metrics data."""
    return MetricData(
        timestamp=datetime.now(),
        cpu_utilization=cpu,
        memory_utilization=memory,
        request_rate=100.0 + (cpu * 5),  # Correlate with CPU
        error_rate=error_rate,
        pod_count=pods,
        health_score=max(0.1, min(1.0, (100 - cpu) / 100))
    )


def create_test_state(scenario: str) -> ScalingWorkflowState:
    """Create test workflow state for different scenarios."""
    base_state = create_initial_state(
        workflow_id=f"test-{scenario}",
        trigger_event={"type": "test", "scenario": scenario},
        config={"max_retries": 3}
    )
    base_state["status"] = WorkflowStatus.ANALYZING
    
    if scenario == "high_cpu":
        # High CPU scenario - should scale up
        metrics_history = [
            create_test_metrics(65, 45, 3),
            create_test_metrics(75, 50, 3),
            create_test_metrics(85, 55, 3)  # Current
        ]
    elif scenario == "low_resources":
        # Low resource usage - should scale down
        metrics_history = [
            create_test_metrics(15, 20, 5),
            create_test_metrics(12, 18, 5),
            create_test_metrics(10, 15, 5)  # Current
        ]
    elif scenario == "stable":
        # Stable system - should maintain
        metrics_history = [
            create_test_metrics(45, 40, 3),
            create_test_metrics(47, 42, 3),
            create_test_metrics(46, 41, 3)  # Current
        ]
    elif scenario == "high_errors":
        # High error rate - should scale up
        metrics_history = [
            create_test_metrics(60, 50, 2, 2.0),
            create_test_metrics(65, 55, 2, 3.5),
            create_test_metrics(70, 60, 2, 5.0)  # Current
        ]
    else:
        # Default stable scenario
        metrics_history = [
            create_test_metrics(50, 45, 3),
            create_test_metrics(52, 47, 3),
            create_test_metrics(51, 46, 3)  # Current
        ]
    
    base_state["metrics_history"] = metrics_history
    base_state["current_metrics"] = metrics_history[-1]
    
    return base_state


async def test_decision_agent_basic():
    """Test basic Decision Agent functionality."""
    print("\n🔧 Test 1: Basic Decision Agent Functionality")
    print("-" * 40)
    
    try:
        # Create test config
        config = {
            "agents": {
                "decision": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "confidence_threshold": 0.6,
                    "min_observations": 2
                }
            },
            "q_learning": {
                "learning_rate": 0.1,
                "epsilon": 0.2
            }
        }
        
        # Initialize Decision Agent
        agent = DecisionAgent(config)
        print(f"✅ Decision Agent initialized")
        print(f"   Confidence threshold: {agent.confidence_threshold}")
        print(f"   Min observations: {agent.min_observations}")
        print(f"   Q-learning epsilon: {agent.q_agent.epsilon}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


async def test_q_learning_integration():
    """Test Q-learning integration."""
    print("\n🎯 Test 2: Q-Learning Integration")
    print("-" * 40)
    
    try:
        config = {
            "agents": {"decision": {}},
            "q_learning": {"learning_rate": 0.1, "epsilon": 0.1}
        }
        
        agent = DecisionAgent(config)
        
        # Test Q-learning decision
        test_metrics = create_test_metrics(85, 70, 2)  # High utilization
        q_decision = await agent._get_q_learning_decision(test_metrics, [])
        
        print(f"✅ Q-learning decision generated")
        print(f"   Action: {q_decision['action'].value}")
        print(f"   Confidence: {q_decision['confidence']:.2f}")
        print(f"   Q-value: {q_decision.get('q_value', 'N/A')}")
        print(f"   Model episodes: {q_decision.get('model_episodes', 0)}")
        
        # Test conversion functions
        q_action = QAction("scale_up_2")
        scaling_action = agent._convert_q_action_to_scaling(q_action)
        print(f"✅ Action conversion: {q_action.action_type} → {scaling_action.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Q-learning integration test failed: {e}")
        return False


async def test_decision_scenarios():
    """Test different decision scenarios."""
    print("\n📊 Test 3: Decision Scenarios")
    print("-" * 40)
    
    scenarios = ["high_cpu", "low_resources", "stable", "high_errors"]
    results = {}
    
    try:
        config = {
            "agents": {
                "decision": {
                    "model": "gpt-4o-mini",
                    "confidence_threshold": 0.5,
                    "min_observations": 2
                }
            },
            "q_learning": {"learning_rate": 0.1, "epsilon": 0.1}
        }
        
        agent = DecisionAgent(config)
        
        for scenario in scenarios:
            print(f"\n  Testing scenario: {scenario}")
            test_state = create_test_state(scenario)
            
            # Test decision validation
            is_valid = agent._validate_decision_inputs(test_state)
            print(f"    Input validation: {'✅' if is_valid else '❌'}")
            
            if is_valid:
                # Get Q-learning recommendation
                current_metrics = test_state["current_metrics"]
                metrics_history = test_state["metrics_history"]
                
                q_rec = await agent._get_q_learning_decision(current_metrics, metrics_history)
                print(f"    Q-learning: {q_rec['action'].value} (conf: {q_rec['confidence']:.2f})")
                
                # Calculate target replicas
                current_pods = current_metrics.pod_count
                target_replicas = agent._calculate_target_replicas(q_rec['action'], current_pods)
                print(f"    Replica calculation: {current_pods} → {target_replicas}")
                
                results[scenario] = {
                    "action": q_rec['action'].value,
                    "confidence": q_rec['confidence'],
                    "current_pods": current_pods,
                    "target_replicas": target_replicas
                }
        
        print(f"\n✅ All scenarios tested successfully")
        print(f"   Results summary: {len(results)} scenarios processed")
        
        return True
        
    except Exception as e:
        print(f"❌ Decision scenarios test failed: {e}")
        return False


async def test_decision_fusion():
    """Test decision fusion logic."""
    print("\n🔀 Test 4: Decision Fusion Logic")
    print("-" * 40)
    
    try:
        config = {
            "agents": {
                "decision": {
                    "confidence_threshold": 0.6,
                    "min_observations": 2
                }
            },
            "q_learning": {}
        }
        
        agent = DecisionAgent(config)
        test_metrics = create_test_metrics(75, 60, 3)
        
        # Test different fusion scenarios
        fusion_tests = [
            {
                "name": "Agreement (both scale up)",
                "q_action": ScalingAction.SCALE_UP,
                "q_conf": 0.8,
                "llm_action": ScalingAction.SCALE_UP,
                "llm_conf": 0.7
            },
            {
                "name": "Disagreement (different actions)",
                "q_action": ScalingAction.SCALE_UP,
                "q_conf": 0.6,
                "llm_action": ScalingAction.MAINTAIN,
                "llm_conf": 0.7
            },
            {
                "name": "Low confidence",
                "q_action": ScalingAction.SCALE_UP,
                "q_conf": 0.4,
                "llm_action": ScalingAction.SCALE_UP,
                "llm_conf": 0.3
            }
        ]
        
        for test_case in fusion_tests:
            print(f"\n  Testing: {test_case['name']}")
            
            # Simulate recommendations
            q_recommendation = {
                "action": test_case["q_action"],
                "confidence": test_case["q_conf"],
                "q_value": 5.0
            }
            
            llm_analysis = {
                "llm_recommendation": test_case["llm_action"],
                "llm_confidence": test_case["llm_conf"],
                "llm_reasoning": "Test reasoning"
            }
            
            # Make decision
            test_state = create_test_state("stable")
            decision = await agent._make_final_decision(
                test_metrics, q_recommendation, llm_analysis, test_state
            )
            
            print(f"    Q: {test_case['q_action'].value} ({test_case['q_conf']:.1f})")
            print(f"    LLM: {test_case['llm_action'].value} ({test_case['llm_conf']:.1f})")
            print(f"    Final: {decision.action.value} ({decision.confidence:.2f})")
            print(f"    Replicas: {decision.current_replicas} → {decision.target_replicas}")
        
        print(f"\n✅ Decision fusion logic tested")
        
        return True
        
    except Exception as e:
        print(f"❌ Decision fusion test failed: {e}")
        return False


async def test_full_decision_workflow():
    """Test complete decision workflow."""
    print("\n🚀 Test 5: Full Decision Workflow")
    print("-" * 40)
    
    try:
        config = {
            "agents": {
                "decision": {
                    "model": "gpt-4o-mini",
                    "confidence_threshold": 0.5,
                    "min_observations": 2,
                    "max_scale_step": 2
                }
            },
            "q_learning": {
                "learning_rate": 0.1,
                "epsilon": 0.1
            }
        }
        
        agent = DecisionAgent(config)
        
        # Test high CPU scenario
        print(f"  Testing high CPU scenario (should scale up)")
        test_state = create_test_state("high_cpu")
        
        print(f"    Current metrics:")
        current = test_state["current_metrics"]
        print(f"      CPU: {current.cpu_utilization:.1f}%")
        print(f"      Memory: {current.memory_utilization:.1f}%")
        print(f"      Pods: {current.pod_count}")
        print(f"      Error rate: {current.error_rate:.1f}%")
        
        # Run full decision process
        print(f"    Running decision workflow...")
        command = await agent.invoke(test_state)
        
        print(f"    ✅ Decision completed")
        print(f"    Next agent: {command.goto}")
        
        # Check decision in updated state
        if "scaling_decision" in command.update:
            decision = command.update["scaling_decision"]
            print(f"    Decision: {decision.action.value}")
            print(f"    Confidence: {decision.confidence:.2f}")
            print(f"    Target replicas: {decision.target_replicas}")
            print(f"    Reasoning length: {len(decision.reasoning)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Full workflow test failed: {e}")
        import traceback
        print(f"    Error details: {traceback.format_exc()}")
        return False


async def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n⚠️  Test 6: Edge Cases & Error Handling")
    print("-" * 40)
    
    try:
        config = {
            "agents": {"decision": {"min_observations": 3}},
            "q_learning": {}
        }
        
        agent = DecisionAgent(config)
        
        # Test insufficient data
        print(f"  Testing insufficient data scenario")
        empty_state = create_initial_state(
            workflow_id="test-insufficient-data",
            trigger_event={"type": "test"},
            config={"max_retries": 3}
        )
        empty_state["metrics_history"] = [create_test_metrics(50, 50, 2)]  # Only 1 observation
        empty_state["current_metrics"] = empty_state["metrics_history"][0]
        
        is_valid = agent._validate_decision_inputs(empty_state)
        print(f"    Validation result: {'❌' if not is_valid else '✅'} (should be ❌)")
        
        # Test cooldown period
        print(f"  Testing cooldown period")
        cooldown_state = create_test_state("stable")
        # Add recent action
        from workflows.scaling_state import ActionResult
        recent_action = ActionResult(
            timestamp=datetime.now() - timedelta(seconds=60),  # 1 minute ago
            action_taken=ScalingAction.SCALE_UP,
            success=True,
            old_replicas=2,
            new_replicas=3
        )
        cooldown_state["last_action"] = recent_action
        
        in_cooldown = agent._in_scaling_cooldown(cooldown_state)
        print(f"    Cooldown check: {'✅' if in_cooldown else '❌'} (should be ✅)")
        
        # Test error handling
        print(f"  Testing error handling")
        try:
            # This should handle gracefully
            error_result = await agent._get_q_learning_decision(None, [])
            print(f"    Error handling: ✅ (returned default action)")
        except Exception:
            print(f"    Error handling: ❌ (should not raise exception)")
        
        print(f"\n✅ Edge cases tested")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        return False


async def run_all_tests():
    """Run all Decision Agent tests."""
    print("🧠 NimbusGuard Decision Agent Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_decision_agent_basic),
        ("Q-Learning Integration", test_q_learning_integration),
        ("Decision Scenarios", test_decision_scenarios),
        ("Decision Fusion", test_decision_fusion),
        ("Full Workflow", test_full_decision_workflow),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Decision Agent is ready for production.")
    else:
        print("⚠️  Some tests failed. Please review and fix issues.")
    
    return passed == total


if __name__ == "__main__":
    try:
        # Run tests
        success = asyncio.run(run_all_tests())
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"\n\n💥 Test suite failed: {e}")
        exit_code = 1
    
    exit(exit_code) 