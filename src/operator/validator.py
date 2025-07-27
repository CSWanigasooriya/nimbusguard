import logging
import os
import json
import requests
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from validation_prompt import VALIDATION_PROMPT

class ScalingValidator:
    """
    Validates scaling actions and deployment states for safety and compliance.
    Provides comprehensive checks before scaling operations are executed.
    Supports both regular validation and optional LLM-based validation.
    """
    
    def __init__(self, config=None):
        """
        Initialize validator with configuration.
        
        Args:
            config: Dictionary with validation parameters
        """
        self.config = config or {}
        
        # LLM validation settings
        self.enable_llm_validation = bool(os.getenv('ENABLE_LLM_VALIDATION', 'false').lower() == 'true')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.ai_model = os.getenv('AI_MODEL', 'gpt-4-turbo')
        self.ai_temperature = float(os.getenv('AI_TEMPERATURE', '0.1'))
        self.mcp_server_url = os.getenv('MCP_SERVER_URL', 'http://mcp-server.nimbusguard.svc:8080')
        
        # Scaling rate limits
        self.max_scale_up_per_minute = self.config.get('max_scale_up_per_minute', 3)
        self.max_scale_down_per_minute = self.config.get('max_scale_down_per_minute', 2)
        self.min_time_between_scales = self.config.get('min_time_between_scales', 30)  # seconds
        
        # Resource thresholds
        self.max_cpu_util_for_scale_down = self.config.get('max_cpu_util_for_scale_down', 80.0)
        self.min_memory_util_for_scale_up = self.config.get('min_memory_util_for_scale_up', 60.0)
        self.critical_memory_threshold = self.config.get('critical_memory_threshold', 95.0)
        
        # Stability requirements
        self.min_stable_period_seconds = self.config.get('min_stable_period_seconds', 60)
        self.max_utilization_variance = self.config.get('max_utilization_variance', 20.0)
        
        # Safety limits
        self.emergency_scale_up_threshold = self.config.get('emergency_scale_up_threshold', 90.0)
        self.force_scale_down_threshold = self.config.get('force_scale_down_threshold', 10.0)
        
        # Log validation mode
        if self.enable_llm_validation:
            if self.openai_api_key:
                logging.info("LLM validation enabled with OpenAI API")
            else:
                logging.warning("LLM validation enabled but OpenAI API key not configured - falling back to regular validation")
                self.enable_llm_validation = False
        else:
            logging.info("Using regular validation (LLM validation disabled)")
    
    def validate_scaling_action(self, action: int, current_replicas: int, target_replicas: int,
                               min_replicas: int, max_replicas: int, deployment_info: Dict = None) -> Tuple[bool, str, int]:
        """
        Comprehensive validation of a scaling action for multi-pod environments.
        
        Key validation principles:
        - Trust memory metrics as reported (high usage indicates real problems)
        - Prevent dangerous scale-downs during memory pressure
        - Allow emergency scale-ups when critically needed
        - Rate limit scaling actions to prevent oscillation
        - Validate system stability before scaling
        
        Args:
            action: Scaling action (0=none, 1=scale_up, 2=scale_down)
            current_replicas: Current replica count
            target_replicas: Desired replica count
            min_replicas: Minimum allowed replicas
            max_replicas: Maximum allowed replicas
            deployment_info: Deployment metrics including:
                - current_cpu_util: CPU utilization %
                - current_mem_util: Memory utilization % (trusted as accurate)
                - predicted_mem_util: Predicted memory utilization %
            
        Returns:
            Tuple of (is_valid, reason, adjusted_target)
        """
        # First, always run regular validation for safety
        regular_valid, regular_reason, regular_target = self._validate_regular(
            action, current_replicas, target_replicas, min_replicas, max_replicas, deployment_info
        )
        
        # If regular validation fails, return immediately
        if not regular_valid:
            return regular_valid, regular_reason, regular_target
        
        # If LLM validation is enabled, run additional LLM check
        if self.enable_llm_validation and self.openai_api_key:
            try:
                llm_valid, llm_reason, llm_target = self._validate_with_llm(
                    action, current_replicas, target_replicas, min_replicas, max_replicas, deployment_info
                )
                
                # LLM validation can override regular validation decision
                if not llm_valid:
                    logging.info(f"LLM validation override: {llm_reason}")
                    return llm_valid, f"LLM Override: {llm_reason}", llm_target
                else:
                    logging.info(f"LLM validation confirmed: {llm_reason}")
                    
            except Exception as e:
                logging.error(f"LLM validation failed, falling back to regular validation: {e}")
                # Continue with regular validation result
        
        return regular_valid, regular_reason, regular_target
    
    def _validate_replica_boundaries(self, target_replicas: int, min_replicas: int, max_replicas: int) -> Tuple[bool, str, int]:
        """Validate replica count against boundaries."""
        if target_replicas < min_replicas:
            return False, f"Target replicas {target_replicas} below minimum {min_replicas}", min_replicas
        
        if target_replicas > max_replicas:
            return False, f"Target replicas {target_replicas} above maximum {max_replicas}", max_replicas
        
        return True, "Boundary validation passed", target_replicas
    
    def _validate_scaling_rate(self, action: int, current_replicas: int, target_replicas: int) -> Tuple[bool, str]:
        """Validate scaling rate limits."""
        from state_manager import state
        
        # Check if we have scaling history
        if not hasattr(state, 'scaling_history') or not state.scaling_history:
            return True, "No scaling history to check"
        
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Count recent scaling actions
        recent_scale_ups = 0
        recent_scale_downs = 0
        last_scaling_time = None
        
        for record in state.scaling_history:
            if record['timestamp'] > one_minute_ago:
                if record['action'] == 'scale_up':
                    recent_scale_ups += 1
                elif record['action'] == 'scale_down':
                    recent_scale_downs += 1
                
                if last_scaling_time is None or record['timestamp'] > last_scaling_time:
                    last_scaling_time = record['timestamp']
        
        # Check rate limits
        if action == 1:  # Scale up
            if recent_scale_ups >= self.max_scale_up_per_minute:
                return False, f"Scale up rate limit exceeded: {recent_scale_ups}/{self.max_scale_up_per_minute} in last minute"
        elif action == 2:  # Scale down
            if recent_scale_downs >= self.max_scale_down_per_minute:
                return False, f"Scale down rate limit exceeded: {recent_scale_downs}/{self.max_scale_down_per_minute} in last minute"
        
        # Check minimum time between scaling actions
        if last_scaling_time:
            time_since_last = (now - last_scaling_time).total_seconds()
            if time_since_last < self.min_time_between_scales:
                return False, f"Minimum time between scales not met: {time_since_last:.0f}s < {self.min_time_between_scales}s"
        
        return True, "Rate limit validation passed"
    
    def _validate_resource_constraints(self, action: int, current_replicas: int, target_replicas: int, 
                                     deployment_info: Dict) -> Tuple[bool, str]:
        """Validate against resource utilization constraints."""
        # This would typically use current resource metrics
        # For now, we'll use placeholder logic
        
        # Get current utilization (would come from metrics)
        current_cpu_util = deployment_info.get('current_cpu_util', 0)
        current_mem_util = deployment_info.get('current_mem_util', 0)
        predicted_mem_util = deployment_info.get('predicted_mem_util', current_mem_util)
        
        # Calculate effective memory utilization (accounting for multiple pods)
        effective_mem_util = self._calculate_effective_memory_utilization(current_mem_util, current_replicas)
        
        max_mem_util = max(effective_mem_util, predicted_mem_util)
        
        # Emergency scale-up logic - trust memory metrics and act decisively
        if max_mem_util > self.emergency_scale_up_threshold:
            if action == 1:  # Already scaling up - approve it
                return True, f"Scale-up approved for emergency memory pressure (memory: {max_mem_util:.1f}% > {self.emergency_scale_up_threshold}%)"
            elif action == 2:  # Scale down - absolutely prevent it
                return False, f"Scale-down BLOCKED: Emergency memory pressure detected (memory: {max_mem_util:.1f}% > {self.emergency_scale_up_threshold}%)"
            else:  # No action - allow validation to continue, emergency logic in should_force_action()
                # High memory usage is a real problem that needs addressing
                # Don't reject here, but should_force_action() will recommend emergency scaling
                logging.warning(f"High memory pressure detected but no scaling action proposed (memory: {max_mem_util:.1f}%)")
        
        # Prevent scale-down under high load
        if action == 2:  # Scale down
            if current_cpu_util > self.max_cpu_util_for_scale_down:
                return False, f"Cannot scale down: CPU utilization too high ({current_cpu_util:.1f}% > {self.max_cpu_util_for_scale_down}%)"
            
            if current_mem_util > self.max_cpu_util_for_scale_down:
                return False, f"Cannot scale down: Memory utilization too high ({current_mem_util:.1f}% > {self.max_cpu_util_for_scale_down}%)"
        
        # Require minimum utilization for scale-up
        if action == 1:  # Scale up
            if (current_mem_util < self.min_memory_util_for_scale_up and 
                predicted_mem_util < self.min_memory_util_for_scale_up):
                return False, f"Scale-up not justified: Memory utilization too low (current: {current_mem_util:.1f}%, predicted: {predicted_mem_util:.1f}% < {self.min_memory_util_for_scale_up}%)"
        
        return True, "Resource constraint validation passed"
    
    def _validate_system_stability(self, action: int) -> Tuple[bool, str]:
        """Validate system stability before scaling."""
        from state_manager import state
        
        # Check if deployment is in a stable state
        # This would typically check metrics variance, pod readiness, etc.
        
        # Check for recent deployment instability
        if hasattr(state, 'scaling_history') and state.scaling_history:
            recent_actions = []
            five_minutes_ago = datetime.now() - timedelta(minutes=5)
            
            for record in state.scaling_history:
                if record['timestamp'] > five_minutes_ago:
                    recent_actions.append(record['action'])
            
            # Check for oscillating behavior
            if len(recent_actions) >= 4:
                scale_ups = recent_actions.count('scale_up')
                scale_downs = recent_actions.count('scale_down')
                
                if scale_ups > 0 and scale_downs > 0:
                    return False, f"System instability detected: {scale_ups} scale-ups and {scale_downs} scale-downs in last 5 minutes"
        
        return True, "Stability validation passed"
    
    def validate_deployment_health(self, deployment_info: Dict) -> Tuple[bool, str]:
        """
        Validate deployment health before allowing scaling.
        
        Args:
            deployment_info: Deployment information dictionary
            
        Returns:
            Tuple of (is_healthy, reason)
        """
        if not deployment_info:
            return False, "No deployment information provided"
        
        current = deployment_info.get('current_replicas', 0)
        available = deployment_info.get('available_replicas', 0)
        ready = deployment_info.get('ready_replicas', 0)
        
        if available < current:
            return False, f"Not all replicas available: {available}/{current}"
        
        if ready < current:
            return False, f"Not all replicas ready: {ready}/{current}"
        
        # Check for recent failed scaling attempts
        from state_manager import state
        if hasattr(state, 'scaling_history'):
            recent_failures = []
            ten_minutes_ago = datetime.now() - timedelta(minutes=10)
            
            for record in state.scaling_history:
                if (record['timestamp'] > ten_minutes_ago and 
                    record.get('success') is False):
                    recent_failures.append(record)
            
            if len(recent_failures) >= 3:
                return False, f"Too many recent scaling failures: {len(recent_failures)} in last 10 minutes"
        
        return True, "Deployment is healthy"
    
    def should_force_action(self, current_cpu_util: float, current_mem_util: float, 
                           predicted_mem_util: float, current_replicas: int, 
                           min_replicas: int, max_replicas: int) -> Tuple[Optional[int], str]:
        """
        Determine if a forced scaling action is required for safety.
        
        Args:
            current_cpu_util: Current CPU utilization percentage
            current_mem_util: Current memory utilization percentage
            predicted_mem_util: Predicted memory utilization percentage
            current_replicas: Current replica count
            min_replicas: Minimum allowed replicas
            max_replicas: Maximum allowed replicas
            
        Returns:
            Tuple of (forced_action, reason) where forced_action is None if no force needed
        """
        # Apply multi-pod memory adjustment for force action decisions
        effective_mem_util = self._calculate_effective_memory_utilization(current_mem_util, current_replicas)
        
        max_mem_util = max(effective_mem_util, predicted_mem_util)
        
        # Force scale-up for critical resource usage
        if max_mem_util > self.critical_memory_threshold:
            if current_replicas < max_replicas:
                return 1, f"CRITICAL: Force scale-up due to memory pressure ({max_mem_util:.1f}% > {self.critical_memory_threshold}%)"
        
        # Force scale-up for emergency threshold (less critical but still urgent)
        elif max_mem_util > self.emergency_scale_up_threshold:
            if current_replicas < max_replicas:
                return 1, f"EMERGENCY: Force scale-up due to high memory ({max_mem_util:.1f}% > {self.emergency_scale_up_threshold}%)"
        
        # Force scale-down for very low utilization
        if (current_cpu_util < self.force_scale_down_threshold and 
            effective_mem_util < self.force_scale_down_threshold):
            if current_replicas > min_replicas:
                return 2, f"Force scale-down due to very low utilization (CPU: {current_cpu_util:.1f}%, Mem: {effective_mem_util:.1f}% < {self.force_scale_down_threshold}%)"
        
        return None, "No forced action required"
    
    def get_validation_summary(self, action: int, current_replicas: int, target_replicas: int,
                              min_replicas: int, max_replicas: int, deployment_info: Dict = None) -> Dict:
        """
        Get a comprehensive validation summary for analysis.
        
        Returns:
            Dictionary with detailed validation results
        """
        summary = {
            'timestamp': datetime.now(),
            'action': action,
            'current_replicas': current_replicas,
            'target_replicas': target_replicas,
            'constraints': {
                'min_replicas': min_replicas,
                'max_replicas': max_replicas
            },
            'validations': {}
        }
        
        # Run all validations
        boundary_valid, boundary_reason, boundary_target = self._validate_replica_boundaries(
            target_replicas, min_replicas, max_replicas
        )
        summary['validations']['boundary'] = {
            'valid': boundary_valid,
            'reason': boundary_reason,
            'adjusted_target': boundary_target
        }
        
        rate_valid, rate_reason = self._validate_scaling_rate(action, current_replicas, target_replicas)
        summary['validations']['rate_limit'] = {
            'valid': rate_valid,
            'reason': rate_reason
        }
        
        if deployment_info:
            resource_valid, resource_reason = self._validate_resource_constraints(
                action, current_replicas, target_replicas, deployment_info
            )
            summary['validations']['resource_constraints'] = {
                'valid': resource_valid,
                'reason': resource_reason
            }
            
            health_valid, health_reason = self.validate_deployment_health(deployment_info)
            summary['validations']['deployment_health'] = {
                'valid': health_valid,
                'reason': health_reason
            }
        
        stability_valid, stability_reason = self._validate_system_stability(action)
        summary['validations']['stability'] = {
            'valid': stability_valid,
            'reason': stability_reason
        }
        
        # Overall validation result
        all_valid = all(v.get('valid', True) for v in summary['validations'].values())
        summary['overall_valid'] = all_valid
        summary['final_target'] = target_replicas if all_valid else current_replicas
        
        return summary
    
    def update_config(self, new_config: Dict):
        """Update validator configuration."""
        self.config.update(new_config)
        
        # Update individual parameters
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logging.info(f"Updated validator config: {new_config}")
    
    def _calculate_effective_memory_utilization(self, current_mem_util: float, current_replicas: int) -> float:
        """
        Calculate effective memory utilization for validation purposes.
        
        In multi-pod environments, we trust the memory metrics as reported since:
        - High memory usage indicates genuine resource pressure
        - Load balancing issues are real problems that need addressing
        - Memory metrics should not be artificially reduced
        
        Args:
            current_mem_util: Raw memory utilization percentage
            current_replicas: Current number of replicas
            
        Returns:
            Memory utilization percentage (unchanged - we trust the metrics)
        """
        # Return the actual memory utilization without adjustment
        # High memory usage in multi-pod environments indicates real problems:
        # - Uneven load distribution
        # - Memory pressure on individual pods
        # - Need for additional capacity
        return current_mem_util
    
    def get_config(self) -> Dict:
        """Get current validator configuration."""
        return {
            'max_scale_up_per_minute': self.max_scale_up_per_minute,
            'max_scale_down_per_minute': self.max_scale_down_per_minute,
            'min_time_between_scales': self.min_time_between_scales,
            'max_cpu_util_for_scale_down': self.max_cpu_util_for_scale_down,
            'min_memory_util_for_scale_up': self.min_memory_util_for_scale_up,
            'critical_memory_threshold': self.critical_memory_threshold,
            'min_stable_period_seconds': self.min_stable_period_seconds,
            'max_utilization_variance': self.max_utilization_variance,
            'emergency_scale_up_threshold': self.emergency_scale_up_threshold,
            'force_scale_down_threshold': self.force_scale_down_threshold
        } 

    def _validate_regular(self, action: int, current_replicas: int, target_replicas: int,
                         min_replicas: int, max_replicas: int, deployment_info: Dict = None) -> Tuple[bool, str, int]:
        """
        Regular validation logic (the original validation).
        """        
        # Basic boundary validation
        boundary_valid, boundary_reason, boundary_target = self._validate_replica_boundaries(
            target_replicas, min_replicas, max_replicas
        )
        if not boundary_valid:
            return False, boundary_reason, boundary_target
        
        # No-change validation
        if target_replicas == current_replicas:
            return True, "No scaling required", current_replicas
        
        # Rate limiting validation
        rate_valid, rate_reason = self._validate_scaling_rate(action, current_replicas, target_replicas)
        if not rate_valid:
            return False, rate_reason, current_replicas
        
        # Resource-based validation
        if deployment_info:
            resource_valid, resource_reason = self._validate_resource_constraints(
                action, current_replicas, target_replicas, deployment_info
            )
            if not resource_valid:
                return False, resource_reason, current_replicas
        
        # Stability validation
        stability_valid, stability_reason = self._validate_system_stability(action)
        if not stability_valid:
            return False, stability_reason, current_replicas
        
        return True, "Regular validation passed", target_replicas
    
    def _validate_with_llm(self, action: int, current_replicas: int, target_replicas: int,
                          min_replicas: int, max_replicas: int, deployment_info: Dict = None) -> Tuple[bool, str, int]:
        """
        LLM-based validation using OpenAI API with MCP tools.
        
        Returns:
            Tuple of (is_valid, reason, adjusted_target)
        """
        try:
            # Run async validation
            return asyncio.run(self._validate_with_llm_async(
                action, current_replicas, target_replicas, min_replicas, max_replicas, deployment_info
            ))
        except Exception as e:
            logging.error(f"LLM validation failed: {e}")
            return False, f"LLM validation error: {str(e)}", current_replicas
    
    async def _validate_with_llm_async(self, action: int, current_replicas: int, target_replicas: int,
                                      min_replicas: int, max_replicas: int, deployment_info: Dict = None) -> Tuple[bool, str, int]:
        """
        Async LLM-based validation using OpenAI API with MCP tools.
        """
        try:
                         # Import LLM dependencies
            from langchain_openai import ChatOpenAI
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langgraph.prebuilt import create_react_agent
            
            # Initialize LLM
            llm = ChatOpenAI(
                model=self.ai_model,
                temperature=self.ai_temperature,
                api_key=self.openai_api_key
            )
            
            # Initialize MCP client
            mcp_client = MultiServerMCPClient(
                connections={
                    "kubernetes": {
                        "url": f"{self.mcp_server_url}/sse",
                        "transport": "sse"
                    }
                }
            )
            
            # Get tools from MCP client
            tools = await mcp_client.get_tools()
            
            # Create react agent with LLM and tools
            agent = create_react_agent(llm, tools)
            
            # Prepare context for LLM
            context = {
                "scaling_action": {
                    "action": action,
                    "action_name": ["no_action", "scale_up", "scale_down"][action] if 0 <= action <= 2 else "unknown",
                    "current_replicas": current_replicas,
                    "target_replicas": target_replicas,
                    "replica_change": target_replicas - current_replicas
                },
                "constraints": {
                    "min_replicas": min_replicas,
                    "max_replicas": max_replicas
                },
                "deployment_info": deployment_info or {},
                "timestamp": datetime.now().isoformat(),
                "system_thresholds": {
                    "emergency_scale_up_threshold": self.emergency_scale_up_threshold,
                    "critical_memory_threshold": self.critical_memory_threshold,
                    "max_cpu_util_for_scale_down": self.max_cpu_util_for_scale_down,
                    "min_memory_util_for_scale_up": self.min_memory_util_for_scale_up
                }
            }
            
                         # Create validation prompt with MCP tools
            validation_prompt = self._create_validation_prompt(context, tools)
            
            # Get agent response with access to Kubernetes tools
            response = await agent.ainvoke({
                "messages": [("human", validation_prompt)]
            })
            
            # Extract the final message content
            if response and "messages" in response:
                final_message = response["messages"][-1]
                response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
            else:
                response_content = str(response)
            
            # Parse LLM response
            return self._parse_llm_response_content(response_content, target_replicas, current_replicas)
            
        except Exception as e:
            logging.error(f"LLM validation with MCP tools failed: {e}")
            return False, f"LLM validation error: {str(e)}", current_replicas
    

    
    def _create_validation_prompt(self, context: Dict, mcp_tools) -> str:
        """Create a structured prompt for LLM validation using MCP tools."""
        action_name = context["scaling_action"]["action_name"]
        current_replicas = context["scaling_action"]["current_replicas"]
        target_replicas = context["scaling_action"]["target_replicas"]
        replica_change = context["scaling_action"]["replica_change"]
        
        deployment_info = context["deployment_info"]
        current_cpu = deployment_info.get("current_cpu_util", "unknown")
        current_mem = deployment_info.get("current_mem_util", "unknown")
        predicted_mem = deployment_info.get("predicted_mem_util", "unknown")
        
        # Extract tool names and descriptions from MCP tools
        tool_descriptions = []
        for tool in mcp_tools:
            tool_name = getattr(tool, 'name', str(tool))
            tool_desc = getattr(tool, 'description', '')
            if tool_desc:
                tool_descriptions.append(f"{tool_name}: {tool_desc}")
            else:
                tool_descriptions.append(tool_name)
        tools_text = ", ".join(tool_descriptions)
        
        # Format the prompt with all the context
        prompt = VALIDATION_PROMPT.format(
            tools=tools_text,
            action_name=action_name,
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            replica_change=replica_change,
            min_replicas=context["constraints"]["min_replicas"],
            max_replicas=context["constraints"]["max_replicas"],
            current_cpu=current_cpu,
            current_mem=current_mem,
            predicted_mem=predicted_mem,
            emergency_scale_up_threshold=context["system_thresholds"]["emergency_scale_up_threshold"],
            critical_memory_threshold=context["system_thresholds"]["critical_memory_threshold"],
            max_cpu_util_for_scale_down=context["system_thresholds"]["max_cpu_util_for_scale_down"],
            min_memory_util_for_scale_up=context["system_thresholds"]["min_memory_util_for_scale_up"],
            deployment_info=json.dumps(deployment_info, indent=2) if deployment_info else "No additional deployment information"
        )
        
        return prompt.strip()
    
    def _parse_llm_response_content(self, content: str, target_replicas: int, current_replicas: int) -> Tuple[bool, str, int]:
        """Parse LLM response content and extract validation decision."""
        try:
            # Clean up the content - sometimes LLM responses have extra text
            content = content.strip()
            
            # Try to extract JSON from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = content[json_start:json_end]
            else:
                json_content = content
            
            # Try to parse JSON response
            llm_decision = json.loads(json_content)
            
            approved = llm_decision.get("approved", False)
            reason = llm_decision.get("reason", "LLM validation completed")
            confidence = llm_decision.get("confidence", 0.0)
            recommended_target = llm_decision.get("recommended_target", target_replicas)
            risk_assessment = llm_decision.get("risk_assessment", "UNKNOWN")
            
            # Log detailed LLM response
            logging.info(f"LLM Validation Decision: approved={approved}, confidence={confidence:.2f}, risk={risk_assessment}")
            
            if not approved:
                return False, f"{reason} (confidence: {confidence:.2f}, risk: {risk_assessment})", recommended_target
            else:
                return True, f"{reason} (confidence: {confidence:.2f}, risk: {risk_assessment})", recommended_target
                
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: {e}")
            logging.error(f"Raw response: {content}")
            return False, "LLM response parsing failed", current_replicas
        except Exception as e:
            logging.error(f"Unexpected error parsing LLM response: {e}")
            return False, f"LLM validation error: {str(e)}", current_replicas 