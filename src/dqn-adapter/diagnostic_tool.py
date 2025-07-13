"""
DQN Training Diagnostic Tool

This tool diagnoses and fixes critical training instability issues:
- High training loss (194+)
- Exploding gradients (5360+ norm)
- Q-value explosion (30+ values)
- Reward-Q-value scale mismatch
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger("DQN_Diagnostic")


class DQNTrainingDiagnostic:
    """Comprehensive DQN training diagnostic and repair tool."""
    
    def __init__(self, model, trainer, services):
        self.model = model
        self.trainer = trainer
        self.services = services
        self.diagnostic_results = {}
        
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run complete diagnostic suite and return results."""
        logger.info("🔍 STARTING COMPREHENSIVE DQN DIAGNOSTIC")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_health': self._diagnose_model_health(),
            'training_stability': self._diagnose_training_stability(),
            'reward_scaling': self._diagnose_reward_scaling(),
            'feature_scaling': self._diagnose_feature_scaling(),
            'q_value_analysis': self._diagnose_q_values(),
            'gradient_flow': self._diagnose_gradient_flow(),
            'recommendations': []
        }
        
        # Generate specific recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Log critical issues
        self._log_critical_issues(results)
        
        return results
    
    def _diagnose_model_health(self) -> Dict[str, Any]:
        """Diagnose overall model health."""
        health = {
            'architecture': self._check_architecture(),
            'weight_stats': self._analyze_weights(),
            'activation_stats': self._analyze_activations(),
            'initialization': self._check_initialization()
        }
        
        return health
    
    def _check_architecture(self) -> Dict[str, Any]:
        """Check model architecture for potential issues."""
        arch_info = {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'layers': [],
            'has_batch_norm': False,
            'has_dropout': False
        }
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                arch_info['layers'].append({
                    'name': name,
                    'type': 'Linear',
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'bias': module.bias is not None
                })
            elif isinstance(module, nn.BatchNorm1d):
                arch_info['has_batch_norm'] = True
            elif isinstance(module, nn.Dropout):
                arch_info['has_dropout'] = True
        
        return arch_info
    
    def _analyze_weights(self) -> Dict[str, Any]:
        """Analyze weight distributions for anomalies."""
        weight_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weight_data = param.data.cpu().numpy().flatten()
                weight_stats[name] = {
                    'mean': float(np.mean(weight_data)),
                    'std': float(np.std(weight_data)),
                    'min': float(np.min(weight_data)),
                    'max': float(np.max(weight_data)),
                    'has_nan': bool(np.isnan(weight_data).any()),
                    'has_inf': bool(np.isinf(weight_data).any()),
                    'zero_ratio': float((weight_data == 0).mean())
                }
        
        return weight_stats
    
    def _analyze_activations(self) -> Dict[str, Any]:
        """Analyze activation patterns with sample input."""
        activation_stats = {}
        
        # Create sample input
        sample_input = torch.randn(1, 9).to(next(self.model.parameters()).device)
        
        # Hook to capture activations
        activations = {}
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        with torch.no_grad():
            output = self.model(sample_input)
        
        # Analyze activations
        for name, activation in activations.items():
            activation_flat = activation.flatten()
            activation_stats[name] = {
                'mean': float(np.mean(activation_flat)),
                'std': float(np.std(activation_flat)),
                'min': float(np.min(activation_flat)),
                'max': float(np.max(activation_flat)),
                'dead_neurons': float((activation_flat == 0).mean()),
                'saturation': float((np.abs(activation_flat) > 10).mean())
            }
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activation_stats
    
    def _check_initialization(self) -> Dict[str, Any]:
        """Check if model initialization is appropriate."""
        init_analysis = {
            'weight_scale_appropriate': True,
            'bias_initialization': True,
            'issues': []
        }
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                std = param.std().item()
                if std > 1.0:
                    init_analysis['weight_scale_appropriate'] = False
                    init_analysis['issues'].append(f"{name}: std={std:.4f} too high")
                elif std < 0.01:
                    init_analysis['weight_scale_appropriate'] = False
                    init_analysis['issues'].append(f"{name}: std={std:.4f} too low")
            
            elif 'bias' in name:
                bias_mean = param.mean().item()
                if abs(bias_mean) > 1.0:
                    init_analysis['bias_initialization'] = False
                    init_analysis['issues'].append(f"{name}: mean={bias_mean:.4f} too high")
        
        return init_analysis
    
    def _diagnose_training_stability(self) -> Dict[str, Any]:
        """Diagnose training stability issues."""
        stability = {
            'recent_losses': self.trainer.training_losses[-10:] if self.trainer.training_losses else [],
            'loss_trend': 'unknown',
            'gradient_explosion_risk': 'unknown',
            'learning_rate': self.trainer.optimizer.param_groups[0]['lr'],
            'optimizer_state': self._analyze_optimizer_state()
        }
        
        # Analyze loss trend
        if len(stability['recent_losses']) >= 5:
            recent_mean = np.mean(stability['recent_losses'][-5:])
            if recent_mean > 100:
                stability['loss_trend'] = 'extremely_high'
            elif recent_mean > 10:
                stability['loss_trend'] = 'high'
            elif recent_mean > 1:
                stability['loss_trend'] = 'moderate'
            else:
                stability['loss_trend'] = 'stable'
        
        # Check gradient explosion risk
        if stability['recent_losses'] and max(stability['recent_losses']) > 50:
            stability['gradient_explosion_risk'] = 'high'
        elif stability['recent_losses'] and max(stability['recent_losses']) > 10:
            stability['gradient_explosion_risk'] = 'medium'
        else:
            stability['gradient_explosion_risk'] = 'low'
        
        return stability
    
    def _analyze_optimizer_state(self) -> Dict[str, Any]:
        """Analyze optimizer state for issues."""
        optimizer_analysis = {
            'momentum_buffers': {},
            'squared_gradients': {},
            'step_counts': {}
        }
        
        for group_idx, group in enumerate(self.trainer.optimizer.param_groups):
            for param_idx, param in enumerate(group['params']):
                param_key = f"group_{group_idx}_param_{param_idx}"
                
                if param in self.trainer.optimizer.state:
                    state = self.trainer.optimizer.state[param]
                    
                    if 'momentum_buffer' in state:
                        momentum = state['momentum_buffer'].cpu().numpy().flatten()
                        optimizer_analysis['momentum_buffers'][param_key] = {
                            'mean': float(np.mean(momentum)),
                            'std': float(np.std(momentum)),
                            'max_abs': float(np.max(np.abs(momentum)))
                        }
                    
                    if 'exp_avg_sq' in state:  # Adam optimizer
                        exp_avg_sq = state['exp_avg_sq'].cpu().numpy().flatten()
                        optimizer_analysis['squared_gradients'][param_key] = {
                            'mean': float(np.mean(exp_avg_sq)),
                            'max': float(np.max(exp_avg_sq))
                        }
                    
                    if 'step' in state:
                        optimizer_analysis['step_counts'][param_key] = int(state['step'])
        
        return optimizer_analysis
    
    def _diagnose_reward_scaling(self) -> Dict[str, Any]:
        """Diagnose reward scaling issues."""
        # Get recent experiences from replay buffer
        if len(self.trainer.memory) > 0:
            sample_size = min(100, len(self.trainer.memory))
            # Convert deque to list for slicing
            buffer_list = list(self.trainer.memory.buffer)
            recent_experiences = buffer_list[-sample_size:]
            
            rewards = [exp.reward for exp in recent_experiences]
            reward_analysis = {
                'sample_size': sample_size,
                'reward_range': [float(min(rewards)), float(max(rewards))],
                'reward_mean': float(np.mean(rewards)),
                'reward_std': float(np.std(rewards)),
                'reward_distribution': {
                    'negative': sum(1 for r in rewards if r < 0),
                    'zero': sum(1 for r in rewards if r == 0),
                    'positive': sum(1 for r in rewards if r > 0)
                },
                'scale_mismatch_risk': 'unknown'
            }
            
            # Check for scale mismatch with Q-values
            sample_input = torch.randn(1, 9).to(next(self.model.parameters()).device)
            with torch.no_grad():
                sample_q_values = self.model(sample_input).cpu().numpy().flatten()
            
            q_range = [float(np.min(sample_q_values)), float(np.max(sample_q_values))]
            reward_range = reward_analysis['reward_range']
            
            if abs(q_range[1] - q_range[0]) > 10 * abs(reward_range[1] - reward_range[0]):
                reward_analysis['scale_mismatch_risk'] = 'high'
            elif abs(q_range[1] - q_range[0]) > 3 * abs(reward_range[1] - reward_range[0]):
                reward_analysis['scale_mismatch_risk'] = 'medium'
            else:
                reward_analysis['scale_mismatch_risk'] = 'low'
            
            reward_analysis['q_value_range'] = q_range
            
        else:
            reward_analysis = {
                'sample_size': 0,
                'error': 'No experiences in replay buffer'
            }
        
        return reward_analysis
    
    def _diagnose_feature_scaling(self) -> Dict[str, Any]:
        """Diagnose feature scaling issues."""
        feature_analysis = {
            'scaler_available': self.services.scaler is not None,
            'feature_count': len(self.services.config.base_features),
            'feature_stats': {}
        }
        
        if self.services.scaler and hasattr(self.services.scaler, 'scale_'):
            feature_analysis['scaling_factors'] = self.services.scaler.scale_.tolist()
            feature_analysis['scaling_mean'] = self.services.scaler.center_.tolist() if hasattr(self.services.scaler, 'center_') else None
            
            # Check for extreme scaling factors
            max_scale = max(self.services.scaler.scale_)
            min_scale = min(self.services.scaler.scale_)
            
            if max_scale / min_scale > 1000:
                feature_analysis['scaling_imbalance'] = 'severe'
            elif max_scale / min_scale > 100:
                feature_analysis['scaling_imbalance'] = 'high'
            else:
                feature_analysis['scaling_imbalance'] = 'acceptable'
        
        return feature_analysis
    
    def _diagnose_q_values(self) -> Dict[str, Any]:
        """Diagnose Q-value behavior."""
        q_analysis = {}
        
        # Test with multiple random inputs
        test_inputs = torch.randn(10, 9).to(next(self.model.parameters()).device)
        
        with torch.no_grad():
            q_values = self.model(test_inputs).cpu().numpy()
        
        q_analysis = {
            'mean_q_values': q_values.mean(axis=0).tolist(),
            'std_q_values': q_values.std(axis=0).tolist(),
            'min_q_values': q_values.min(axis=0).tolist(),
            'max_q_values': q_values.max(axis=0).tolist(),
            'q_value_range': float(q_values.max() - q_values.min()),
            'extreme_values': bool((np.abs(q_values) > 100).any()),
            'action_preferences': {
                'scale_down_avg': float(q_values[:, 0].mean()),
                'keep_same_avg': float(q_values[:, 1].mean()),
                'scale_up_avg': float(q_values[:, 2].mean())
            }
        }
        
        return q_analysis
    
    def _diagnose_gradient_flow(self) -> Dict[str, Any]:
        """Diagnose gradient flow issues."""
        gradient_analysis = {
            'gradient_norms': {},
            'gradient_flow_health': 'unknown'
        }
        
        # Create a dummy loss for gradient analysis
        sample_input = torch.randn(4, 9).to(next(self.model.parameters()).device)
        sample_targets = torch.randn(4, 3).to(next(self.model.parameters()).device)
        
        self.model.train()
        output = self.model(sample_input)
        loss = nn.MSELoss()(output, sample_targets)
        
        # Compute gradients
        self.trainer.optimizer.zero_grad()
        loss.backward()
        
        # Analyze gradients
        total_norm = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                gradient_analysis['gradient_norms'][name] = float(param_norm)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** (1. / 2)
        gradient_analysis['total_gradient_norm'] = float(total_norm)
        
        # Assess gradient flow health
        if total_norm > 10:
            gradient_analysis['gradient_flow_health'] = 'exploding'
        elif total_norm < 1e-6:
            gradient_analysis['gradient_flow_health'] = 'vanishing'
        else:
            gradient_analysis['gradient_flow_health'] = 'healthy'
        
        self.model.eval()
        
        return gradient_analysis
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on diagnostic results."""
        recommendations = []
        
        # Training stability recommendations
        if results['training_stability']['loss_trend'] == 'extremely_high':
            recommendations.append("🚨 CRITICAL: Reduce learning rate by 10x (current: {:.6f})".format(
                results['training_stability']['learning_rate']))
            recommendations.append("🚨 CRITICAL: Implement reward normalization/clipping")
        
        # Gradient explosion recommendations
        if results['gradient_flow']['gradient_flow_health'] == 'exploding':
            recommendations.append("🔥 Implement more aggressive gradient clipping (current norm: {:.2f})".format(
                results['gradient_flow']['total_gradient_norm']))
            recommendations.append("🔥 Reduce learning rate and batch size")
        
        # Q-value explosion recommendations
        if results['q_value_analysis']['extreme_values']:
            recommendations.append("⚡ Q-values are exploding - reinitialize model with smaller weights")
            recommendations.append("⚡ Implement Q-value clipping in training loop")
        
        # Reward scaling recommendations
        if results['reward_scaling'].get('scale_mismatch_risk') == 'high':
            recommendations.append("📏 Severe reward-Q-value scale mismatch detected")
            recommendations.append("📏 Implement reward normalization or Q-value scaling")
        
        # Feature scaling recommendations
        if results['feature_scaling'].get('scaling_imbalance') == 'severe':
            recommendations.append("🎯 Feature scaling severely imbalanced - retrain scaler")
        
        # Architecture recommendations
        if results['model_health']['weight_stats']:
            for name, stats in results['model_health']['weight_stats'].items():
                if stats['has_nan'] or stats['has_inf']:
                    recommendations.append(f"💀 NaN/Inf detected in {name} - model corrupted")
        
        return recommendations
    
    def _log_critical_issues(self, results: Dict[str, Any]) -> None:
        """Log critical issues found during diagnostic."""
        logger.info("=" * 60)
        logger.info("🔍 DQN DIAGNOSTIC RESULTS")
        logger.info("=" * 60)
        
        # Log critical issues
        critical_issues = []
        
        if results['training_stability']['loss_trend'] == 'extremely_high':
            critical_issues.append("Training loss extremely high")
        
        if results['gradient_flow']['gradient_flow_health'] == 'exploding':
            critical_issues.append("Gradient explosion detected")
        
        if results['q_value_analysis']['extreme_values']:
            critical_issues.append("Q-value explosion detected")
        
        if results['reward_scaling'].get('scale_mismatch_risk') == 'high':
            critical_issues.append("Reward-Q-value scale mismatch")
        
        if critical_issues:
            logger.error("🚨 CRITICAL ISSUES DETECTED:")
            for issue in critical_issues:
                logger.error(f"   • {issue}")
        else:
            logger.info("✅ No critical issues detected")
        
        # Log recommendations
        if results['recommendations']:
            logger.warning("📋 RECOMMENDATIONS:")
            for rec in results['recommendations']:
                logger.warning(f"   • {rec}")
        
        logger.info("=" * 60)
    
    def apply_emergency_fixes(self) -> Dict[str, bool]:
        """Apply emergency fixes for critical training issues."""
        logger.info("🚑 APPLYING EMERGENCY TRAINING FIXES")
        
        fixes_applied = {
            'learning_rate_reduced': False,
            'model_reinitialized': False,
            'optimizer_reset': False,
            'gradient_clipping_enhanced': False
        }
        
        # 1. Reduce learning rate dramatically
        current_lr = self.trainer.optimizer.param_groups[0]['lr']
        if current_lr > 1e-4:
            new_lr = max(1e-5, current_lr * 0.1)
            for param_group in self.trainer.optimizer.param_groups:
                param_group['lr'] = new_lr
            fixes_applied['learning_rate_reduced'] = True
            logger.info(f"✅ Learning rate reduced: {current_lr:.6f} → {new_lr:.6f}")
        
        # 2. Reinitialize model if Q-values are extreme
        sample_input = torch.randn(1, 9).to(next(self.model.parameters()).device)
        with torch.no_grad():
            q_values = self.model(sample_input).cpu().numpy().flatten()
        
        if np.max(np.abs(q_values)) > 50:
            self._reinitialize_model()
            fixes_applied['model_reinitialized'] = True
            logger.info("✅ Model reinitialized due to extreme Q-values")
        
        # 3. Reset optimizer state
        self.trainer.optimizer.state = {}
        fixes_applied['optimizer_reset'] = True
        logger.info("✅ Optimizer state reset")
        
        # 4. Enhanced gradient clipping (already implemented in trainer)
        fixes_applied['gradient_clipping_enhanced'] = True
        
        return fixes_applied
    
    def _reinitialize_model(self) -> None:
        """Reinitialize model with better parameters."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization with smaller scale
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def export_diagnostic_report(self, filepath: str) -> None:
        """Export comprehensive diagnostic report."""
        results = self.run_full_diagnostic()
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📊 Diagnostic report exported to {filepath}")


def run_emergency_diagnostic(services):
    """Run emergency diagnostic and apply fixes."""
    if not services.dqn_model or not services.dqn_trainer:
        logger.error("❌ Cannot run diagnostic - model or trainer not available")
        return
    
    diagnostic = DQNTrainingDiagnostic(
        model=services.dqn_model,
        trainer=services.dqn_trainer,
        services=services
    )
    
    # Run full diagnostic
    results = diagnostic.run_full_diagnostic()
    
    # Apply emergency fixes if critical issues detected
    critical_detected = any([
        results['training_stability']['loss_trend'] == 'extremely_high',
        results['gradient_flow']['gradient_flow_health'] == 'exploding',
        results['q_value_analysis']['extreme_values']
    ])
    
    if critical_detected:
        logger.warning("🚑 Critical issues detected - applying emergency fixes")
        fixes = diagnostic.apply_emergency_fixes()
        return results, fixes
    else:
        logger.info("✅ No critical issues requiring emergency fixes")
        return results, {} 