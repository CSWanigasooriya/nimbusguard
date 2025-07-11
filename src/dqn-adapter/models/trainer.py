from io import BytesIO

from config import TrainingExperience
from .network import EnhancedQNetwork
import torch
from data.buffer import ReplayBuffer
import asyncio
import time
import logging
import numpy as np
import torch.nn.functional as F
from monitoring.metrics import DQN_TRAINING_LOSS_GAUGE, DQN_BUFFER_SIZE_GAUGE, DQN_TRAINING_STEPS_GAUGE
logger = logging.getLogger("Trainer")

class DQNTrainer:
    """Combines real-time training with historical data loading."""

    def __init__(self, policy_net, device, feature_order, services=None):
        self.device = device
        self.feature_order = feature_order
        self.services = services  # ServiceContainer for dependencies

        if not services or not services.config:
            raise ValueError("DQNTrainer requires services with config")

        # Networks
        self.policy_net = policy_net
        self.target_net = EnhancedQNetwork(len(feature_order), 3, services.config.dqn.hidden_dims).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Training components
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=services.config.dqn.learning_rate)
        self.memory = ReplayBuffer(services.config.dqn.memory_capacity)
        self.training_queue = asyncio.Queue(maxsize=1000)

        # Training state
        self.batches_trained = 0
        self.last_save_time = time.time()
        self.last_research_time = time.time()
        self.training_losses = []  # Resource limit: only keep last 1000 losses
        self.max_loss_history = 1000

        logger.info(f"TRAINER: initialized device={device}")

        # Try to initialize replay buffer with historical data
        asyncio.create_task(self._load_historical_data())

    def _dict_to_feature_vector(self, state_dict):
        """Convert state dictionary to feature vector using only base features."""
        if not self.services or not self.services.scaler:
            return np.zeros(len(self.feature_order))

        try:
            # Scale the 9 scientifically-selected base features from Prometheus
            raw_features = [state_dict.get(feat, 0.0) for feat in self.services.config.base_features]

            # Transform numpy array to maintain consistency with how scaler was fitted
            raw_features_array = np.array([raw_features])
            scaled_raw_features = self.services.scaler.transform(raw_features_array)

            # Use only the scaled raw features (9 scientifically-selected features)
            final_feature_vector = scaled_raw_features[0].tolist()

            return np.array(final_feature_vector)

        except Exception as e:
            logger.warning(f"Feature scaling failed: {e}")
            # Fallback: Return unscaled features in correct order
            raw_features = [state_dict.get(feat, 0.0) for feat in self.services.config.base_features]
            return np.array(raw_features)

    async def add_experience_for_training(self, experience_dict):
        """Add experience to training queue (non-blocking)."""
        try:
            await self.training_queue.put(experience_dict)
        except Exception as e:
            logger.error(f"Failed to queue experience: {e}")

    async def continuous_training_loop(self):
        """Background training loop that doesn't block decision making."""
        logger.info("TRAINING: continuous_loop_started")

        while True:
            try:
                # Wait for new experience (with timeout to prevent hanging)
                try:
                    experience_dict = await asyncio.wait_for(
                        self.training_queue.get(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Validate and convert experience
                if not self._validate_experience(experience_dict):
                    continue

                # Convert to training format
                state_vec = self._dict_to_feature_vector(experience_dict['state'])
                next_state_vec = self._dict_to_feature_vector(experience_dict['next_state'])

                action_map = {"Scale Down": 0, "Keep Same": 1, "Scale Up": 2}
                action_idx = action_map.get(experience_dict['action'], 1)

                training_exp = TrainingExperience(
                    state=state_vec,
                    action=action_idx,
                    reward=experience_dict['reward'],
                    next_state=next_state_vec,
                    done=False
                )

                # Add to replay buffer
                self.memory.push(training_exp)

                # Train if we have enough experiences
                if len(self.memory) >= self.services.config.dqn.min_batch_size:
                    await self._async_train_step()

                # Periodic saves
                if time.time() - self.last_save_time > self.services.config.save_interval_seconds:
                    await self._save_model()
                    self.last_save_time = time.time()

                # Periodic evaluation
                if (self.services.config.enable_evaluation_outputs and self.services and self.services.evaluator and
                        time.time() - self.last_research_time > self.services.config.evaluation_interval):
                    await self._generate_evaluation_outputs()
                    self.last_research_time = time.time()

            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(5)

    def _validate_experience(self, exp_dict):
        """Validate experience structure."""
        required_keys = ['state', 'action', 'reward', 'next_state']
        return all(key in exp_dict for key in required_keys)

    async def _async_train_step(self):
        """Async training step that doesn't block inference."""
        try:
            # Run training in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            loss = await loop.run_in_executor(None, self._sync_train_step)

            if self.batches_trained % 10 == 0:  # Log every 10 steps
                logger.info(f"TRAINING: batch={self.batches_trained} loss={loss:.4f} buffer_size={len(self.memory)}")

        except Exception as e:
            logger.error(f"Training step error: {e}")

    def _sync_train_step(self):
        """Synchronous training step (runs in thread pool)."""
        try:
            # Progressive batch sizing
            current_batch_size = min(self.services.config.dqn.target_batch_size, max(self.services.config.dqn.min_batch_size, len(self.memory) // 4))
            batch = self.memory.sample(current_batch_size)

            if not batch:
                return 0.0

            # Prepare batch tensors (optimized to avoid slow list-to-tensor conversion)
            states_array = np.array([e.state for e in batch], dtype=np.float32)
            actions_array = np.array([e.action for e in batch], dtype=np.int64)
            rewards_array = np.array([e.reward for e in batch], dtype=np.float32)
            next_states_array = np.array([e.next_state for e in batch], dtype=np.float32)
            dones_array = np.array([e.done for e in batch], dtype=bool)

            states = torch.from_numpy(states_array).to(self.device)
            actions = torch.from_numpy(actions_array).to(self.device)
            rewards = torch.from_numpy(rewards_array).to(self.device)
            next_states = torch.from_numpy(next_states_array).to(self.device)
            dones = torch.from_numpy(dones_array).to(self.device)

            # Current Q values
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Double DQN: use policy net to select actions, target net to evaluate
            with torch.no_grad():
                next_actions = self.policy_net(next_states).max(1)[1]
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q_values = rewards + (self.services.config.dqn.gamma * next_q_values * ~dones)

            # Calculate loss and backpropagate
            loss = F.mse_loss(current_q_values, target_q_values)

            # Validate loss before proceeding
            if not torch.isfinite(loss) or loss.item() < 0:
                logger.warning(f"TRAINER: invalid_loss_detected loss={loss.item()} - skipping training step")
                return 0.0

            # Diagnostic logging for high losses
            if loss.item() > 100:
                q_val_range = f"[{current_q_values.min().item():.2f}, {current_q_values.max().item():.2f}]"
                target_range = f"[{target_q_values.min().item():.2f}, {target_q_values.max().item():.2f}]"
                reward_range = f"[{rewards.min().item():.2f}, {rewards.max().item():.2f}]"
                logger.warning(f"TRAINER: high_loss_analysis loss={loss.item():.4f} "
                             f"q_values={q_val_range} targets={target_range} rewards={reward_range}")
                
                # Check for extreme values that might indicate scaling issues
                if torch.abs(current_q_values).max() > 1000 or torch.abs(target_q_values).max() > 1000:
                    logger.error(f"TRAINER: extreme_q_values_detected - possible_feature_scaling_issue "
                               f"max_q={torch.abs(current_q_values).max().item():.2f} "
                               f"max_target={torch.abs(target_q_values).max().item():.2f}")

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients (more aggressive for high losses)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            
            # Log gradient information for high losses
            if loss.item() > 100:
                logger.warning(f"TRAINER: gradient_analysis grad_norm={grad_norm:.4f} clipped_to=1.0")

            self.optimizer.step()
            self.batches_trained += 1

            # Update target network periodically
            if self.batches_trained % self.services.config.dqn.target_update_interval == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                logger.info(f"TRAINING: target_network_updated batch={self.batches_trained}")

            # Track training metrics with resource limits
            self.training_losses.append(loss.item())
            # Enforce memory limit for training losses
            if len(self.training_losses) > self.max_loss_history:
                self.training_losses = self.training_losses[-self.max_loss_history:]

            # Add training metrics to evaluator with validation
            if self.services and self.services.evaluator:
                try:
                    # Validate all metrics before sending to evaluator
                    loss_value = float(loss.item())
                    # Get epsilon from services (no more global state)
                    epsilon_value = float(self.services.get_epsilon()) if self.services else 0.0
                    batch_size_value = int(current_batch_size) if current_batch_size is not None else 32
                    buffer_size_value = int(len(self.memory))

                    # Additional validation - allow higher losses during training
                    if np.isfinite(loss_value) and 0 <= loss_value <= 10000:  # Increased from 1000 to 10000
                        self.services.evaluator.add_training_metrics(
                            loss=loss_value,
                            epsilon=epsilon_value,
                            batch_size=batch_size_value,
                            buffer_size=buffer_size_value
                        )
                        logger.debug(
                            f"TRAINER: metrics_sent_to_evaluator loss={loss_value:.4f} buffer_size={buffer_size_value}")

                        # Update Prometheus training metrics
                        DQN_TRAINING_LOSS_GAUGE.set(loss_value)
                        DQN_BUFFER_SIZE_GAUGE.set(buffer_size_value)
                        DQN_TRAINING_STEPS_GAUGE.inc()

                        # Log warning for very high losses (but still accept them)
                        if loss_value > 1000:
                            logger.warning(f"TRAINER: high_loss_detected loss={loss_value:.4f} - model_may_need_tuning")

                    else:
                        logger.warning(
                            f"TRAINER: invalid_metrics_not_sent loss={loss_value} finite={np.isfinite(loss_value)} threshold=10000")

                except Exception as eval_error:
                    logger.warning(f"TRAINER: evaluator_metrics_failed error={eval_error}")

            return loss.item()

        except Exception as e:
            logger.error(f"Sync training step error: {e}")
            return 0.0

    async def _save_model(self):
        """Save model to MinIO."""
        logger.info("MODEL_SAVE: attempting_minio_upload")
        try:
            if not self.services or not self.services.minio_client:
                logger.warning("MinIO client not available, skipping model save")
                return

            # Get epsilon from services (no more global state)
            current_epsilon = self.services.get_epsilon() if self.services else 0.3

            # Create comprehensive checkpoint
            checkpoint = {
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'batches_trained': self.batches_trained,
                'epsilon': current_epsilon,
                'feature_order': self.feature_order,
                'model_architecture': 'Simplified DQN (64-32)',
                'state_dim': len(self.feature_order),
                'action_dim': 3,
                'saved_at': time.time()
            }

            # Save to buffer
            buffer = BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)

            self.services.minio_client.put_object(
                self.services.config.minio.bucket_name, "dqn_model.pt", data=buffer, length=buffer.getbuffer().nbytes,
                content_type='application/octet-stream'
            )
            logger.info(f"MODEL_SAVE: success batch={self.batches_trained} size={buffer.getbuffer().nbytes}")

        except Exception as e:
            logger.error(f"MODEL_SAVE: failed error={e}")

    async def _load_historical_data(self):
        """Load historical training data to initialize replay buffer with validation."""
        try:
            if not self.services or not self.services.minio_client:
                logger.info("HISTORICAL_DATA: minio_client_not_available")
                return

            # Skip historical data loading to avoid reward poisoning
            # Historical data has zero rewards which biases learning toward conservative actions
            logger.info("HISTORICAL_DATA: skipping_to_avoid_reward_poisoning")
            logger.info("HISTORICAL_DATA: model_will_learn_from_live_experiences_only")
            return
            
            # DISABLED CODE BELOW - keeping for future reward reconstruction
            logger.info("HISTORICAL_DATA: loading_from_minio")

            # Try to load the dataset from MinIO
            response = self.services.minio_client.get_object(self.services.config.minio.bucket_name, "dqn_features.parquet")
            dataset_bytes = response.read()

            # Load dataset using pandas
            import pandas as pd
            from io import BytesIO
            df = pd.read_parquet(BytesIO(dataset_bytes))

            logger.info(f"HISTORICAL_DATA: dataset_loaded rows={len(df)} columns={len(df.columns)}")

            # Validate dataset structure
            required_columns = set(self.services.config.base_features)
            available_columns = set(df.columns)
            missing_columns = required_columns - available_columns

            if missing_columns:
                logger.warning(f"HISTORICAL_DATA: missing_columns {missing_columns}")

            # Convert to training experiences (sample a subset to avoid overwhelming the buffer)
            max_historical = min(1000, len(df))  # Limit to 1000 historical experiences
            sampled_df = df.sample(n=max_historical, random_state=42) if len(df) > max_historical else df

            action_map = {0: "Scale Down", 1: "Keep Same", 2: "Scale Up"}
            loaded_count = 0
            validation_failures = 0

            for _, row in sampled_df.iterrows():
                try:
                    # Validate row data
                    if row.isnull().sum() > len(row) * 0.5:  # More than 50% null values
                        validation_failures += 1
                        continue

                    # Create state from features (adapt to current feature order)
                    state_dict = {}
                    for feat in self.feature_order:
                        if feat in row and pd.notna(row[feat]):
                            value = float(row[feat])
                            # Validate feature values
                            if not np.isfinite(value):
                                validation_failures += 1
                                continue
                            state_dict[feat] = value
                        else:
                            state_dict[feat] = 0.0  # Default value for missing features

                    # Validate scaling action
                    scaling_action = row.get('scaling_action', 1)
                    if not isinstance(scaling_action, (int, float)) or scaling_action not in [0, 1, 2]:
                        scaling_action = 1  # Default to "Keep Same"

                    # Create a simple next state (assume minimal change)
                    next_state_dict = state_dict.copy()

                    experience_dict = {
                        'state': state_dict,
                        'action': action_map.get(int(scaling_action), "Keep Same"),
                        'reward': 0.0,  # Historical reward unknown, use neutral
                        'next_state': next_state_dict
                    }

                    # Convert and add to replay buffer
                    state_vec = self._dict_to_feature_vector(state_dict)
                    next_state_vec = self._dict_to_feature_vector(next_state_dict)

                    # Validate feature vectors
                    if not (np.isfinite(state_vec).all() and np.isfinite(next_state_vec).all()):
                        validation_failures += 1
                        continue

                    training_exp = TrainingExperience(
                        state=state_vec,
                        action=int(scaling_action),
                        reward=0.0,
                        next_state=next_state_vec,
                        done=False
                    )

                    self.memory.push(training_exp)
                    loaded_count += 1

                except Exception as exp_error:
                    validation_failures += 1
                    logger.debug(f"HISTORICAL_DATA: experience_validation_failed error={exp_error}")
                    continue  # Skip malformed experiences

            logger.info(f"HISTORICAL_DATA: loading_complete "
                        f"loaded_experiences={loaded_count} "
                        f"validation_failures={validation_failures} "
                        f"success_rate={loaded_count / (loaded_count + validation_failures) * 100:.1f}%")

        except Exception as e:
            logger.warning(f"HISTORICAL_DATA: load_failed error={e}")
            # Don't raise - historical data is optional

    async def _generate_evaluation_outputs(self):
        """Generate and save all evaluation outputs."""
        try:
            if not self.services or not self.services.evaluator:
                return

            logger.info("EVALUATION: generating_outputs")

            # Get model state information
            total_params = sum(p.numel() for p in self.policy_net.parameters())
            model_state = {
                'total_params': total_params,
                'batches_trained': self.batches_trained,
                'device': str(self.device),
                'architecture': 'Simplified DQN (64-32)',
                'training_losses': self.training_losses[-100:],  # Last 100 losses
            }

            # Generate all evaluation outputs
            loop = asyncio.get_event_loop()
            saved_files = await loop.run_in_executor(
                None,
                self.services.evaluator.generate_all_outputs,
                model_state
            )

            logger.info(f"EVALUATION: completed files_generated={len(saved_files)}")

        except Exception as e:
            logger.error(f"Failed to generate evaluation outputs: {e}")
