"""LSTM model for time series forecasting."""
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from config.settings import load_config

config = load_config()

logger = logging.getLogger(__name__)


class LSTMForecaster(nn.Module):
    """LSTM neural network for consumer workload forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1,
                 forecast_horizon: int = 5, dropout: float = 0.1):
        super(LSTMForecaster, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        # Input normalization layer
        self.input_norm = nn.LayerNorm(input_size)

        # LSTM layers optimized for consumer workload patterns
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Unidirectional for real-time forecasting
        )

        # Attention mechanism for focusing on important features
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Output projection layers with residual connections
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, input_size * forecast_horizon)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Better activation for small models

        # Initialize weights properly
        self._init_weights()

        logger.info(f"Created optimized LSTM model: input_size={input_size}, hidden_size={hidden_size}, "
                    f"num_layers={num_layers}, forecast_horizon={forecast_horizon}")

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

    def forward(self, x):
        """Forward pass through the network."""
        # x shape: [batch_size, sequence_length, input_size]
        batch_size = x.size(0)

        # Input normalization
        x = self.input_norm(x)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: [batch_size, sequence_length, hidden_size]

        # Apply attention to focus on important time steps
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # attn_out shape: [batch_size, sequence_length, hidden_size]

        # Use the last time step for prediction
        last_hidden = attn_out[:, -1, :]  # [batch_size, hidden_size]

        # Output projection with residual connection
        out = self.activation(self.fc1(last_hidden))
        out = self.dropout(out)
        out = self.fc2(out)

        # Reshape to [batch_size, forecast_horizon, input_size]
        out = out.view(batch_size, self.forecast_horizon, self.input_size)

        return out

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last output from LSTM
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]

        # Project to forecast horizon
        output = self.fc(last_output)  # [batch_size, input_size * forecast_horizon]

        # Reshape to [batch_size, forecast_horizon, input_size]
        output = output.view(batch_size, self.forecast_horizon, self.input_size)

        return output

    def init_hidden(self, batch_size: int):
        """Initialize hidden states."""
        device = next(self.parameters()).device
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell


class LSTMTrainer:
    """Handles LSTM model training and inference."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = "/tmp/lstm_forecaster.pth"
        self.is_trained = False

        logger.info(f"Using device: {self.device}")

    def create_model(self, input_size: int) -> None:
        """Create a new LSTM model."""
        self.model = LSTMForecaster(
            input_size=input_size,
            hidden_size=config.forecasting.lstm_hidden_size,
            num_layers=config.forecasting.lstm_num_layers,
            forecast_horizon=config.forecasting.forecast_horizon_minutes
        ).to(self.device)

        logger.info(f"Created new LSTM model with {input_size} input features")

    def load_model(self, input_size: int) -> bool:
        """Load existing model from disk."""
        try:
            if os.path.exists(self.model_path):
                self.create_model(input_size)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                self.is_trained = True
                logger.info("Loaded existing LSTM model")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return False

    def save_model(self) -> None:
        """Save model to disk."""
        if self.model is not None:
            try:
                torch.save(self.model.state_dict(), self.model_path)
                logger.info("Saved LSTM model")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = None,
              learning_rate: float = None, batch_size: int = None) -> bool:
        """Train the LSTM model with improved training procedure."""
        try:
            # Use config defaults if not provided
            if epochs is None:
                epochs = config.forecasting.max_epochs
            if learning_rate is None:
                learning_rate = config.forecasting.learning_rate
            if batch_size is None:
                batch_size = config.forecasting.batch_size

            if self.model is None:
                self.create_model(X.shape[2])  # input_size = number of features

            logger.info(f"🚀 Starting LSTM training: samples={len(X)}, features={X.shape[2]}, "
                        f"epochs={epochs}, lr={learning_rate}, batch_size={batch_size}")

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            # Split into train/validation sets
            val_split = config.forecasting.validation_split
            val_size = int(len(X_tensor) * val_split)
            train_size = len(X_tensor) - val_size

            if val_size > 0:
                train_X, val_X = X_tensor[:train_size], X_tensor[train_size:]
                train_y, val_y = y_tensor[:train_size], y_tensor[train_size:]
            else:
                train_X, val_X = X_tensor, None
                train_y, val_y = y_tensor, None

            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            if val_X is not None:
                val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            else:
                val_loader = None

            # Setup training with improved loss function
            criterion = nn.SmoothL1Loss()  # More robust to outliers than MSE
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=config.forecasting.early_stopping_patience // 2,
                factor=0.5
            )

            # Early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            self.model.train()

            for epoch in range(epochs):
                # Training phase
                total_train_loss = 0.0
                num_batches = 0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()

                    # Forward pass
                    predictions = self.model(batch_X)
                    loss = criterion(predictions, batch_y)

                    # Backward pass with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_train_loss += loss.item()
                    num_batches += 1

                avg_train_loss = total_train_loss / num_batches

                # Validation phase
                avg_val_loss = 0.0
                if val_loader is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        val_batches = 0
                        for batch_X, batch_y in val_loader:
                            predictions = self.model(batch_X)
                            loss = criterion(predictions, batch_y)
                            val_loss += loss.item()
                            val_batches += 1
                        avg_val_loss = val_loss / val_batches if val_batches > 0 else avg_train_loss
                    self.model.train()
                else:
                    avg_val_loss = avg_train_loss

                # Learning rate scheduling
                scheduler.step(avg_val_loss)

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                # Logging
                if epoch % 10 == 0 or epoch == epochs - 1:
                    logger.info(f"📊 Epoch {epoch}/{epochs}: train_loss={avg_train_loss:.6f}, "
                                f"val_loss={avg_val_loss:.6f}, patience={patience_counter}")

                # Early stopping
                if patience_counter >= config.forecasting.early_stopping_patience:
                    logger.info(f"⏹️ Early stopping at epoch {epoch}")
                    break

            # Restore best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            self.model.eval()
            self.is_trained = True
            self.save_model()

            logger.info(f"✅ LSTM training completed: best_val_loss={best_val_loss:.6f}")
            return True

        except Exception as e:
            logger.error(f"❌ LSTM training failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions using the trained model."""
        if self.model is None or not self.is_trained:
            logger.error("Model not trained, cannot make predictions")
            return None

        try:
            self.model.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                predictions = self.model(X_tensor)

                # Convert back to numpy
                predictions_np = predictions.cpu().numpy()

                logger.debug(f"Made prediction with shape: {predictions_np.shape}")
                return predictions_np

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Evaluate model performance."""
        if self.model is None or not self.is_trained:
            logger.error("Model not trained, cannot evaluate")
            return None

        try:
            predictions = self.predict(X)
            if predictions is None:
                return None

            # Calculate MSE
            mse = np.mean((predictions - y) ** 2)
            logger.info(f"Model evaluation MSE: {mse:.6f}")
            return mse

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None

    def get_model_info(self) -> dict:
        """Get model information."""
        if self.model is None:
            return {"status": "not_created"}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "status": "trained" if self.is_trained else "created",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_size": self.model.input_size,
            "hidden_size": self.model.hidden_size,
            "num_layers": self.model.num_layers,
            "forecast_horizon": self.model.forecast_horizon,
            "device": str(self.device)
        }
