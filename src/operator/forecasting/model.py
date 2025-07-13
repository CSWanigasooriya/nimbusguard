"""LSTM model for time series forecasting."""
import torch
import torch.nn as nn
import numpy as np
import logging
import os
from typing import Optional, Tuple

from config.settings import load_config
config = load_config()


logger = logging.getLogger(__name__)


class LSTMForecaster(nn.Module):
    """LSTM neural network for multi-variate time series forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 forecast_horizon: int = 10, dropout: float = 0.2):
        super(LSTMForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, input_size * forecast_horizon)
        )
        
        logger.info(f"Created LSTM model: input_size={input_size}, hidden_size={hidden_size}, "
                   f"num_layers={num_layers}, forecast_horizon={forecast_horizon}")
    
    def forward(self, x):
        """Forward pass through the network."""
        # x shape: [batch_size, sequence_length, input_size]
        batch_size = x.size(0)
        
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
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, 
              learning_rate: float = 0.001, batch_size: int = 32) -> bool:
        """Train the LSTM model."""
        try:
            if self.model is None:
                self.create_model(X.shape[2])  # input_size = number of features
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Setup training
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            self.model.train()
            
            for epoch in range(epochs):
                total_loss = 0.0
                num_batches = 0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    predictions = self.model(batch_X)
                    loss = criterion(predictions, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                scheduler.step(avg_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
            
            self.model.eval()
            self.is_trained = True
            self.save_model()
            
            logger.info(f"Training completed. Final loss: {avg_loss:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
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