import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class SumLSTMPredictor:
    """
    LSTM that predicts:
    - Total memory sum for next 15s interval
    - Number of pods for next 15s interval
    
    This matches how Kubernetes autoscaling actually works!
    """
    
    def __init__(self, csv_path='container_memory_usage_bytes.csv'):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        self.lookback_window = 20  # 5 minutes of history
        
    def load_and_aggregate_data(self):
        """Load data and create aggregated time series"""
        print("🔄 Loading data with aggregation approach...")
        
        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        print(f"Total rows loaded: {len(self.df)}")
        
        # Convert timestamp
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Filter consumer pods only
        self.consumer_df = self.df[self.df['pod'].str.contains('consumer-', na=False)].copy()
        print(f"Consumer pod rows: {len(self.consumer_df)}")
        
        # Simple aggregation by timestamp - this is the key!
        self.timeseries_df = self.consumer_df.groupby('timestamp').agg({
            'value': 'sum',    # Total memory across all consumer pods
            'pod': 'count'     # Number of consumer pods
        }).reset_index()
        
        # Rename columns for clarity
        self.timeseries_df.columns = ['timestamp', 'total_memory_bytes', 'pod_count']
        
        # Convert to MB for better numerical properties
        self.timeseries_df['total_memory_mb'] = self.timeseries_df['total_memory_bytes'] / (1024 * 1024)
        
        # Sort by timestamp
        self.timeseries_df = self.timeseries_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Created time series with {len(self.timeseries_df)} intervals")
        print(f"Time range: {self.timeseries_df['timestamp'].min()} to {self.timeseries_df['timestamp'].max()}")
        print(f"Memory range: {self.timeseries_df['total_memory_mb'].min():.0f} - {self.timeseries_df['total_memory_mb'].max():.0f} MB")
        print(f"Pod count range: {self.timeseries_df['pod_count'].min()} - {self.timeseries_df['pod_count'].max()}")
        
        return self.timeseries_df
    
    def create_sequences(self):
        """
        Create sequences for prediction - always predicts both memory and pod count
        """
        
        # Always predict both memory and pod count
        features = ['total_memory_mb', 'pod_count']
        targets = ['total_memory_mb', 'pod_count']
        print("🎯 Predicting: Total Memory + Pod Count")
        
        print(f"Features: {features}")
        print(f"Targets: {targets}")
        print(f"Lookback window: {self.lookback_window} intervals ({self.lookback_window * 15} seconds)")
        
        # Prepare data arrays
        feature_data = self.timeseries_df[features].values
        target_data = self.timeseries_df[targets].values
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback_window, len(feature_data)):
            # Use last N intervals as features
            X.append(feature_data[i-self.lookback_window:i])
            # Predict next interval
            y.append(target_data[i])
        
        X, y = np.array(X), np.array(y)
        
        # Temporal split (crucial for time series!)
        split_index = int(0.8 * len(X))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        print(f"\n📊 Dataset:")
        print(f"Training sequences: {X_train.shape[0]}")
        print(f"Test sequences: {X_test.shape[0]}")
        print(f"Sequence length: {X_train.shape[1]} time steps")
        print(f"Features per step: {X_train.shape[2]}")
        print(f"Targets per prediction: {y_train.shape[1]}")
        
        # Scale the data
        # Reshape for scaling
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_2d)
        
        # Transform features
        X_train_scaled = self.scaler.transform(X_train_2d).reshape(X_train.shape)
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # For targets, use a separate scaler for both memory and pod count
        self.target_scaler = StandardScaler()
        y_train_2d = y_train.reshape(-1, y_train.shape[-1])
        self.target_scaler.fit(y_train_2d)
        y_train_scaled = self.target_scaler.transform(y_train_2d).reshape(y_train.shape)
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
        
        # Store datasets
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train_scaled, y_test_scaled
        self.y_train_orig, self.y_test_orig = y_train, y_test
        self.predict_both = True  # Always predict both
        
        return features, targets
    
    def build_model(self, features, targets):
        """Build LSTM model"""
        
        n_features = len(features)
        n_targets = len(targets)
        sequence_length = self.lookback_window
        
        print(f"\n🏗️  Building LSTM model...")
        print(f"Input shape: ({sequence_length}, {n_features})")
        print(f"Output shape: {n_targets}")
        
        self.model = Sequential([
            # LSTM layers
            LSTM(32, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(0.1),
            
            LSTM(16, return_sequences=False),
            Dropout(0.1),
            
            # Dense layers
            Dense(8, activation='relu'),
            Dense(n_targets)  # Output layer
        ])
        
        # Compile
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("Model architecture:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, epochs=100, batch_size=32):
        """Train the model"""
        
        print(f"\n🚀 Training model...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Don't shuffle time series!
        )
        
        print("✅ Training completed!")
    
    def evaluate_performance(self):
        """Evaluate the model"""
        
        # Make predictions
        y_train_pred_scaled = self.model.predict(self.X_train, verbose=0)
        y_test_pred_scaled = self.model.predict(self.X_test, verbose=0)
        
        # Inverse transform predictions
        y_train_pred = self.target_scaler.inverse_transform(
            y_train_pred_scaled.reshape(-1, y_train_pred_scaled.shape[-1])
        ).reshape(y_train_pred_scaled.shape)
        y_test_pred = self.target_scaler.inverse_transform(
            y_test_pred_scaled.reshape(-1, y_test_pred_scaled.shape[-1])
        ).reshape(y_test_pred_scaled.shape)
        
        # Calculate metrics for memory (first column)
        train_mae_memory = mean_absolute_error(self.y_train_orig[:, 0], y_train_pred[:, 0])
        test_mae_memory = mean_absolute_error(self.y_test_orig[:, 0], y_test_pred[:, 0])
        train_mape_memory = np.mean(np.abs((self.y_train_orig[:, 0] - y_train_pred[:, 0]) / self.y_train_orig[:, 0])) * 100
        test_mape_memory = np.mean(np.abs((self.y_test_orig[:, 0] - y_test_pred[:, 0]) / self.y_test_orig[:, 0])) * 100
        
        # Calculate metrics for pod count (second column)
        train_mae_pods = mean_absolute_error(self.y_train_orig[:, 1], y_train_pred[:, 1])
        test_mae_pods = mean_absolute_error(self.y_test_orig[:, 1], y_test_pred[:, 1])
        
        print("\n" + "="*60)
        print("LSTM PERFORMANCE (Memory + Pods)")
        print("="*60)
        print(f"Memory Prediction:")
        print(f"  Training MAE: {train_mae_memory:.1f} MB")
        print(f"  Test MAE: {test_mae_memory:.1f} MB")
        print(f"  Training MAPE: {train_mape_memory:.1f}%")
        print(f"  Test MAPE: {test_mape_memory:.1f}%")
        print(f"Pod Count Prediction:")
        print(f"  Training MAE: {train_mae_pods:.2f} pods")
        print(f"  Test MAE: {test_mae_pods:.2f} pods")
        
        # Store predictions
        self.y_train_pred = y_train_pred
        self.y_test_pred = y_test_pred
        
        return {
            'test_mae_memory': test_mae_memory,
            'test_mape_memory': test_mape_memory,
            'test_mae_pods': test_mae_pods
        }
    
    def plot_results(self, save_plots=True):
        """Plot results for model"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Memory predictions
        axes[0,0].plot(self.y_test_orig[:, 0], label='Actual Memory', alpha=0.8)
        axes[0,0].plot(self.y_test_pred[:, 0], label='Predicted Memory', alpha=0.8)
        axes[0,0].set_title('Memory Prediction (Test Set)')
        axes[0,0].set_ylabel('Memory (MB)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Pod count predictions
        axes[0,1].plot(self.y_test_orig[:, 1], label='Actual Pods', alpha=0.8, marker='o', markersize=2)
        axes[0,1].plot(self.y_test_pred[:, 1], label='Predicted Pods', alpha=0.8, marker='s', markersize=2)
        axes[0,1].set_title('Pod Count Prediction (Test Set)')
        axes[0,1].set_ylabel('Number of Pods')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Memory scatter
        axes[1,0].scatter(self.y_test_orig[:, 0], self.y_test_pred[:, 0], alpha=0.6)
        min_val, max_val = self.y_test_orig[:, 0].min(), self.y_test_orig[:, 0].max()
        axes[1,0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1,0].set_title('Memory: Predicted vs Actual')
        axes[1,0].set_xlabel('Actual (MB)')
        axes[1,0].set_ylabel('Predicted (MB)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Pod scatter
        axes[1,1].scatter(self.y_test_orig[:, 1], self.y_test_pred[:, 1], alpha=0.6)
        min_val, max_val = self.y_test_orig[:, 1].min(), self.y_test_orig[:, 1].max()
        axes[1,1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1,1].set_title('Pod Count: Predicted vs Actual')
        axes[1,1].set_xlabel('Actual (count)')
        axes[1,1].set_ylabel('Predicted (count)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('lstm_results.png', dpi=300, bbox_inches='tight')
            print("LSTM plots saved as 'lstm_results.png'")
        
        plt.show()
    
    def predict_next_interval(self, raw_pod_data=None):
        """
        Predict the next 15-second interval
        
        Args:
            raw_pod_data: Optional. List of dictionaries with format:
                [{'timestamp': str, 'pod_name': str, 'memory_bytes': int}, ...]
                If None, uses test data (for training/evaluation)
                If provided, uses real-time data (for production)
        """
        
        if raw_pod_data is not None:
            # Production mode: Use real-time pod data
            try:
                # Convert raw pod data to the format the model expects
                df = pd.DataFrame(raw_pod_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Aggregate by timestamp (same as training)
                aggregated = df.groupby('timestamp').agg({
                    'memory_bytes': 'sum',    # Total memory across all consumer pods
                    'pod_name': 'count'       # Number of consumer pods
                }).reset_index()
                
                # Rename columns to match training format
                aggregated.columns = ['timestamp', 'total_memory_bytes', 'pod_count']
                
                # Convert to MB (same as training)
                aggregated['total_memory_mb'] = aggregated['total_memory_bytes'] / (1024 * 1024)
                
                # Sort by timestamp
                aggregated = aggregated.sort_values('timestamp').reset_index(drop=True)
                
                # Check if we have enough history
                if len(aggregated) < self.lookback_window:
                    print(f"Not enough data for prediction. Need {self.lookback_window}, got {len(aggregated)}")
                    return None
                
                # Extract features for last N intervals
                features = ['total_memory_mb', 'pod_count']
                recent_data = aggregated[features].tail(self.lookback_window).values
                
                # Scale using training scalers
                scaled_data = self.scaler.transform(recent_data)
                
                # Reshape for LSTM (batch_size=1, sequence_length=N, features=2)
                last_sequence = scaled_data.reshape(1, self.lookback_window, len(features))
                
            except Exception as e:
                print(f"❌ Error processing real-time data: {e}")
                return None
        else:
            # Training/evaluation mode: Use test data
            if not hasattr(self, 'X_test') or self.X_test is None:
                print("❌ No test data available for prediction")
                return None
            last_sequence = self.X_test[-1:].copy()
        
        # Make prediction
        prediction_scaled = self.model.predict(last_sequence, verbose=0)
        
        # Always predict both memory and pod count
        prediction = self.target_scaler.inverse_transform(prediction_scaled)[0]
        print(f"\n🔮 NEXT 15-SECOND PREDICTION:")
        print(f"Predicted total memory: {prediction[0]:.0f} MB")
        print(f"Predicted pod count: {prediction[1]:.0f} pods")
        
        # Return dictionary format for operator compatibility
        if raw_pod_data is not None:
            # For production use, return structured result
            latest_data = pd.DataFrame(raw_pod_data).groupby(pd.to_datetime(pd.DataFrame(raw_pod_data)['timestamp'])).agg({
                'memory_bytes': 'sum', 'pod_name': 'count'
            }).iloc[-1]
            
            return {
                'predicted_memory_mb': round(prediction[0], 1),
                'predicted_pod_count': round(prediction[1], 0),
                'predicted_memory_bytes': int(prediction[0] * 1024 * 1024),
                'current_memory_bytes': int(latest_data['memory_bytes']),
                'memory_change_mb': round(prediction[0] - (latest_data['memory_bytes'] / (1024*1024)), 1),
                'pod_change': round(prediction[1] - latest_data['pod_name'], 0)
            }
        else:
            # For training/evaluation, return simple array
            return prediction
    

    
    def save_model(self, model_path='memory_lstm_model.keras', scaler_path='memory_lstm_scaler.pkl'):
        """Save the trained model and scalers"""
        import joblib
        
        if self.model is None:
            print("❌ No trained model to save. Train the model first.")
            return
        
        try:
            # Save the Keras model
            self.model.save(model_path)
            print(f"✅ Model saved to: {model_path}")
            
            # Save the scalers
            scalers_data = {
                'feature_scaler': self.scaler,
                'target_scaler': self.target_scaler,
                'predict_both': self.predict_both,
                'lookback_window': self.lookback_window
            }
            joblib.dump(scalers_data, scaler_path)
            print(f"✅ Scalers and metadata saved to: {scaler_path}")
            
            print(f"\n📦 Model files ready for deployment:")
            print(f"   • Model: {model_path}")
            print(f"   • Scalers: {scaler_path}")
            print(f"   • Prediction mode: {'Both memory+pods' if self.predict_both else 'Memory only'}")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    @classmethod
    def load_model(cls, model_path='memory_lstm_model.keras', scaler_path='memory_lstm_scaler.pkl'):
        """Load a saved model and scalers"""
        import joblib
        
        try:
            # Create instance
            predictor = cls()
            
            # Load the Keras model
            predictor.model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded from: {model_path}")
            
            # Load the scalers
            scalers_data = joblib.load(scaler_path)
            predictor.scaler = scalers_data['feature_scaler']
            predictor.target_scaler = scalers_data['target_scaler']
            predictor.predict_both = scalers_data['predict_both']
            predictor.lookback_window = scalers_data['lookback_window']
            print(f"✅ Scalers and metadata loaded from: {scaler_path}")
            
            print(f"\n📦 Model loaded successfully:")
            print(f"   • Prediction mode: {'Both memory+pods' if predictor.predict_both else 'Memory only'}")
            print(f"   • Lookback window: {predictor.lookback_window} intervals")
            
            return predictor
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

def main():
    """Main training pipeline for LSTM approach"""
    
    print("🎯 Sum-Based LSTM for Kubernetes Autoscaling")
    print("Focus: Predict total memory sum + pod count for next 15s")
    print("=" * 60)
    
    predictor = SumLSTMPredictor('container_memory_usage_bytes.csv')
    
    try:
        # Load and aggregate data
        df = predictor.load_and_aggregate_data()
        
        # Create sequences (always predicts both memory and pod count)
        features, targets = predictor.create_sequences()
        
        # Build model
        model = predictor.build_model(features, targets)
        
        # Train
        predictor.train_model(epochs=100, batch_size=32)
        
        # Evaluate
        metrics = predictor.evaluate_performance()
        
        # Plot
        predictor.plot_results()
        
        # Make next prediction
        next_pred = predictor.predict_next_interval()
        
        # Save the model
        print("\n" + "="*60)
        print("💾 SAVING MODEL...")
        print("="*60)
        predictor.save_model()
        
        print("\n" + "="*60)
        print("✅ LSTM TRAINING COMPLETED!")
        print("="*60)
        print("🎯 This approach is perfect for Kubernetes autoscaling:")
        print("   • Predicts total memory demand")
        print("   • Predicts pod count")
        print("   • Handles dynamic pod scaling")
        print("   • Fast predictions for real-time decisions")
        print("\n📂 Model files have been saved and are ready for deployment!")
        
        return predictor
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    predictor = main()