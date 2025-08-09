import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import logging
import os
import pickle
from config import *

class LSTMTradingModel:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = LSTM_SEQUENCE_LENGTH
        self.model_path = '/workspace/models/lstm_model.h5'
        self.scaler_path = '/workspace/models/scaler.pkl'
        self.is_trained = False
        
        # Create models directory if it doesn't exist
        os.makedirs('/workspace/models', exist_ok=True)
        
        # Load existing model if available
        self.load_model()
        
    def prepare_data(self, data, target_column='close'):
        """Prepare data for LSTM training"""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data[[target_column]])
            
            # Create sequences
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            return X, y
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            return None, None
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        try:
            model = Sequential([
                LSTM(LSTM_UNITS, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(LSTM_UNITS, return_sequences=True),
                Dropout(0.2),
                LSTM(LSTM_UNITS, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='mean_squared_error',
                         metrics=['mae'])
            
            return model
        except Exception as e:
            logging.error(f"Error building model: {e}")
            return None
    
    def train_model(self, data, target_column='close'):
        """Train the LSTM model"""
        try:
            logging.info("Starting LSTM model training...")
            
            # Prepare data
            X, y = self.prepare_data(data, target_column)
            if X is None or y is None:
                return False
            
            # Split data
            split_index = int(0.8 * len(X))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            # Build model
            self.model = self.build_model((X_train.shape[1], 1))
            if self.model is None:
                return False
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                batch_size=LSTM_BATCH_SIZE,
                epochs=LSTM_EPOCHS,
                validation_data=(X_test, y_test),
                verbose=1,
                shuffle=False
            )
            
            # Save model and scaler
            self.save_model()
            self.is_trained = True
            
            logging.info("LSTM model training completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            return False
    
    def predict(self, data, steps_ahead=1):
        """Make predictions using the trained model"""
        try:
            if not self.is_trained or self.model is None:
                logging.warning("Model not trained or loaded")
                return None
            
            # Prepare input data
            if len(data) < self.sequence_length:
                logging.warning("Not enough data for prediction")
                return None
            
            # Scale the input data
            scaled_data = self.scaler.transform(data[['close']].tail(self.sequence_length))
            
            # Reshape for prediction
            X = np.reshape(scaled_data, (1, self.sequence_length, 1))
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)
            
            # Inverse transform to get actual price
            prediction = self.scaler.inverse_transform(prediction)
            
            return prediction[0][0]
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return None
    
    def predict_direction(self, data):
        """Predict price direction (BUY/SELL)"""
        try:
            current_price = data['close'].iloc[-1]
            predicted_price = self.predict(data)
            
            if predicted_price is None:
                return None, 0
            
            # Calculate direction and confidence
            price_change = (predicted_price - current_price) / current_price
            
            if price_change > 0.001:  # 0.1% threshold
                direction = "BUY"
                confidence = min(95, abs(price_change) * 10000)
            elif price_change < -0.001:
                direction = "SELL"
                confidence = min(95, abs(price_change) * 10000)
            else:
                direction = "HOLD"
                confidence = 50
            
            return direction, confidence
            
        except Exception as e:
            logging.error(f"Error predicting direction: {e}")
            return None, 0
    
    def calculate_accuracy(self, data, predictions):
        """Calculate model accuracy"""
        try:
            # Prepare actual values
            actual_values = self.scaler.transform(data[['close']].values)
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(actual_values[-len(predictions):], predictions))
            
            # Convert RMSE to accuracy percentage
            accuracy = max(0, 100 - (rmse * 100))
            
            return min(accuracy, 99)  # Cap at 99%
            
        except Exception as e:
            logging.error(f"Error calculating accuracy: {e}")
            return 0
    
    def save_model(self):
        """Save the trained model and scaler"""
        try:
            if self.model is not None:
                self.model.save(self.model_path)
                
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            logging.info("Model and scaler saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load existing model and scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = load_model(self.model_path)
                
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.is_trained = True
                logging.info("Model and scaler loaded successfully")
                return True
            else:
                logging.info("No existing model found")
                return False
                
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def retrain_with_new_data(self, new_data):
        """Retrain model with new market data"""
        try:
            logging.info("Retraining model with new data...")
            return self.train_model(new_data)
            
        except Exception as e:
            logging.error(f"Error retraining model: {e}")
            return False
    
    def get_model_confidence(self, data):
        """Calculate AI confidence in the prediction"""
        try:
            if not self.is_trained:
                return 0
            
            # Make multiple predictions with slight variations
            predictions = []
            for i in range(5):
                # Add small noise to test robustness
                noisy_data = data.copy()
                noise = np.random.normal(0, 0.001, len(noisy_data))
                noisy_data['close'] += noise
                
                pred = self.predict(noisy_data)
                if pred is not None:
                    predictions.append(pred)
            
            if len(predictions) < 3:
                return 50
            
            # Calculate variance in predictions
            variance = np.var(predictions)
            
            # Convert variance to confidence (lower variance = higher confidence)
            confidence = max(50, 100 - (variance * 1000))
            
            return min(confidence, 98)
            
        except Exception as e:
            logging.error(f"Error calculating model confidence: {e}")
            return 50