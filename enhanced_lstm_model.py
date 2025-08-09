import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, Concatenate, Bidirectional, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import joblib
from config import *

class EnhancedLSTMModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.ensemble_weights = {}
        self.sequence_length = LSTM_SEQUENCE_LENGTH
        self.model_path = '/workspace/models/enhanced_lstm/'
        self.is_trained = False
        
        # Create models directory
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize different model types
        self.initialize_models()
        
        # Load existing models if available
        self.load_models()
        
    def initialize_models(self):
        """Initialize different types of models for ensemble"""
        try:
            # LSTM Models
            self.models['lstm_basic'] = None
            self.models['lstm_attention'] = None
            self.models['lstm_bidirectional'] = None
            self.models['lstm_ensemble'] = None
            
            # Traditional ML Models
            self.models['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            self.models['svr'] = SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            )
            
            # Initialize scalers
            for model_name in self.models.keys():
                if 'lstm' in model_name:
                    self.scalers[model_name] = MinMaxScaler(feature_range=(0, 1))
                else:
                    self.scalers[model_name] = StandardScaler()
            
            # Initialize ensemble weights
            self.ensemble_weights = {
                'lstm_basic': 0.2,
                'lstm_attention': 0.25,
                'lstm_bidirectional': 0.2,
                'lstm_ensemble': 0.15,
                'random_forest': 0.1,
                'gradient_boosting': 0.05,
                'xgboost': 0.03,
                'lightgbm': 0.01,
                'svr': 0.01
            }
            
            logging.info("Enhanced LSTM models initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        try:
            if data is None or len(data) < self.sequence_length + 1:
                return None, None
            
            # Ensure we have required columns
            if target_column not in data.columns:
                logging.error(f"Target column {target_column} not found in data")
                return None, None
            
            # Create features
            features = self.create_features(data)
            
            # Scale the features
            scaled_features = self.scalers['lstm_basic'].fit_transform(features)
            
            # Create sequences
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_features)):
                X.append(scaled_features[i-self.sequence_length:i])
                y.append(scaled_features[i, 0])  # Target is the first column (close price)
            
            X, y = np.array(X), np.array(y)
            
            return X, y
            
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            return None, None
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        try:
            features = data.copy()
            
            # Price-based features
            features['price_change'] = features['close'].pct_change()
            features['price_change_2'] = features['close'].pct_change(2)
            features['price_change_5'] = features['close'].pct_change(5)
            
            # Volatility features
            features['volatility'] = features['close'].rolling(20).std()
            features['volatility_5'] = features['close'].rolling(5).std()
            
            # Moving averages
            features['sma_5'] = features['close'].rolling(5).mean()
            features['sma_10'] = features['close'].rolling(10).mean()
            features['sma_20'] = features['close'].rolling(20).mean()
            features['ema_12'] = features['close'].ewm(span=12).mean()
            features['ema_26'] = features['close'].ewm(span=26).mean()
            
            # Price position relative to moving averages
            features['price_vs_sma5'] = features['close'] / features['sma_5'] - 1
            features['price_vs_sma20'] = features['close'] / features['sma_20'] - 1
            features['sma5_vs_sma20'] = features['sma_5'] / features['sma_20'] - 1
            
            # RSI
            delta = features['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            features['macd'] = features['ema_12'] - features['ema_26']
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands
            features['bb_middle'] = features['close'].rolling(20).mean()
            bb_std = features['close'].rolling(20).std()
            features['bb_upper'] = features['bb_middle'] + (bb_std * 2)
            features['bb_lower'] = features['bb_middle'] - (bb_std * 2)
            features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # Volume features (if available)
            if 'volume' in features.columns:
                features['volume_sma'] = features['volume'].rolling(20).mean()
                features['volume_ratio'] = features['volume'] / features['volume_sma']
            else:
                features['volume_sma'] = 1000
                features['volume_ratio'] = 1.0
            
            # Time-based features
            if 'timestamp' in features.columns:
                features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
                features['day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
                features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
            else:
                features['hour'] = 12
                features['day_of_week'] = 0
                features['is_weekend'] = 0
            
            # Remove NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logging.error(f"Error creating features: {e}")
            return data
    
    def build_lstm_basic(self, input_shape: Tuple) -> Model:
        """Build basic LSTM model"""
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
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logging.error(f"Error building basic LSTM: {e}")
            return None
    
    def build_lstm_attention(self, input_shape: Tuple) -> Model:
        """Build LSTM model with attention mechanism"""
        try:
            inputs = Input(shape=input_shape)
            
            # LSTM layers
            lstm1 = LSTM(LSTM_UNITS, return_sequences=True)(inputs)
            lstm1 = Dropout(0.2)(lstm1)
            
            lstm2 = LSTM(LSTM_UNITS, return_sequences=True)(lstm1)
            lstm2 = Dropout(0.2)(lstm2)
            
            # Attention mechanism
            attention = Attention()([lstm2, lstm2])
            
            # Global average pooling
            pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
            
            # Dense layers
            dense1 = Dense(25, activation='relu')(pooled)
            dense1 = Dropout(0.2)(dense1)
            
            outputs = Dense(1)(dense1)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logging.error(f"Error building LSTM with attention: {e}")
            return None
    
    def build_lstm_bidirectional(self, input_shape: Tuple) -> Model:
        """Build bidirectional LSTM model"""
        try:
            model = Sequential([
                Bidirectional(LSTM(LSTM_UNITS, return_sequences=True), input_shape=input_shape),
                Dropout(0.2),
                Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
                Dropout(0.2),
                Bidirectional(LSTM(LSTM_UNITS, return_sequences=False)),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logging.error(f"Error building bidirectional LSTM: {e}")
            return None
    
    def build_lstm_ensemble(self, input_shape: Tuple) -> Model:
        """Build ensemble LSTM model with multiple outputs"""
        try:
            inputs = Input(shape=input_shape)
            
            # Multiple LSTM branches
            branch1 = LSTM(LSTM_UNITS, return_sequences=True)(inputs)
            branch1 = LSTM(LSTM_UNITS, return_sequences=False)(branch1)
            
            branch2 = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(inputs)
            branch2 = Bidirectional(LSTM(LSTM_UNITS, return_sequences=False))(branch2)
            
            # Concatenate branches
            concatenated = Concatenate()([branch1, branch2])
            
            # Dense layers
            dense1 = Dense(50, activation='relu')(concatenated)
            dense1 = Dropout(0.3)(dense1)
            
            dense2 = Dense(25, activation='relu')(dense1)
            dense2 = Dropout(0.2)(dense2)
            
            outputs = Dense(1)(dense2)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logging.error(f"Error building ensemble LSTM: {e}")
            return None
    
    def train_models(self, data: pd.DataFrame, target_column: str = 'close'):
        """Train all models in the ensemble"""
        try:
            logging.info("Starting enhanced LSTM model training...")
            
            # Prepare data
            X, y = self.prepare_data(data, target_column)
            if X is None or y is None:
                logging.error("Failed to prepare data for training")
                return False
            
            # Split data
            split_index = int(0.8 * len(X))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=10, min_lr=0.0001)
            ]
            
            # Train LSTM models
            lstm_models = ['lstm_basic', 'lstm_attention', 'lstm_bidirectional', 'lstm_ensemble']
            
            for model_name in lstm_models:
                try:
                    logging.info(f"Training {model_name}...")
                    
                    if model_name == 'lstm_basic':
                        self.models[model_name] = self.build_lstm_basic((X_train.shape[1], X_train.shape[2]))
                    elif model_name == 'lstm_attention':
                        self.models[model_name] = self.build_lstm_attention((X_train.shape[1], X_train.shape[2]))
                    elif model_name == 'lstm_bidirectional':
                        self.models[model_name] = self.build_lstm_bidirectional((X_train.shape[1], X_train.shape[2]))
                    elif model_name == 'lstm_ensemble':
                        self.models[model_name] = self.build_lstm_ensemble((X_train.shape[1], X_train.shape[2]))
                    
                    if self.models[model_name] is not None:
                        history = self.models[model_name].fit(
                            X_train, y_train,
                            batch_size=LSTM_BATCH_SIZE,
                            epochs=LSTM_EPOCHS,
                            validation_data=(X_test, y_test),
                            callbacks=callbacks,
                            verbose=1
                        )
                        
                        # Evaluate model
                        test_loss = self.models[model_name].evaluate(X_test, y_test, verbose=0)
                        logging.info(f"{model_name} test loss: {test_loss[0]:.6f}")
                        
                except Exception as e:
                    logging.error(f"Error training {model_name}: {e}")
                    continue
            
            # Prepare data for traditional ML models
            X_ml = self.prepare_ml_data(data, target_column)
            if X_ml is not None:
                X_ml_train, X_ml_test = X_ml[:split_index], X_ml[split_index:]
                y_ml_train, y_ml_test = y[:split_index], y[split_index:]
                
                # Train traditional ML models
                ml_models = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'svr']
                
                for model_name in ml_models:
                    try:
                        logging.info(f"Training {model_name}...")
                        
                        # Scale features for ML models
                        X_ml_train_scaled = self.scalers[model_name].fit_transform(X_ml_train)
                        X_ml_test_scaled = self.scalers[model_name].transform(X_ml_test)
                        
                        # Train model
                        self.models[model_name].fit(X_ml_train_scaled, y_ml_train)
                        
                        # Evaluate model
                        y_pred = self.models[model_name].predict(X_ml_test_scaled)
                        mse = mean_squared_error(y_ml_test, y_pred)
                        logging.info(f"{model_name} test MSE: {mse:.6f}")
                        
                    except Exception as e:
                        logging.error(f"Error training {model_name}: {e}")
                        continue
            
            # Update ensemble weights based on performance
            self.update_ensemble_weights(X_test, y_test, X_ml_test, y_ml_test)
            
            self.is_trained = True
            logging.info("Enhanced LSTM model training completed")
            
            # Save models
            self.save_models()
            
            return True
            
        except Exception as e:
            logging.error(f"Error training models: {e}")
            return False
    
    def prepare_ml_data(self, data: pd.DataFrame, target_column: str = 'close') -> Optional[np.ndarray]:
        """Prepare data for traditional ML models"""
        try:
            features = self.create_features(data)
            
            # Select relevant features for ML models
            ml_features = [
                'price_change', 'price_change_2', 'price_change_5',
                'volatility', 'volatility_5',
                'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',
                'price_vs_sma5', 'price_vs_sma20', 'sma5_vs_sma20',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_position', 'volume_ratio',
                'hour', 'day_of_week', 'is_weekend'
            ]
            
            # Filter available features
            available_features = [f for f in ml_features if f in features.columns]
            
            if len(available_features) < 5:
                logging.warning("Insufficient features for ML models")
                return None
            
            return features[available_features].values
            
        except Exception as e:
            logging.error(f"Error preparing ML data: {e}")
            return None
    
    def update_ensemble_weights(self, X_test: np.ndarray, y_test: np.ndarray, X_ml_test: np.ndarray, y_ml_test: np.ndarray):
        """Update ensemble weights based on model performance"""
        try:
            performance_scores = {}
            
            # Evaluate LSTM models
            lstm_models = ['lstm_basic', 'lstm_attention', 'lstm_bidirectional', 'lstm_ensemble']
            
            for model_name in lstm_models:
                if self.models[model_name] is not None:
                    try:
                        y_pred = self.models[model_name].predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        performance_scores[model_name] = 1 / (1 + mse)  # Higher score for lower MSE
                    except Exception as e:
                        logging.warning(f"Error evaluating {model_name}: {e}")
                        performance_scores[model_name] = 0.1
            
            # Evaluate ML models
            ml_models = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'svr']
            
            for model_name in ml_models:
                if self.models[model_name] is not None and X_ml_test is not None:
                    try:
                        X_ml_test_scaled = self.scalers[model_name].transform(X_ml_test)
                        y_pred = self.models[model_name].predict(X_ml_test_scaled)
                        mse = mean_squared_error(y_ml_test, y_pred)
                        performance_scores[model_name] = 1 / (1 + mse)
                    except Exception as e:
                        logging.warning(f"Error evaluating {model_name}: {e}")
                        performance_scores[model_name] = 0.1
            
            # Normalize weights
            total_score = sum(performance_scores.values())
            if total_score > 0:
                for model_name in performance_scores:
                    self.ensemble_weights[model_name] = performance_scores[model_name] / total_score
            
            logging.info(f"Updated ensemble weights: {self.ensemble_weights}")
            
        except Exception as e:
            logging.error(f"Error updating ensemble weights: {e}")
    
    def predict(self, data: pd.DataFrame, steps_ahead: int = 1) -> Dict:
        """Make ensemble predictions"""
        try:
            if not self.is_trained:
                logging.warning("Models not trained yet")
                return self.get_default_prediction()
            
            # Prepare data
            X, _ = self.prepare_data(data)
            X_ml = self.prepare_ml_data(data)
            
            if X is None:
                return self.get_default_prediction()
            
            predictions = {}
            ensemble_prediction = 0
            total_weight = 0
            
            # LSTM predictions
            lstm_models = ['lstm_basic', 'lstm_attention', 'lstm_bidirectional', 'lstm_ensemble']
            
            for model_name in lstm_models:
                if self.models[model_name] is not None:
                    try:
                        y_pred = self.models[model_name].predict(X)
                        predictions[model_name] = y_pred[-1][0]  # Last prediction
                        
                        weight = self.ensemble_weights.get(model_name, 0.1)
                        ensemble_prediction += predictions[model_name] * weight
                        total_weight += weight
                        
                    except Exception as e:
                        logging.warning(f"Error predicting with {model_name}: {e}")
            
            # ML predictions
            ml_models = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'svr']
            
            if X_ml is not None:
                for model_name in ml_models:
                    if self.models[model_name] is not None:
                        try:
                            X_ml_scaled = self.scalers[model_name].transform(X_ml)
                            y_pred = self.models[model_name].predict(X_ml_scaled)
                            predictions[model_name] = y_pred[-1]  # Last prediction
                            
                            weight = self.ensemble_weights.get(model_name, 0.01)
                            ensemble_prediction += predictions[model_name] * weight
                            total_weight += weight
                            
                        except Exception as e:
                            logging.warning(f"Error predicting with {model_name}: {e}")
            
            # Calculate final ensemble prediction
            if total_weight > 0:
                final_prediction = ensemble_prediction / total_weight
            else:
                final_prediction = 0
            
            # Calculate confidence based on prediction variance
            if len(predictions) > 1:
                pred_values = list(predictions.values())
                confidence = max(0, 1 - np.std(pred_values) / np.mean(pred_values))
            else:
                confidence = 0.5
            
            return {
                'ensemble_prediction': final_prediction,
                'individual_predictions': predictions,
                'confidence': confidence,
                'ensemble_weights': self.ensemble_weights,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            return self.get_default_prediction()
    
    def predict_direction(self, data: pd.DataFrame) -> Dict:
        """Predict price direction with confidence"""
        try:
            prediction_result = self.predict(data)
            
            if prediction_result['ensemble_prediction'] == 0:
                return self.get_default_direction_prediction()
            
            # Get current price
            current_price = data['close'].iloc[-1]
            
            # Calculate predicted price change
            predicted_change = prediction_result['ensemble_prediction'] - current_price
            predicted_change_pct = (predicted_change / current_price) * 100
            
            # Determine direction
            if predicted_change_pct > 0.1:  # 0.1% threshold
                direction = 'bullish'
            elif predicted_change_pct < -0.1:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            # Calculate direction confidence
            direction_confidence = min(abs(predicted_change_pct) * 10, 100)
            
            return {
                'direction': direction,
                'confidence': direction_confidence,
                'predicted_change_pct': predicted_change_pct,
                'predicted_price': prediction_result['ensemble_prediction'],
                'current_price': current_price,
                'ensemble_confidence': prediction_result['confidence'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Error predicting direction: {e}")
            return self.get_default_direction_prediction()
    
    def analyze_best_currency_pair(self, available_pairs: List[str], market_data_func) -> Tuple[Optional[str], Dict]:
        """Analyze all pairs and select the best one for trading"""
        try:
            if not available_pairs:
                return None, self.get_default_pair_analysis()
            
            pair_scores = {}
            
            for pair in available_pairs:
                try:
                    # Get market data
                    data = market_data_func(pair)
                    if data is None or len(data) < 100:
                        continue
                    
                    # Get direction prediction
                    direction_result = self.predict_direction(data)
                    
                    # Calculate pair score
                    score = direction_result['confidence'] * direction_result['ensemble_confidence'] / 100
                    
                    # Adjust score based on direction
                    if direction_result['direction'] == 'neutral':
                        score *= 0.5
                    
                    pair_scores[pair] = {
                        'score': score,
                        'direction': direction_result['direction'],
                        'confidence': direction_result['confidence'],
                        'predicted_change_pct': direction_result['predicted_change_pct'],
                        'ensemble_confidence': direction_result['ensemble_confidence']
                    }
                    
                except Exception as e:
                    logging.warning(f"Error analyzing pair {pair}: {e}")
                    continue
            
            if not pair_scores:
                return None, self.get_default_pair_analysis()
            
            # Select best pair
            best_pair = max(pair_scores.keys(), key=lambda x: pair_scores[x]['score'])
            best_analysis = pair_scores[best_pair]
            
            return best_pair, {
                'selected_pair': best_pair,
                'score': best_analysis['score'],
                'direction': best_analysis['direction'],
                'confidence': best_analysis['confidence'],
                'predicted_change_pct': best_analysis['predicted_change_pct'],
                'ensemble_confidence': best_analysis['ensemble_confidence'],
                'all_pairs_analyzed': len(pair_scores),
                'pair_scores': pair_scores,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Error analyzing currency pairs: {e}")
            return None, self.get_default_pair_analysis()
    
    def save_models(self):
        """Save all trained models"""
        try:
            for model_name, model in self.models.items():
                if model is not None:
                    if 'lstm' in model_name:
                        # Save Keras models
                        model.save(f"{self.model_path}{model_name}.h5")
                    else:
                        # Save scikit-learn models
                        joblib.dump(model, f"{self.model_path}{model_name}.pkl")
            
            # Save scalers
            for scaler_name, scaler in self.scalers.items():
                joblib.dump(scaler, f"{self.model_path}{scaler_name}_scaler.pkl")
            
            # Save ensemble weights
            with open(f"{self.model_path}ensemble_weights.pkl", 'wb') as f:
                pickle.dump(self.ensemble_weights, f)
            
            logging.info("Models saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load saved models"""
        try:
            for model_name in self.models.keys():
                if 'lstm' in model_name:
                    # Load Keras models
                    model_path = f"{self.model_path}{model_name}.h5"
                    if os.path.exists(model_path):
                        self.models[model_name] = load_model(model_path)
                else:
                    # Load scikit-learn models
                    model_path = f"{self.model_path}{model_name}.pkl"
                    if os.path.exists(model_path):
                        self.models[model_name] = joblib.load(model_path)
            
            # Load scalers
            for scaler_name in self.scalers.keys():
                scaler_path = f"{self.model_path}{scaler_name}_scaler.pkl"
                if os.path.exists(scaler_path):
                    self.scalers[scaler_name] = joblib.load(scaler_path)
            
            # Load ensemble weights
            weights_path = f"{self.model_path}ensemble_weights.pkl"
            if os.path.exists(weights_path):
                with open(weights_path, 'rb') as f:
                    self.ensemble_weights = pickle.load(f)
            
            self.is_trained = any(model is not None for model in self.models.values())
            logging.info("Models loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading models: {e}")
    
    def get_default_prediction(self) -> Dict:
        """Return default prediction result"""
        return {
            'ensemble_prediction': 0,
            'individual_predictions': {},
            'confidence': 0,
            'ensemble_weights': self.ensemble_weights,
            'timestamp': datetime.now()
        }
    
    def get_default_direction_prediction(self) -> Dict:
        """Return default direction prediction"""
        return {
            'direction': 'neutral',
            'confidence': 0,
            'predicted_change_pct': 0,
            'predicted_price': 0,
            'current_price': 0,
            'ensemble_confidence': 0,
            'timestamp': datetime.now()
        }
    
    def get_default_pair_analysis(self) -> Dict:
        """Return default pair analysis"""
        return {
            'selected_pair': None,
            'score': 0,
            'direction': 'neutral',
            'confidence': 0,
            'predicted_change_pct': 0,
            'ensemble_confidence': 0,
            'all_pairs_analyzed': 0,
            'pair_scores': {},
            'timestamp': datetime.now()
        }