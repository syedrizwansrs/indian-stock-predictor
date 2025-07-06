"""
Machine Learning Prediction Module for Stock Market Analysis.
Implements various ML models for stock price direction prediction.
"""
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    """Handles machine learning predictions for stock price movements."""
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the StockPredictor.
        
        Args:
            models_dir (str): Directory to save/load models
        """
        self.models_dir = models_dir or Config.MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all ML models with default parameters."""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=Config.RANDOM_STATE,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=Config.RANDOM_STATE,
                verbosity=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=Config.RANDOM_STATE
            ),
            'logistic_regression': LogisticRegression(
                random_state=Config.RANDOM_STATE,
                max_iter=1000
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=Config.RANDOM_STATE
            )
        }
        
        logger.info("Initialized ML models")
    
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'Target',
                    feature_cols: List[str] = None, test_size: float = 0.2) -> Tuple:
        """
        Prepare data for machine learning.
        
        Args:
            data (pd.DataFrame): Processed stock data with features
            target_col (str): Target column name
            feature_cols (List[str]): List of feature columns to use
            test_size (float): Proportion of data for testing
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        logger.info("Preparing data for machine learning...")
        
        # Replace inf/-inf with NaN, then remove rows with NaN values
        clean_data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if clean_data.empty:
            raise ValueError("No clean data available after removing NaN values")
        
        # Define features if not provided
        if feature_cols is None:
            # Exclude OHLCV, target columns, and date-related columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'Target', 
                          'Target_Return', 'Target_Price']
            feature_cols = [col for col in clean_data.columns if col not in exclude_cols]
        
        # Prepare features and target
        X = clean_data[feature_cols]
        y = clean_data[target_col]
        
        # Use time series split to maintain temporal order
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Number of features: {len(feature_cols)}")
        logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Target distribution - Test: {y_test.value_counts().to_dict()}")
        
        # Store feature names for later prediction
        self.last_feature_names = feature_cols
        return X_train, X_test, y_train, y_test, feature_cols
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      scaler_name: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using specified scaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            scaler_name (str): Type of scaler ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple: (scaled_X_train, scaled_X_test)
        """
        if scaler_name == 'standard':
            scaler = StandardScaler()
        elif scaler_name == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif scaler_name == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaler: {scaler_name}, using StandardScaler")
            scaler = StandardScaler()
        
        # Fit on training data and transform both
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Store scaler for later use
        self.scalers[scaler_name] = scaler
        
        logger.info(f"Features scaled using {scaler_name} scaler")
        return X_train_scaled, X_test_scaled
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   use_scaling: bool = True, hyperparameter_tuning: bool = False) -> Any:
        """
        Train a specific model.
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            use_scaling (bool): Whether to scale features
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_name} model...")
        
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        # Scale features if needed
        if use_scaling and model_name in ['logistic_regression', 'svm']:
            X_train_processed, _ = self.scale_features(X_train, X_train, 'standard')
        else:
            X_train_processed = X_train
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            model = self._tune_hyperparameters(model_name, model, X_train_processed, y_train)
        
        # Train the model
        model.fit(X_train_processed, y_train)
        
        # Store feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_name] = pd.Series(
                model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
        
        logger.info(f"{model_name} model trained successfully")
        return model
    
    def _tune_hyperparameters(self, model_name: str, model: Any, 
                            X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_name (str): Name of the model
            model (Any): Model instance
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Best model from grid search
        """
        logger.info(f"Tuning hyperparameters for {model_name}...")
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return model
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            model, param_grids[model_name],
            cv=tscv, scoring='accuracy',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model (Any): Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model for logging
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Store performance
        if model_name:
            self.model_performance[model_name] = metrics
        
        # Log results
        logger.info(f"Model evaluation results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def train_all_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train and evaluate all models.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            y_train (pd.Series): Training target
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        logger.info("Training all models...")
        
        trained_models = {}
        
        for model_name in self.models.keys():
            try:
                # Train model
                model = self.train_model(model_name, X_train, y_train)
                
                # Prepare test data (scale if needed)
                if model_name in ['logistic_regression', 'svm']:
                    _, X_test_processed = self.scale_features(X_train, X_test, 'standard')
                else:
                    X_test_processed = X_test
                
                # Evaluate model
                metrics = self.evaluate_model(model, X_test_processed, y_test, model_name)
                
                trained_models[model_name] = model
                
                logger.info(f"✓ {model_name} - Accuracy: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        return trained_models
    
    def create_lstm_model(self, input_shape: Tuple[int, int], 
                         lstm_units: List[int] = [50, 50],
                         dropout_rate: float = 0.2) -> Sequential:
        """
        Create LSTM model for time series prediction.
        
        Args:
            input_shape (Tuple[int, int]): Shape of input data (timesteps, features)
            lstm_units (List[int]): Number of units in each LSTM layer
            dropout_rate (float): Dropout rate
            
        Returns:
            Sequential: Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            lstm_units[0],
            return_sequences=len(lstm_units) > 1,
            input_shape=input_shape
        ))
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:], 1):
            return_sequences = i < len(lstm_units) - 1
            model.add(LSTM(units, return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Created LSTM model with architecture: {lstm_units}")
        return model
    
    def prepare_lstm_data(self, X: pd.DataFrame, y: pd.Series, 
                         sequence_length: int = 60) -> Tuple:
        """
        Prepare data for LSTM model.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            sequence_length (int): Length of input sequences
            
        Returns:
            Tuple: (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X.iloc[i-sequence_length:i].values)
            y_sequences.append(y.iloc[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_lstm_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series,
                        sequence_length: int = 60, epochs: int = 50) -> Sequential:
        """
        Train LSTM model.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            y_train (pd.Series): Training target
            y_test (pd.Series): Test target
            sequence_length (int): Length of input sequences
            epochs (int): Number of training epochs
            
        Returns:
            Sequential: Trained LSTM model
        """
        logger.info("Training LSTM model...")
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, 'standard')
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_lstm_data(X_train_scaled, y_train, sequence_length)
        X_test_seq, y_test_seq = self.prepare_lstm_data(X_test_scaled, y_test, sequence_length)
        
        # Create model
        model = self.create_lstm_model((sequence_length, X_train.shape[1]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test_seq, y_test_seq),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        y_pred = (model.predict(X_test_seq) > 0.5).astype(int).flatten()
        metrics = {
            'accuracy': accuracy_score(y_test_seq, y_pred),
            'precision': precision_score(y_test_seq, y_pred),
            'recall': recall_score(y_test_seq, y_pred),
            'f1_score': f1_score(y_test_seq, y_pred)
        }
        
        self.model_performance['lstm'] = metrics
        
        logger.info("LSTM model training completed")
        logger.info(f"LSTM Accuracy: {metrics['accuracy']:.4f}")
        
        return model
    
    def save_model(self, model: Any, model_name: str):
        """
        Save trained model to disk.
        
        Args:
            model (Any): Trained model
            model_name (str): Name of the model
        """
        filepath = os.path.join(self.models_dir, f"{model_name}_model.pkl")
        
        if model_name == 'lstm':
            model.save(filepath.replace('.pkl', '.h5'))
        else:
            joblib.dump(model, filepath)
        
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str) -> Any:
        """
        Load trained model from disk.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Any: Loaded model
        """
        if model_name == 'lstm':
            from tensorflow.keras.models import load_model
            filepath = os.path.join(self.models_dir, f"{model_name}_model.h5")
            return load_model(filepath)
        else:
            filepath = os.path.join(self.models_dir, f"{model_name}_model.pkl")
            return joblib.load(filepath)
    
    def predict_next_day(self, model: Any, latest_data: pd.DataFrame, 
                        model_name: str = None) -> Dict[str, Any]:
        """
        Predict next day's price direction.
        
        Args:
            model (Any): Trained model
            latest_data (pd.DataFrame): Latest stock data with features
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        # Get the last row and select only the features used during training
        feature_names = getattr(self, 'last_feature_names', None)
        if feature_names is None:
            # Fallback: try to infer from model or use all columns except known non-features
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'Target', 'Target_Return', 'Target_Price']
            feature_names = [col for col in latest_data.columns if col not in exclude_cols]

        features = latest_data[feature_names].tail(1)

        # Scale features if needed
        if model_name in ['logistic_regression', 'svm'] and 'standard' in self.scalers:
            features_scaled = pd.DataFrame(
                self.scalers['standard'].transform(features),
                columns=features.columns
            )
        else:
            features_scaled = features

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0] if hasattr(model, 'predict_proba') else None

        result = {
            'prediction': int(prediction),
            'direction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': float(max(prediction_proba)) if prediction_proba is not None else None,
            'model': model_name,
            'timestamp': latest_data.index[-1]
        }

        return result
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get comparison of all model performances.
        
        Returns:
            pd.DataFrame: Model comparison table
        """
        if not self.model_performance:
            logger.warning("No model performance data available")
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(self.model_performance).T
        comparison_df = comparison_df.sort_values('accuracy', ascending=False)
        
        return comparison_df
    
    def get_feature_importance_summary(self, top_n: int = 10) -> Dict[str, pd.Series]:
        """
        Get feature importance summary for models that support it.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            Dict[str, pd.Series]: Feature importance for each model
        """
        importance_summary = {}
        
        for model_name, importance in self.feature_importance.items():
            importance_summary[model_name] = importance.head(top_n)
        
        return importance_summary
    
    def train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all models on the given data.
        
        Args:
            data (pd.DataFrame): Stock data with features and target
            
        Returns:
            Dict: Dictionary of trained models
        """
        try:
            logger.info("Starting model training process...")
            
            # Prepare features and target
            X_train, X_test, y_train, y_test, feature_cols = self.prepare_data(data)
            
            if len(X_train) < 30:
                raise ValueError("Insufficient data for training. Need at least 30 samples.")
            
            # Train all models
            trained_models = self.train_all_models(X_train, X_test, y_train, y_test)
            
            logger.info(f"Successfully trained {len(trained_models)} models")
            return trained_models
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    from .data_fetcher import DataFetcher
    from .technical_analysis import TechnicalAnalyzer
    
    # Initialize modules
    fetcher = DataFetcher()
    analyzer = TechnicalAnalyzer()
    predictor = StockPredictor()
    
    # Load and process data
    symbol = "RELIANCE.BSE"
    data = fetcher.load_from_database(symbol)
    
    if data is not None and len(data) > 100:
        # Process data
        processed_data = analyzer.process_stock_data(data)
        
        # Prepare data for ML
        X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data(processed_data)
        
        # Train models
        trained_models = predictor.train_all_models(X_train, X_test, y_train, y_test)
        
        # Show model comparison
        comparison = predictor.get_model_comparison()
        print("\nModel Performance Comparison:")
        print(comparison)
        
        # Show feature importance
        importance = predictor.get_feature_importance_summary()
        if importance:
            print(f"\nTop Features for {list(importance.keys())[0]}:")
            print(list(importance.values())[0])
        
        # Make a prediction
        if trained_models:
            best_model_name = comparison.index[0]
            best_model = trained_models[best_model_name]
            
            prediction = predictor.predict_next_day(best_model, processed_data, best_model_name)
            print(f"\nNext day prediction using {best_model_name}:")
            print(f"Direction: {prediction['direction']}")
            print(f"Confidence: {prediction['confidence']:.2f}" if prediction['confidence'] else "N/A")
    else:
        print(f"Insufficient data for {symbol}. Please run data_fetcher.py first.")
