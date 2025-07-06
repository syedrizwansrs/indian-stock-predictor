# Indian Stock Market Analysis & Prediction System

A comprehensive Python-based application that acquires historical stock data for the Indian market, performs technical analysis, visualizes charts, and attempts to predict future stock price movements using machine learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-Educational-yellow.svg)

## 🚀 Features

### 📊 Data Acquisition
- **Multi-source data fetching**: Alpha Vantage API (primary) with Yahoo Finance fallback
- **Indian market focus**: NSE/BSE stocks including NIFTY 50 and SENSEX 30
- **Local storage**: SQLite database with CSV backup for efficient data management
- **Automatic updates**: Smart data refresh with rate limiting compliance

### 📈 Technical Analysis
- **Comprehensive indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV
- **Advanced features**: Stochastic Oscillator, Williams %R, CCI, Momentum, ROC
- **Feature engineering**: 50+ engineered features for machine learning
- **Time series preparation**: Proper handling of financial time series data

### 📉 Interactive Visualizations
- **Candlestick charts** with volume and technical overlays
- **Multi-panel layouts** for technical indicators (RSI, MACD, OBV)
- **Statistical analysis**: Returns distribution, correlation heatmaps
- **Volatility analysis**: Rolling volatility with multiple timeframes
- **Prediction visualization**: Model signals and confidence intervals

### 🤖 Machine Learning Predictions
- **Multiple algorithms**: Random Forest, XGBoost, LightGBM, Gradient Boosting
- **Deep learning**: LSTM neural networks for time series prediction
- **Traditional ML**: Logistic Regression, SVM with proper feature scaling
- **Model evaluation**: Comprehensive metrics with time series validation
- **Feature importance**: Analysis of most predictive technical indicators

### 🌐 Web Interface
- **Modern responsive design**: Bootstrap-based UI with mobile support
- **Real-time charts**: Interactive Plotly.js visualizations
- **Dashboard view**: Multi-stock portfolio monitoring
- **Bulk operations**: Analyze multiple stocks simultaneously
- **Progress tracking**: Real-time feedback for long-running operations

## 🏗️ Project Structure

```
indian-stock-predictor/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data_fetcher.py         # Data acquisition module
│   ├── technical_analysis.py   # Technical indicators & feature engineering
│   ├── visualization.py       # Chart generation with Plotly
│   └── prediction_model.py     # ML models and predictions
├── templates/
│   ├── base.html              # Base template with navigation
│   ├── index.html             # Home page with stock input
│   ├── analysis.html          # Detailed stock analysis page
│   └── dashboard.html         # Multi-stock dashboard
├── data/                      # Local data storage
├── models/                    # Trained ML models
├── static/                    # Static assets (CSS, JS, images)
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
├── .env.template             # Environment variables template
└── README.md                 # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Step 1: Clone or Download
```bash
# Option 1: Clone with Git
git clone <repository-url>
cd indian-stock-predictor

# Option 2: Download and extract the ZIP file
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configuration
```bash
# Copy environment template
cp .env.template .env

# Edit .env file with your API key
# Get free API key from: https://www.alphavantage.co/support/#api-key
```

### Step 5: Initialize Database
```bash
# The database will be created automatically on first run
# Or manually initialize:
python -c "from src.data_fetcher import DataFetcher; DataFetcher()"
```

## 🚀 Usage

### Option 1: Web Application (Recommended)
```bash
# Start the Flask web server
python app.py

# Open your browser and navigate to:
# http://localhost:5000
```

### Option 2: Command Line Interface
```python
from src.data_fetcher import DataFetcher
from src.technical_analysis import TechnicalAnalyzer
from src.prediction_model import StockPredictor

# Initialize modules
fetcher = DataFetcher()
analyzer = TechnicalAnalyzer()
predictor = StockPredictor()

# Fetch data for a stock
symbol = "RELIANCE.BSE"
data = fetcher.fetch_and_store_data(symbol)

# Analyze with technical indicators
processed_data = analyzer.process_stock_data(data)

# Train ML models and predict
X_train, X_test, y_train, y_test, features = predictor.prepare_data(processed_data)
models = predictor.train_all_models(X_train, X_test, y_train, y_test)

# Make prediction
best_model = models['random_forest']
prediction = predictor.predict_next_day(best_model, processed_data, 'random_forest')
print(f"Next day prediction: {prediction['direction']}")
```

### Option 3: Jupyter Notebook
```bash
# Install Jupyter (if not already installed)
pip install jupyter

# Start Jupyter Notebook
jupyter notebook

# Create a new notebook and use the modules
```

## 📊 Supported Stocks

The application supports Indian stocks from:
- **NSE (National Stock Exchange)**: Use `.NS` suffix (e.g., `RELIANCE.NS`)
- **BSE (Bombay Stock Exchange)**: Use `.BSE` suffix (e.g., `RELIANCE.BSE`)

### Pre-configured NIFTY 50 stocks:
- RELIANCE.BSE, TCS.BSE, HDFCBANK.BSE, INFY.BSE
- HINDUNILVR.BSE, ITC.BSE, SBIN.BSE, BHARTIARTL.BSE
- KOTAKBANK.BSE, LT.BSE, and more...

## 🔧 Configuration Options

### API Configuration
```python
# In src/config.py or .env file
ALPHA_VANTAGE_API_KEY = "your_api_key_here"
API_CALL_DELAY = 12  # seconds between API calls
```

### Technical Analysis Parameters
```python
SMA_PERIODS = [50, 200]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14
BOLLINGER_PERIOD = 20
```

### Machine Learning Parameters
```python
TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_STATE = 42
```

## 📈 Technical Indicators

### Trend Indicators
- **Simple Moving Average (SMA)**: 50 and 200 periods
- **Exponential Moving Average (EMA)**: 12 and 26 periods
- **Moving Average Convergence Divergence (MACD)**

### Momentum Indicators
- **Relative Strength Index (RSI)**: 14 periods
- **Stochastic Oscillator**: %K and %D
- **Williams %R**: Momentum oscillator
- **Rate of Change (ROC)**: Price momentum

### Volatility Indicators
- **Bollinger Bands**: 20 periods with 2 standard deviations
- **Average True Range (ATR)**: 14 periods

### Volume Indicators
- **On-Balance Volume (OBV)**: Volume-price trend
- **Volume ratios**: Various volume-based features

## 🤖 Machine Learning Models

### Traditional Models
1. **Random Forest Classifier**
   - Ensemble of decision trees
   - Good for non-linear relationships
   - Feature importance analysis

2. **XGBoost Classifier**
   - Gradient boosting framework
   - High performance on structured data
   - Advanced regularization

3. **LightGBM Classifier**
   - Fast gradient boosting
   - Memory efficient
   - Good for large datasets

4. **Gradient Boosting Classifier**
   - Scikit-learn implementation
   - Sequential learning
   - Robust to outliers

### Deep Learning
5. **LSTM Neural Network**
   - Long Short-Term Memory networks
   - Specialized for time series
   - Captures long-term dependencies

### Linear Models
6. **Logistic Regression**
   - Linear baseline model
   - Interpretable coefficients
   - Fast training and prediction

7. **Support Vector Machine (SVM)**
   - Non-linear classification
   - Kernel-based approach
   - Good for high-dimensional data

## 📊 Model Evaluation

### Metrics Used
- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

### Validation Strategy
- **Time Series Split**: Respects temporal order
- **No look-ahead bias**: Future data never used for past predictions
- **Rolling window validation**: Multiple train-test splits

## ⚠️ Important Disclaimers

### Investment Risk Warning
```
🚨 IMPORTANT DISCLAIMER 🚨

This application is for EDUCATIONAL and RESEARCH purposes only.
Stock market predictions are inherently uncertain and involve 
significant financial risk. The predictions and analysis provided 
should NOT be considered as financial advice.

Always consult with qualified financial advisors before making 
investment decisions. Past performance does not guarantee future results.
```

### Technical Limitations
- **Market Conditions**: Models trained on historical data may not perform well during unprecedented market conditions
- **Data Quality**: Predictions are only as good as the underlying data quality
- **Overfitting**: Models may overfit to historical patterns that don't persist
- **External Factors**: Cannot account for news, policy changes, or other external events

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Solution: Get a free API key from Alpha Vantage and update .env file
   ```

2. **Import Errors**
   ```bash
   # Make sure you're in the project directory and virtual environment is activated
   pip install -r requirements.txt
   ```

3. **Database Issues**
   ```bash
   # Delete and recreate database
   rm data/stocks.db
   python -c "from src.data_fetcher import DataFetcher; DataFetcher()"
   ```

4. **Memory Issues with LSTM**
   ```python
   # Reduce sequence length or batch size in prediction_model.py
   sequence_length = 30  # instead of 60
   batch_size = 16      # instead of 32
   ```

5. **Rate Limiting**
   ```
   Solution: Increase API_CALL_DELAY in config.py or wait before making more requests
   ```

## 📚 Educational Use Cases

### For Students
- Learn technical analysis concepts
- Understand machine learning in finance
- Practice data visualization
- Study time series analysis

### For Researchers
- Experiment with different indicators
- Test new ML algorithms
- Analyze market patterns
- Compare model performances

### For Developers
- Learn Flask web development
- Practice API integration
- Study database design
- Understand financial data structures

## 🔄 Future Enhancements

### Planned Features
- [ ] Real-time data streaming
- [ ] Portfolio optimization
- [ ] Sentiment analysis from news
- [ ] Options pricing models
- [ ] Risk management tools
- [ ] Mobile app version
- [ ] Cloud deployment guides
- [ ] Automated trading simulation

### Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with:
- API terms of service (Alpha Vantage, Yahoo Finance)
- Local financial regulations
- Academic institution policies

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the configuration
3. Ensure all dependencies are installed
4. Verify API key is valid and has quota remaining

## 🙏 Acknowledgments

- **Alpha Vantage**: For providing free financial data API
- **Yahoo Finance**: For backup data source
- **Plotly**: For excellent interactive charts
- **Scikit-learn**: For machine learning algorithms
- **TensorFlow**: For deep learning capabilities
- **Flask**: For web framework
- **Bootstrap**: For responsive UI components

---

**Happy Trading! 📈** (But remember, this is for education only! 😉)
