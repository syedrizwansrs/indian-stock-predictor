"""
Flask Web Application for Indian Stock Market Predictor.
Provides a user-friendly interface for stock analysis and predictions.
"""
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import json
import plotly
import plotly.utils
from datetime import datetime, timedelta
import logging
import traceback
import os
import yfinance as yf
from src.data_fetcher import DataFetcher
from src.technical_analysis import TechnicalAnalyzer
from src.visualization import StockVisualizer
from src.prediction_model import StockPredictor
from src.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Initialize modules
data_fetcher = DataFetcher()
technical_analyzer = TechnicalAnalyzer()
visualizer = StockVisualizer()
predictor = StockPredictor()

# Global variables to store current data and models
current_data = {}
trained_models = {}

def to_yahoo_symbol(symbol):
    """Convert a stock symbol to Yahoo Finance format."""
    symbol = symbol.strip().upper()
    if symbol.endswith('.BSE'):
        return symbol.replace('.BSE', '.BO')
    elif symbol.endswith('.NSE'):
        return symbol.replace('.NSE', '.NS')
    elif not symbol.endswith(('.NS', '.BO')) and '.' not in symbol:
        return symbol + '.NS'  # Default to NSE
    return symbol

@app.route('/api/predict_15min/<symbol>')
def predict_15min(symbol):
    """Fetch intraday data and predict the next 15-minute close price."""
    try:
        # Validate input symbol
        if not symbol or symbol.strip() == '' or symbol.strip().upper() in ['.NS', '.BO']:
            return jsonify({'error': 'Invalid symbol provided'}), 400
        yf_symbol = to_yahoo_symbol(symbol)
        logger.info(f"Converting symbol {symbol} to Yahoo Finance format: {yf_symbol}")
        # Fetch 15-min interval data for the last 2 days
        df = data_fetcher.fetch_intraday_yfinance(yf_symbol, interval='15m', period='2d')
        if df is None or len(df) < 20:
            return jsonify({'error': 'Not enough intraday data for prediction.'}), 400
        last_close = df['close'].iloc[-1]
        sma_1h = df['close'].iloc[-4:].mean()
        last_time = df.index[-1]
        next_time = last_time + pd.Timedelta(minutes=15)
        return jsonify({
            'symbol': symbol,
            'last_close': last_close,
            'sma_1h': sma_1h,
            'predicted_time': next_time.strftime('%Y-%m-%d %H:%M'),
            'prediction': sma_1h
        })
    except Exception as e:
        logger.error(f"Error in 15-min prediction for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/live_price/<symbol>')
def live_price(symbol):
    """Fetch the latest live price for a given stock symbol using yfinance."""
    try:
        if not symbol or symbol.strip() == '' or symbol.strip().upper() in ['.NS', '.BO']:
            return jsonify({'error': 'Invalid symbol provided'}), 400
        yf_symbol = to_yahoo_symbol(symbol)
        logger.info(f"Fetching live price for symbol {symbol} -> {yf_symbol}")
        ticker = yf.Ticker(yf_symbol)
        price = ticker.fast_info.get('last_price')
        if price is None:
            price = ticker.info.get('regularMarketPrice')
        if price is None:
            return jsonify({'error': 'Price not available'}), 404
        return jsonify({'symbol': yf_symbol, 'price': price})
    except Exception as e:
        logger.error(f"Error fetching live price for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html', default_stocks=Config.DEFAULT_STOCKS)

@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    """Fetch stock data for analysis."""
    try:
        symbol = request.form.get('symbol', '').upper().strip()
        if not symbol:
            flash('Please enter a stock symbol', 'error')
            return redirect(url_for('index'))
        # Normalize to .BSE or .NSE for all backend operations
        if '.' not in symbol:
            canonical_symbol = symbol + '.BSE'
        elif symbol.endswith('.NS'):
            canonical_symbol = symbol.replace('.NS', '.NSE')
        elif symbol.endswith('.BO'):
            canonical_symbol = symbol.replace('.BO', '.BSE')
        else:
            canonical_symbol = symbol
        logger.info(f"Fetching data for {canonical_symbol}")
        data = data_fetcher.update_stock_data(canonical_symbol)
        if data is None or data.empty:
            flash(f'Could not fetch data for {canonical_symbol}', 'error')
            return redirect(url_for('index'))
        processed_data = technical_analyzer.process_stock_data(data)
        current_data[canonical_symbol] = processed_data
        flash(f'Successfully loaded {len(data)} records for {canonical_symbol}', 'success')
        return redirect(url_for('analysis', symbol=canonical_symbol))
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        flash(f'Error fetching data: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/analysis/<symbol>')
def analysis(symbol):
    """Analysis page for a specific stock."""
    try:
        # Normalize symbol to canonical for lookup
        canonical_symbol = symbol.upper()
        if canonical_symbol.endswith('.NS'):
            canonical_symbol = canonical_symbol.replace('.NS', '.NSE')
        elif canonical_symbol.endswith('.BO'):
            canonical_symbol = canonical_symbol.replace('.BO', '.BSE')
        if canonical_symbol not in current_data:
            flash(f'No data available for {symbol}. Please fetch data first.', 'warning')
            return redirect(url_for('index'))
        data = current_data[canonical_symbol]
        
        # Create charts
        candlestick_chart = visualizer.create_candlestick_chart(data, symbol)
        indicators_chart = visualizer.create_technical_indicators_chart(data, symbol)
        correlation_chart = visualizer.create_correlation_heatmap(data)
        returns_chart = visualizer.create_returns_distribution(data, symbol)
        volatility_chart = visualizer.create_volatility_chart(data, symbol)
        
        # Convert charts to JSON
        charts_json = {
            'candlestick': json.dumps(candlestick_chart, cls=plotly.utils.PlotlyJSONEncoder),
            'indicators': json.dumps(indicators_chart, cls=plotly.utils.PlotlyJSONEncoder),
            'correlation': json.dumps(correlation_chart, cls=plotly.utils.PlotlyJSONEncoder),
            'returns': json.dumps(returns_chart, cls=plotly.utils.PlotlyJSONEncoder),
            'volatility': json.dumps(volatility_chart, cls=plotly.utils.PlotlyJSONEncoder)
        }
        
        # Get latest values for display
        latest = data.tail(1).iloc[0]
        # Calculate daily change
        if len(data) > 1:
            prev_close = data['close'].iloc[-2]
            change_val = latest['close'] - prev_close
            change_pct = (change_val / prev_close) * 100
        else:
            change_val = 0
            change_pct = 0
        # Format change as string with sign for template logic
        if change_val > 0:
            change_str = f"+{change_val:.2f}"
        elif change_val < 0:
            change_str = f"{change_val:.2f}"
        else:
            change_str = "0.00"
        stock_info = {
            'symbol': symbol,
            'close': f"{latest['close']:.2f}",
            'volume': f"{int(latest['volume']):,}",
            'latest_date': data.index[-1].strftime('%Y-%m-%d'),
            'total_records': len(data),
            'change': change_str,
            'change_pct': change_pct
        }
        # Add technical indicators if available, round RSI to 2 decimals
        for indicator in ['RSI', 'MACD', 'BB_upper', 'BB_lower', 'SMA_20', 'EMA_12']:
            if indicator in latest:
                if indicator == 'RSI':
                    stock_info['rsi'] = f"{latest['RSI']:.2f}"
                else:
                    stock_info[indicator.lower()] = latest[indicator]
        return render_template('analysis.html', 
                             charts=charts_json, 
                             summary=stock_info)
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        flash(f'Error loading analysis: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/train_models', methods=['POST'])
def train_models():
    """Train machine learning models."""
    try:
        symbol = request.form.get('symbol')
        logger.info(f"[TRAIN] Received train_models request for symbol={symbol}")
        if not symbol or symbol not in current_data:
            logger.warning(f"[TRAIN] No data available for {symbol}")
            return jsonify({'error': 'No data available for the selected symbol'})
        data = current_data[symbol]
        if len(data) < 50:
            logger.warning(f"[TRAIN] Insufficient data for {symbol}: {len(data)} records")
            return jsonify({'error': 'Insufficient data for training. Need at least 50 records.'})
        logger.info(f"[TRAIN] Starting model training for {symbol} with {len(data)} records")
        # Train models
        models = predictor.train_models(data)
        if not models:
            logger.error(f"[TRAIN] Failed to train models for {symbol}")
            return jsonify({'error': 'Failed to train models'})
        # Store trained models
        trained_models[symbol] = models
        logger.info(f"[TRAIN] Trained and stored models for {symbol}: {list(models.keys())}")
        return jsonify({
            'success': True,
            'models_trained': list(models.keys()),
            'data_points': len(data)
        })
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction for next day."""
    try:
        symbol = request.form.get('symbol')
        model_name = request.form.get('model_name', 'random_forest')
        logger.info(f"[PREDICT] Received prediction request for symbol={symbol}, model={model_name}")
        # Normalize to canonical symbol for all lookups
        canonical_symbol = symbol.upper()
        if canonical_symbol.endswith('.NS'):
            canonical_symbol = canonical_symbol.replace('.NS', '.NSE')
        elif canonical_symbol.endswith('.BO'):
            canonical_symbol = canonical_symbol.replace('.BO', '.BSE')
        logger.info(f"[PREDICT] Normalized symbol: {canonical_symbol}")
        if not canonical_symbol or canonical_symbol not in current_data:
            logger.warning(f"[PREDICT] No data available for {canonical_symbol}")
            return jsonify({'error': 'No data available'})
        if canonical_symbol not in trained_models or not trained_models[canonical_symbol]:
            logger.warning(f"[PREDICT] No trained models for {canonical_symbol}")
            return jsonify({'error': 'No trained models available. Please train models first.'})
        if model_name not in trained_models[canonical_symbol]:
            logger.warning(f"[PREDICT] Model {model_name} not available for {canonical_symbol}")
            return jsonify({'error': f'Model {model_name} not available'})
        data = current_data[canonical_symbol]
        model = trained_models[canonical_symbol][model_name]
        symbol = canonical_symbol  # for downstream use
        logger.info(f"[PREDICT] Running prediction for {symbol} using {model_name}")
        # Make prediction
        prediction_result = predictor.predict_next_day(model, data, model_name)
        logger.info(f"[PREDICT] Prediction result: {prediction_result}")
        # Save prediction to database
        data_fetcher.save_prediction(
            symbol=symbol,
            date=str(prediction_result['timestamp'].date()),
            model=model_name,
            direction=prediction_result['direction'],
            confidence=prediction_result['confidence']
        )
        logger.info(f"[PREDICT] Saved prediction for {symbol} on {prediction_result['timestamp'].date()} using {model_name}")
        # Create prediction chart
        predictions_series = pd.Series([prediction_result['prediction']], 
                                     index=[data.index[-1]])
        prediction_chart = visualizer.create_prediction_chart(
            data.tail(30), predictions_series, symbol, model_name
        )
        logger.info(f"[PREDICT] Created prediction chart for {symbol} using {model_name}")
        return jsonify({
            'success': True,
            'prediction': prediction_result,
            'chart': json.dumps(prediction_chart, cls=plotly.utils.PlotlyJSONEncoder)
        })
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/past_predictions/<symbol>')
def past_predictions(symbol):
    """Get past predictions for a symbol."""
    try:
        rows = data_fetcher.get_past_predictions(symbol)
        # Format for JSON
        predictions = [
            {
                'symbol': symbol,
                'date': r[0],
                'model': r[1],
                'direction': r[2],
                'confidence': r[3],
                'created_at': r[4]
            } for r in rows
        ]
        return jsonify({'predictions': predictions})
    except Exception as e:
        logger.error(f"Error fetching past predictions for {symbol}: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/api/stock_info/<symbol>')
def stock_info(symbol):
    """Get basic stock information."""
    try:
        if symbol not in current_data:
            return jsonify({'error': 'Stock not found'})
        
        data = current_data[symbol]
        latest = data.tail(1).iloc[0]
        
        info = {
            'symbol': symbol,
            'latest_date': data.index[-1].isoformat(),
            'latest_close': float(latest['close']),
            'latest_volume': int(latest['volume']),
            'total_records': len(data),
            'data_range': {
                'start': data.index[0].isoformat(),
                'end': data.index[-1].isoformat()
            },
            'has_indicators': 'RSI' in data.columns
        }
        
        # Add technical indicators if available
        for indicator in ['RSI', 'MACD', 'BB_upper', 'BB_lower']:
            if indicator in latest:
                info[indicator.lower()] = float(latest[indicator])
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting stock info: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/bulk_fetch', methods=['POST'])
def bulk_fetch():
    """Fetch data for multiple stocks."""
    try:
        symbols = request.form.get('symbols', '').upper().strip()
        if not symbols:
            return jsonify({'error': 'No symbols provided'})
        
        symbols_list = [s.strip() for s in symbols.split(',') if s.strip()]
        results = {'success': [], 'failed': []}
        
        for symbol in symbols_list:
            try:
                # Normalize to .BSE or .NSE for all backend operations
                if '.' not in symbol:
                    canonical_symbol = symbol + '.BSE'
                elif symbol.endswith('.NS'):
                    canonical_symbol = symbol.replace('.NS', '.NSE')
                elif symbol.endswith('.BO'):
                    canonical_symbol = symbol.replace('.BO', '.BSE')
                else:
                    canonical_symbol = symbol
                logger.info(f"Bulk fetching data for {canonical_symbol}")
                data = data_fetcher.update_stock_data(canonical_symbol)
                if data is not None and not data.empty:
                    processed_data = technical_analyzer.process_stock_data(data)
                    current_data[canonical_symbol] = processed_data
                    results['success'].append(canonical_symbol)
                else:
                    results['failed'].append(canonical_symbol)
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {str(e)}")
                results['failed'].append(symbol)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in bulk fetch: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/dashboard')
def dashboard():
    """Dashboard showing all loaded stocks."""
    try:
        stocks_info = []
        
        for symbol, data in current_data.items():
            latest = data.tail(1).iloc[0]
            stock_info = {
                'symbol': symbol,
                'latest_close': float(latest['close']) if 'close' in latest else None,
                'change_pct': float(latest.get('Returns', 0)) * 100 if 'Returns' in latest else 0,
                'volume': int(latest['volume']) if 'volume' in latest else None,
                'rsi': float(latest['RSI']) if 'RSI' in latest and latest['RSI'] is not None else None,
                'records': len(data),
                'last_updated': data.index[-1].strftime('%Y-%m-%d')
            }
            stocks_info.append(stock_info)
        
        return render_template('dashboard.html', stocks=stocks_info)
        
    except Exception as e:
        logger.error(f"Error in dashboard: {str(e)}")
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {str(error)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
