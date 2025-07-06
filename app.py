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
        
        # Add default exchange if not specified
        if '.' not in symbol:
            symbol += '.BSE'
        
        logger.info(f"Fetching data for {symbol}")
        
        # Try to load existing data first, then fetch if needed
        data = data_fetcher.update_stock_data(symbol)
        
        if data is None or data.empty:
            flash(f'Could not fetch data for {symbol}', 'error')
            return redirect(url_for('index'))
        
        # Process data with technical indicators
        processed_data = technical_analyzer.process_stock_data(data)
        
        # Store in global variable
        current_data[symbol] = processed_data
        
        flash(f'Successfully loaded {len(data)} records for {symbol}', 'success')
        return redirect(url_for('analysis', symbol=symbol))
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        flash(f'Error fetching data: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/analysis/<symbol>')
def analysis(symbol):
    """Analysis page for a specific stock."""
    try:
        if symbol not in current_data:
            flash(f'No data available for {symbol}. Please fetch data first.', 'warning')
            return redirect(url_for('index'))
        
        data = current_data[symbol]
        
        # Create charts
        candlestick_chart = visualizer.create_candlestick_chart(data, symbol)
        indicators_chart = visualizer.create_technical_indicators_chart(data, symbol)
        correlation_chart = visualizer.create_correlation_heatmap(data)
        returns_chart = visualizer.create_returns_distribution(data, symbol)
        volatility_chart = visualizer.create_volatility_chart(data, symbol)
        
        # Convert charts to JSON
        charts = {
            'candlestick': json.dumps(candlestick_chart, cls=plotly.utils.PlotlyJSONEncoder),
            'indicators': json.dumps(indicators_chart, cls=plotly.utils.PlotlyJSONEncoder),
            'correlation': json.dumps(correlation_chart, cls=plotly.utils.PlotlyJSONEncoder),
            'returns': json.dumps(returns_chart, cls=plotly.utils.PlotlyJSONEncoder),
            'volatility': json.dumps(volatility_chart, cls=plotly.utils.PlotlyJSONEncoder)
        }
        
        # Get latest data summary
        latest_data = data.tail(1).iloc[0]
        summary = {
            'symbol': symbol,
            'date': data.index[-1].strftime('%Y-%m-%d'),
            'close': f"₹{latest_data['close']:.2f}",
            'change': f"{latest_data.get('Returns', 0) * 100:.2f}%",
            'volume': f"{latest_data['volume']:,.0f}",
            'rsi': f"{latest_data.get('RSI', 0):.2f}" if 'RSI' in data.columns else 'N/A',
            'total_records': len(data)
        }
        
        return render_template('analysis.html', 
                             symbol=symbol, 
                             charts=charts, 
                             summary=summary)
        
    except Exception as e:
        logger.error(f"Error in analysis page: {str(e)}")
        flash(f'Error loading analysis: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/train_models', methods=['POST'])
def train_models():
    """Train ML models for prediction."""
    try:
        symbol = request.form.get('symbol')
        if not symbol or symbol not in current_data:
            return jsonify({'error': 'No data available for training'})
        
        data = current_data[symbol]
        
        # Prepare data for ML
        X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data(data)
        
        # Train models
        models = predictor.train_all_models(X_train, X_test, y_train, y_test)
        trained_models[symbol] = models
        
        # Get model performance comparison
        comparison = predictor.get_model_comparison()
        
        # Get feature importance
        importance = predictor.get_feature_importance_summary(top_n=10)
        
        return jsonify({
            'success': True,
            'models_trained': len(models),
            'performance': comparison.to_dict(),
            'feature_importance': {k: v.to_dict() for k, v in importance.items()},
            'best_model': comparison.index[0] if not comparison.empty else None
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
        
        if not symbol or symbol not in current_data:
            return jsonify({'error': 'No data available'})
        
        if symbol not in trained_models or not trained_models[symbol]:
            return jsonify({'error': 'No trained models available. Please train models first.'})
        
        if model_name not in trained_models[symbol]:
            return jsonify({'error': f'Model {model_name} not available'})
        
        data = current_data[symbol]
        model = trained_models[symbol][model_name]
        
        # Make prediction
        prediction_result = predictor.predict_next_day(model, data, model_name)

        # Save prediction to database
        from src.data_fetcher import DataFetcher
        data_fetcher = DataFetcher()
        data_fetcher.save_prediction(
            symbol=symbol,
            date=str(prediction_result['timestamp'].date()),
            model=model_name,
            direction=prediction_result['direction'],
            confidence=prediction_result['confidence']
        )

        # Create prediction chart
        predictions_series = pd.Series([prediction_result['prediction']], 
                                     index=[data.index[-1]])
        prediction_chart = visualizer.create_prediction_chart(
            data.tail(30), predictions_series, symbol, model_name
        )

        return jsonify({
            'success': True,
            'prediction': prediction_result,
            'chart': json.dumps(prediction_chart, cls=plotly.utils.PlotlyJSONEncoder)
        })
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)})

# Route to fetch past predictions for a symbol (must be at top-level, not inside a function)
@app.route('/past_predictions/<symbol>')
def past_predictions(symbol):
    from src.data_fetcher import DataFetcher
    data_fetcher = DataFetcher()
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
            }
        }
        
        # Add technical indicators if available
        if 'RSI' in data.columns:
            info['rsi'] = float(latest['RSI'])
        if 'SMA_50' in data.columns:
            info['sma_50'] = float(latest['SMA_50'])
        if 'SMA_200' in data.columns:
            info['sma_200'] = float(latest['SMA_200'])
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/bulk_fetch', methods=['POST'])
def bulk_fetch():
    """Fetch data for multiple stocks."""
    try:
        symbols = request.form.getlist('symbols')
        if not symbols:
            symbols = Config.DEFAULT_STOCKS[:5]  # Limit to 5 for demo
        
        logger.info(f"Bulk fetching data for {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            try:
                data = data_fetcher.update_stock_data(symbol)
                if data is not None and not data.empty:
                    processed_data = technical_analyzer.process_stock_data(data)
                    current_data[symbol] = processed_data
                    results[symbol] = {'status': 'success', 'records': len(data)}
                else:
                    results[symbol] = {'status': 'failed', 'error': 'No data available'}
            except Exception as e:
                results[symbol] = {'status': 'failed', 'error': str(e)}
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len([r for r in results.values() if r['status'] == 'success'])
        })
        
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
            stocks_info.append({
                'symbol': symbol,
                'latest_close': latest['close'],
                'change_pct': latest.get('Returns', 0) * 100,
                'volume': latest['volume'],
                'rsi': latest.get('RSI', None),
                'records': len(data),
                'last_updated': data.index[-1].strftime('%Y-%m-%d')
            })
        
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
