"""
Visualization Module for Stock Market Data.
Creates interactive charts using Plotly for technical analysis visualization.
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import mplfinance as mpf
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockVisualizer:
    """Handles creation of stock market visualizations."""
    
    def __init__(self):
        """Initialize the StockVisualizer."""
        self.color_scheme = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'volume': '#1f77b4',
            'sma': '#ff7f0e',
            'ema': '#2ca02c',
            'bb_upper': '#d62728',
            'bb_lower': '#d62728',
            'bb_middle': '#9467bd',
            'rsi': '#8c564b',
            'macd': '#e377c2',
            'signal': '#7f7f7f'
        }
    
    def create_candlestick_chart(self, data: pd.DataFrame, symbol: str, 
                               include_volume: bool = True,
                               include_sma: bool = True,
                               include_ema: bool = True,
                               include_bollinger: bool = True) -> go.Figure:
        """
        Create an interactive candlestick chart with technical indicators.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV and indicators
            symbol (str): Stock symbol for title
            include_volume (bool): Whether to include volume subplot
            include_sma (bool): Whether to include SMA overlays
            include_ema (bool): Whether to include EMA overlays
            include_bollinger (bool): Whether to include Bollinger Bands
            
        Returns:
            go.Figure: Plotly figure object
        """
        logger.info(f"Creating candlestick chart for {symbol}")
        
        # Determine subplot configuration
        subplot_titles = ['Price']
        rows = 1
        
        if include_volume:
            subplot_titles.append('Volume')
            rows += 1
        
        # Create subplots
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=subplot_titles,
            row_width=[0.7, 0.3] if include_volume else [1.0]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color=self.color_scheme['bullish'],
                decreasing_line_color=self.color_scheme['bearish']
            ),
            row=1, col=1
        )
        
        # Add Simple Moving Averages
        if include_sma:
            for period in [50, 200]:
                sma_col = f'SMA_{period}'
                if sma_col in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[sma_col],
                            mode='lines',
                            name=f'SMA {period}',
                            line=dict(color=self.color_scheme['sma'], width=1),
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
        
        # Add Exponential Moving Averages
        if include_ema:
            for period in [12, 26]:
                ema_col = f'EMA_{period}'
                if ema_col in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[ema_col],
                            mode='lines',
                            name=f'EMA {period}',
                            line=dict(color=self.color_scheme['ema'], width=1),
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
        
        # Add Bollinger Bands
        if include_bollinger and all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            # Upper band
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color=self.color_scheme['bb_upper'], width=1, dash='dash'),
                    opacity=0.5
                ),
                row=1, col=1
            )
            
            # Lower band
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color=self.color_scheme['bb_lower'], width=1, dash='dash'),
                    opacity=0.5,
                    fill='tonexty',
                    fillcolor='rgba(214, 39, 40, 0.1)'
                ),
                row=1, col=1
            )
            
            # Middle band
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color=self.color_scheme['bb_middle'], width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Add volume subplot
        if include_volume:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(data['close'], data['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - Stock Price Analysis',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            height=800 if include_volume else 600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Update y-axis labels
        if include_volume:
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def create_technical_indicators_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create a multi-subplot chart with technical indicators.
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            symbol (str): Stock symbol for title
            
        Returns:
            go.Figure: Plotly figure object
        """
        logger.info(f"Creating technical indicators chart for {symbol}")
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=['Price', 'RSI', 'MACD', 'Volume & OBV'],
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price chart (simplified)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color=self.color_scheme['rsi'], width=2)
                ),
                row=2, col=1
            )
            
            # Add RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
        
        # MACD
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color=self.color_scheme['macd'], width=2)
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color=self.color_scheme['signal'], width=1)
                ),
                row=3, col=1
            )
            
            # MACD Histogram
            if 'MACD_Histogram' in data.columns:
                colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['MACD_Histogram'],
                        name='MACD Histogram',
                        marker_color=colors,
                        opacity=0.6
                    ),
                    row=3, col=1
                )
        
        # Volume and OBV
        if 'volume' in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    marker_color=self.color_scheme['volume'],
                    opacity=0.6,
                    yaxis='y4'
                ),
                row=4, col=1
            )
        
        if 'OBV' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['OBV'],
                    mode='lines',
                    name='OBV',
                    line=dict(color='orange', width=2),
                    yaxis='y5'
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - Technical Indicators Analysis',
            template='plotly_white',
            height=900,
            showlegend=True,
            xaxis_title='Date'
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="Volume/OBV", row=4, col=1)
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame, features: List[str] = None) -> go.Figure:
        """
        Create a correlation heatmap of technical indicators.
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            features (List[str]): List of features to include in correlation
            
        Returns:
            go.Figure: Plotly figure object
        """
        logger.info("Creating correlation heatmap")
        
        if features is None:
            # Select numeric columns (indicators)
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            features = [col for col in numeric_cols if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Calculate correlation matrix
        corr_matrix = data[features].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Technical Indicators Correlation Matrix',
            template='plotly_white',
            width=800,
            height=800
        )
        
        return fig
    
    def create_returns_distribution(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create a distribution plot of daily returns.
        
        Args:
            data (pd.DataFrame): Stock data with returns
            symbol (str): Stock symbol for title
            
        Returns:
            go.Figure: Plotly figure object
        """
        logger.info(f"Creating returns distribution for {symbol}")
        
        if 'Returns' not in data.columns:
            data['Returns'] = data['close'].pct_change()
        
        returns = data['Returns'].dropna()
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Daily Returns',
            opacity=0.7,
            marker_color='lightblue'
        ))
        
        # Add normal distribution overlay
        import scipy.stats as stats
        x = np.linspace(returns.min(), returns.max(), 100)
        y = stats.norm.pdf(x, returns.mean(), returns.std())
        y = y * len(returns) * (returns.max() - returns.min()) / 50  # Scale to match histogram
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f'{symbol} - Daily Returns Distribution',
            xaxis_title='Daily Returns',
            yaxis_title='Frequency',
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def create_volatility_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create a volatility analysis chart.
        
        Args:
            data (pd.DataFrame): Stock data
            symbol (str): Stock symbol for title
            
        Returns:
            go.Figure: Plotly figure object
        """
        logger.info(f"Creating volatility chart for {symbol}")
        
        # Calculate rolling volatility
        data['Returns'] = data['close'].pct_change()
        data['Volatility_20'] = data['Returns'].rolling(20).std() * np.sqrt(252)  # Annualized
        data['Volatility_60'] = data['Returns'].rolling(60).std() * np.sqrt(252)  # Annualized
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=['Price', 'Rolling Volatility (Annualized)'],
            row_heights=[0.6, 0.4]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Volatility chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Volatility_20'],
                mode='lines',
                name='20-day Volatility',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Volatility_60'],
                mode='lines',
                name='60-day Volatility',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{symbol} - Price and Volatility Analysis',
            template='plotly_white',
            height=700,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig
    
    def create_prediction_chart(self, data: pd.DataFrame, predictions: pd.Series, 
                              symbol: str, model_name: str) -> go.Figure:
        """
        Create a chart showing actual vs predicted values.
        
        Args:
            data (pd.DataFrame): Stock data
            predictions (pd.Series): Model predictions
            symbol (str): Stock symbol
            model_name (str): Name of the ML model
            
        Returns:
            go.Figure: Plotly figure object
        """
        logger.info(f"Creating prediction chart for {symbol} using {model_name}")
        
        # Align data and predictions
        aligned_data = data.loc[predictions.index]
        
        # Create signals based on predictions
        buy_signals = aligned_data[predictions == 1]
        sell_signals = aligned_data[predictions == 0]
        
        fig = go.Figure()
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=aligned_data.index,
                y=aligned_data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            )
        )
        
        # Buy signals
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='green'
                    )
                )
            )
        
        # Sell signals
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='red'
                    )
                )
            )
        
        fig.update_layout(
            title=f'{symbol} - {model_name} Predictions',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def save_chart(self, fig: go.Figure, filename: str, format: str = 'html'):
        """
        Save chart to file.
        
        Args:
            fig (go.Figure): Plotly figure to save
            filename (str): Output filename
            format (str): Output format ('html', 'png', 'pdf')
        """
        if format == 'html':
            fig.write_html(filename)
        elif format == 'png':
            fig.write_image(filename)
        elif format == 'pdf':
            fig.write_image(filename)
        
        logger.info(f"Chart saved as {filename}")

# Example usage
if __name__ == "__main__":
    from .data_fetcher import DataFetcher
    from .technical_analysis import TechnicalAnalyzer
    
    # Initialize modules
    fetcher = DataFetcher()
    analyzer = TechnicalAnalyzer()
    visualizer = StockVisualizer()
    
    # Fetch and process sample data
    symbol = "RELIANCE.BSE"
    data = fetcher.load_from_database(symbol)
    
    if data is not None:
        # Process data with technical indicators
        processed_data = analyzer.calculate_all_indicators(data)
        
        # Create various charts
        candlestick_fig = visualizer.create_candlestick_chart(processed_data, symbol)
        indicators_fig = visualizer.create_technical_indicators_chart(processed_data, symbol)
        correlation_fig = visualizer.create_correlation_heatmap(processed_data)
        returns_fig = visualizer.create_returns_distribution(processed_data, symbol)
        volatility_fig = visualizer.create_volatility_chart(processed_data, symbol)
        
        # Show charts (if running in Jupyter)
        # candlestick_fig.show()
        # indicators_fig.show()
        
        print("Charts created successfully!")
        print(f"Data shape: {processed_data.shape}")
    else:
        print(f"No data available for {symbol}. Please run data_fetcher.py first.")
