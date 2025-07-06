#!/bin/bash

# Indian Stock Market Predictor - Quick Start Script
echo "🚀 Starting Indian Stock Market Predictor..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please edit .env file and add your Alpha Vantage API key!"
    echo "⚠️  Get your free API key from: https://www.alphavantage.co/support/#api-key"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data models static

# Start the application
echo "🌐 Starting Flask web application..."
echo "📱 Open your browser and go to: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

python app.py
