@echo off
REM Indian Stock Market Predictor - Quick Start Script for Windows

echo 🚀 Starting Indian Stock Market Predictor...

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Check if .env file exists
if not exist ".env" (
    echo ⚙️ Creating .env file from template...
    copy .env.template .env
    echo ⚠️  Please edit .env file and add your Alpha Vantage API key!
    echo ⚠️  Get your free API key from: https://www.alphavantage.co/support/#api-key
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "static" mkdir static

REM Start the application
echo 🌐 Starting Flask web application...
echo 📱 Open your browser and go to: http://localhost:5000
echo 🛑 Press Ctrl+C to stop the application
echo.

python app.py

pause
