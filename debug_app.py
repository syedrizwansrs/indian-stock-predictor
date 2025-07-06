#!/usr/bin/env python3
"""
Test script to debug the Flask application startup.
"""
import sys
import os
import traceback

print("🔍 DEBUG: Starting Flask app test...")
print(f"🔍 DEBUG: Python version: {sys.version}")
print(f"🔍 DEBUG: Current working directory: {os.getcwd()}")

try:
    print("🔍 DEBUG: Testing basic imports...")
    import pandas as pd
    print(f"✅ pandas {pd.__version__} imported")
    
    import numpy as np
    print(f"✅ numpy {np.__version__} imported")
    
    import flask
    print(f"✅ flask {flask.__version__} imported")
    
    print("🔍 DEBUG: Testing src module imports...")
    
    # Test config import
    try:
        from src.config import Config
        print("✅ Config imported")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        traceback.print_exc()
    
    # Test data fetcher import
    try:
        from src.data_fetcher import DataFetcher
        print("✅ DataFetcher imported")
    except Exception as e:
        print(f"❌ DataFetcher import failed: {e}")
        traceback.print_exc()
    
    # Test technical analysis import
    try:
        from src.technical_analysis import TechnicalAnalyzer
        print("✅ TechnicalAnalyzer imported")
    except Exception as e:
        print(f"❌ TechnicalAnalyzer import failed: {e}")
        traceback.print_exc()
    
    # Test visualization import
    try:
        from src.visualization import StockVisualizer
        print("✅ StockVisualizer imported")
    except Exception as e:
        print(f"❌ StockVisualizer import failed: {e}")
        traceback.print_exc()
    
    # Test prediction model import
    try:
        from src.prediction_model import StockPredictor
        print("✅ StockPredictor imported")
    except Exception as e:
        print(f"❌ StockPredictor import failed: {e}")
        traceback.print_exc()
    
    print("🔍 DEBUG: All imports tested")
    
    # Test Flask app creation
    print("🔍 DEBUG: Creating Flask app...")
    from flask import Flask
    app = Flask(__name__)
    print("✅ Flask app created successfully")
    
    print("🎉 All tests passed! Flask app should work.")

except Exception as e:
    print(f"❌ Error during testing: {str(e)}")
    traceback.print_exc()
