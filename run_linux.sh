#!/bin/bash

echo "=========================================="
echo "   Document Assistant - Streamlit App"
echo "=========================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python not found. Please install Python 3.8+"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists"
fi

echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo "[3/4] Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f "app.py" ]; then
    echo "[ERROR] app.py not found in current directory!"
    echo "Current directory: $(pwd)"
    ls *.py
    exit 1
fi

echo ""
echo "=========================================="
echo "   [4/4] Starting application..."
echo "   Open browser: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo "=========================================="
echo ""

streamlit run app.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Failed to start application"
    echo "Check if streamlit is installed: pip show streamlit"
    exit 1
fi