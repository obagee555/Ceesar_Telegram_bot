#!/bin/bash

echo "🚀 Setting up Advanced Binary Trading Bot..."
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✅ Python 3 found"

# Install pip if not available
if ! command -v pip3 &> /dev/null; then
    echo "📦 Installing pip..."
    python3 -m ensurepip --upgrade
fi

echo "✅ pip found"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install --user -r requirements.txt

# Create necessary directories
echo "📂 Creating directories..."
mkdir -p /workspace/models
mkdir -p /workspace/backup

# Set proper permissions
echo "🔐 Setting permissions..."
chmod +x main.py
chmod +x simple_checkup.py

echo ""
echo "✅ Setup completed!"
echo ""
echo "🔍 Running system checkup..."
python3 simple_checkup.py

echo ""
echo "🚀 To start the bot:"
echo "   python3 main.py"