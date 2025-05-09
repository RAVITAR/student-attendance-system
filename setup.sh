#!/bin/bash
set -e  # Exit on error

echo "Updating package list..."
sudo apt update

echo "Installing system dependencies..."
sudo apt install -y python3-rpi.gpio python3-spidev

# Create and activate a virtual environment if it doesn't already exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete!"
