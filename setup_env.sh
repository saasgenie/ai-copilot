#!/bin/bash

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required dependencies
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate

echo "Virtual environment setup complete. To activate it, run 'source venv/bin/activate'."
