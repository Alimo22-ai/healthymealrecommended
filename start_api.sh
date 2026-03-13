#!/bin/bash
# Startup script for Meal Recommendation System

echo "======================================"
echo "NutriMatch AI - Starting Services"
echo "======================================"

# Check if data files exist
if [ ! -f "recips.json" ]; then
    echo "ERROR: recipes.json not found!"
    exit 1
fi

if [ ! -f "nutrition_dataset.csv" ]; then
    echo "ERROR: nutrition_dataset.csv not found!"
    exit 1
fi

echo "Data files found!"
echo ""

# Install dependencies if needed
echo "Installing dependencies..."
cd /workspace/meal_recommender
pip install -q -r requirements.txt 2>/dev/null || true

echo ""
echo "Starting FastAPI server on port 8000..."
echo "API documentation: http://localhost:8000/docs"
echo ""

# Start API server
python -m uvicorn api:app --host 0.0.0.0 --port 8000
