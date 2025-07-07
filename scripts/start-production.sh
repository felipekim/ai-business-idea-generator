#!/bin/bash
# Production startup script for AI Business Idea Generator

set -e

echo "🚀 Starting AI Business Idea Generator in Production Mode"
echo "=================================================="

# Set environment variables
export FLASK_ENV=production
export FLASK_APP=src/main.py

# Validate configuration
echo "📋 Validating production configuration..."
python -c "from production_config import ProductionConfig; ProductionConfig.validate_config(); print('✅ Configuration valid')"

# Initialize database if needed
echo "🗄️ Initializing database..."
python -c "
from src.main import app, db
with app.app_context():
    db.create_all()
    print('✅ Database initialized')
"

# Run database migrations if available
if [ -d "migrations" ]; then
    echo "🔄 Running database migrations..."
    flask db upgrade
    echo "✅ Database migrations completed"
fi

# Start the application with Gunicorn
echo "🌐 Starting application server..."
echo "Listening on 0.0.0.0:8000"
echo "Workers: 4"
echo "Timeout: 120s"
echo "=================================================="

exec gunicorn \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class gevent \
    --worker-connections 1000 \
    --timeout 120 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    "src.main:app"

