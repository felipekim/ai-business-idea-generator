"""
Production Configuration for AI Business Idea Generator
Optimized settings for production deployment
"""

import os
from datetime import timedelta

class ProductionConfig:
    """Production configuration settings"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'production-secret-key-change-this')
    DEBUG = False
    TESTING = False
    
    # Database Configuration
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///production_ideas.db')
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_timeout': 20,
        'max_overflow': 0
    }
    
    # API Keys (Environment Variables)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    
    # CORS Configuration
    CORS_ORIGINS = ["*"]  # Configure specific origins in production
    CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS = ["Content-Type", "Authorization"]
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'memory://')
    RATELIMIT_DEFAULT = "100 per hour"
    
    # Caching Configuration
    CACHE_TYPE = "simple"  # Use Redis in production: "redis"
    CACHE_REDIS_URL = os.environ.get('REDIS_URL')
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Automation Configuration
    AUTOMATION_ENABLED = True
    WEEKLY_AUTOMATION_SCHEDULE = "0 9 * * 1"  # Monday 9 AM
    MAX_CONCURRENT_WORKFLOWS = 3
    WORKFLOW_TIMEOUT_MINUTES = 15
    MIN_IDEA_QUALITY_SCORE = 6.0
    
    # Performance Configuration
    ENABLE_OPTIMIZATION = True
    ENABLE_PARALLEL_EXECUTION = True
    ENABLE_CACHING = True
    ENABLE_MONITORING = True
    MAX_WORKER_THREADS = 8
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'
    
    # Security Configuration
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', '/tmp/uploads')
    
    # Monitoring and Health Checks
    HEALTH_CHECK_ENABLED = True
    METRICS_ENABLED = True
    PERFORMANCE_MONITORING = True
    
    # Email Configuration (if needed)
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    @staticmethod
    def validate_config():
        """Validate required configuration"""
        required_vars = ['OPENAI_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True

class DevelopmentConfig:
    """Development configuration settings"""
    
    SECRET_KEY = 'dev-secret-key'
    DEBUG = True
    TESTING = False
    
    DATABASE_URL = 'sqlite:///development_ideas.db'
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    
    # Use environment variable if available, otherwise use placeholder
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your-openai-api-key-here')
    
    CORS_ORIGINS = ["*"]
    CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    AUTOMATION_ENABLED = True
    MAX_CONCURRENT_WORKFLOWS = 2
    WORKFLOW_TIMEOUT_MINUTES = 10
    MIN_IDEA_QUALITY_SCORE = 5.0
    
    ENABLE_OPTIMIZATION = True
    ENABLE_PARALLEL_EXECUTION = True
    ENABLE_CACHING = True
    ENABLE_MONITORING = True
    MAX_WORKER_THREADS = 4
    
    LOG_LEVEL = 'DEBUG'
    HEALTH_CHECK_ENABLED = True
    METRICS_ENABLED = True

class TestingConfig:
    """Testing configuration settings"""
    
    SECRET_KEY = 'test-secret-key'
    DEBUG = False
    TESTING = True
    
    DATABASE_URL = 'sqlite:///:memory:'
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    OPENAI_API_KEY = 'test-api-key'
    
    AUTOMATION_ENABLED = False
    MAX_CONCURRENT_WORKFLOWS = 1
    WORKFLOW_TIMEOUT_MINUTES = 5
    MIN_IDEA_QUALITY_SCORE = 4.0
    
    ENABLE_OPTIMIZATION = False
    ENABLE_PARALLEL_EXECUTION = False
    ENABLE_CACHING = False
    ENABLE_MONITORING = False
    MAX_WORKER_THREADS = 2
    
    LOG_LEVEL = 'WARNING'
    HEALTH_CHECK_ENABLED = False
    METRICS_ENABLED = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration based on environment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config.get(config_name, config['default'])

