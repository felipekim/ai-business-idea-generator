"""
Research Pipeline Configuration Management
Centralized configuration for all research components
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API configuration settings"""
    # Google Trends
    google_trends_enabled: bool = True
    google_trends_rate_limit: float = 2.0  # seconds between requests
    
    # Reddit API
    reddit_enabled: bool = True
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "BusinessIdeaResearcher/3.0"
    
    # News API
    news_enabled: bool = True
    news_api_key: Optional[str] = None
    news_rate_limit: float = 1.0
    
    # Fallback settings
    use_fallback_on_error: bool = True
    fallback_quality_penalty: float = 2.0  # Penalty for using fallback data

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    enabled: bool = True
    max_size: int = 1000
    default_ttl_hours: int = 24
    
    # TTL settings by data type
    ttl_trends: int = 6      # Google Trends - 6 hours
    ttl_news: int = 2        # News data - 2 hours  
    ttl_social: int = 4      # Social data - 4 hours
    ttl_research: int = 12   # Research results - 12 hours
    
    # Cache warming
    enable_warming: bool = True
    warming_keywords: List[str] = field(default_factory=lambda: [
        'artificial intelligence', 'machine learning', 'blockchain',
        'fintech', 'healthtech', 'edtech', 'saas', 'mobile app'
    ])

@dataclass
class ValidationConfig:
    """Validation configuration settings"""
    enabled: bool = True
    
    # Source diversity requirements
    min_sources: int = 5
    max_sources: int = 8
    min_source_types: int = 3
    max_sources_per_domain: int = 2
    min_unique_domains: int = 4
    
    # Quality thresholds
    min_confidence_score: float = 5.0
    excellent_threshold: float = 9.0
    good_threshold: float = 7.0
    acceptable_threshold: float = 5.0
    
    # Fact checking
    enable_fact_checking: bool = True
    fact_check_confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'high': 0.8,
        'medium': 0.6,
        'low': 0.4
    })

@dataclass
class ResearchConfig:
    """Research pipeline configuration"""
    # Research parameters
    default_target_sources: int = 6
    default_research_depth: str = 'moderate'  # 'light', 'moderate', 'deep'
    max_keywords_per_request: int = 8
    max_processing_time_seconds: int = 30
    
    # Keyword generation
    max_title_keywords: int = 5
    max_description_keywords: int = 3
    business_keyword_templates: List[str] = field(default_factory=lambda: [
        "{title} market",
        "{title} business", 
        "{title} industry",
        "{title} startup",
        "{title} trends"
    ])
    
    # Research synthesis
    max_key_findings: int = 5
    max_risk_factors: int = 3
    max_opportunities: int = 3
    max_recommendations: int = 5

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # File logging
    enable_file_logging: bool = True
    log_file_path: str = 'logs/research_pipeline.log'
    max_log_file_size_mb: int = 10
    backup_count: int = 5
    
    # Performance logging
    log_performance_metrics: bool = True
    log_api_calls: bool = True
    log_cache_operations: bool = False  # Set to True for debugging

@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    # Async settings
    max_concurrent_api_calls: int = 3
    api_timeout_seconds: int = 10
    
    # Rate limiting
    enable_rate_limiting: bool = True
    global_rate_limit: float = 0.5  # seconds between any API calls
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    
    # Memory management
    enable_memory_monitoring: bool = True
    max_memory_usage_mb: int = 500

@dataclass
class ResearchPipelineConfig:
    """Complete research pipeline configuration"""
    api: APIConfig = field(default_factory=APIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Environment settings
    environment: str = 'development'  # 'development', 'staging', 'production'
    debug_mode: bool = True
    
    def __post_init__(self):
        """Post-initialization configuration loading"""
        self._load_from_environment()
        self._validate_config()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # API keys from environment
        self.api.reddit_client_id = os.getenv('REDDIT_CLIENT_ID', self.api.reddit_client_id)
        self.api.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', self.api.reddit_client_secret)
        self.api.news_api_key = os.getenv('NEWS_API_KEY', self.api.news_api_key)
        
        # Environment settings
        self.environment = os.getenv('RESEARCH_ENVIRONMENT', self.environment)
        self.debug_mode = os.getenv('RESEARCH_DEBUG', str(self.debug_mode)).lower() == 'true'
        
        # Performance settings based on environment
        if self.environment == 'production':
            self.debug_mode = False
            self.logging.level = 'WARNING'
            self.logging.log_cache_operations = False
            self.performance.max_concurrent_api_calls = 5
        elif self.environment == 'staging':
            self.logging.level = 'INFO'
            self.performance.max_concurrent_api_calls = 3
    
    def _validate_config(self):
        """Validate configuration settings"""
        # Validate source requirements
        if self.validation.min_sources > self.validation.max_sources:
            raise ValueError("min_sources cannot be greater than max_sources")
        
        if self.validation.min_sources < 1:
            raise ValueError("min_sources must be at least 1")
        
        # Validate cache settings
        if self.cache.max_size < 10:
            raise ValueError("cache max_size must be at least 10")
        
        # Validate performance settings
        if self.performance.max_concurrent_api_calls < 1:
            raise ValueError("max_concurrent_api_calls must be at least 1")
        
        if self.performance.api_timeout_seconds < 1:
            raise ValueError("api_timeout_seconds must be at least 1")
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'api': {
                'google_trends_enabled': self.api.google_trends_enabled,
                'reddit_enabled': self.api.reddit_enabled,
                'news_enabled': self.api.news_enabled,
                'use_fallback_on_error': self.api.use_fallback_on_error
            },
            'cache': {
                'enabled': self.cache.enabled,
                'max_size': self.cache.max_size,
                'default_ttl_hours': self.cache.default_ttl_hours
            },
            'validation': {
                'enabled': self.validation.enabled,
                'min_sources': self.validation.min_sources,
                'max_sources': self.validation.max_sources,
                'min_confidence_score': self.validation.min_confidence_score
            },
            'research': {
                'default_target_sources': self.research.default_target_sources,
                'default_research_depth': self.research.default_research_depth,
                'max_processing_time_seconds': self.research.max_processing_time_seconds
            },
            'performance': {
                'max_concurrent_api_calls': self.performance.max_concurrent_api_calls,
                'enable_rate_limiting': self.performance.enable_rate_limiting,
                'max_retries': self.performance.max_retries
            },
            'environment': self.environment,
            'debug_mode': self.debug_mode
        }
    
    def save_to_file(self, file_path: str):
        """Save configuration to JSON file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ResearchPipelineConfig':
        """Load configuration from JSON file"""
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            
            # Create config with loaded values
            config = cls()
            
            # Update API settings
            if 'api' in config_dict:
                api_config = config_dict['api']
                config.api.google_trends_enabled = api_config.get('google_trends_enabled', config.api.google_trends_enabled)
                config.api.reddit_enabled = api_config.get('reddit_enabled', config.api.reddit_enabled)
                config.api.news_enabled = api_config.get('news_enabled', config.api.news_enabled)
                config.api.use_fallback_on_error = api_config.get('use_fallback_on_error', config.api.use_fallback_on_error)
            
            # Update cache settings
            if 'cache' in config_dict:
                cache_config = config_dict['cache']
                config.cache.enabled = cache_config.get('enabled', config.cache.enabled)
                config.cache.max_size = cache_config.get('max_size', config.cache.max_size)
                config.cache.default_ttl_hours = cache_config.get('default_ttl_hours', config.cache.default_ttl_hours)
            
            # Update validation settings
            if 'validation' in config_dict:
                validation_config = config_dict['validation']
                config.validation.enabled = validation_config.get('enabled', config.validation.enabled)
                config.validation.min_sources = validation_config.get('min_sources', config.validation.min_sources)
                config.validation.max_sources = validation_config.get('max_sources', config.validation.max_sources)
                config.validation.min_confidence_score = validation_config.get('min_confidence_score', config.validation.min_confidence_score)
            
            # Update research settings
            if 'research' in config_dict:
                research_config = config_dict['research']
                config.research.default_target_sources = research_config.get('default_target_sources', config.research.default_target_sources)
                config.research.default_research_depth = research_config.get('default_research_depth', config.research.default_research_depth)
                config.research.max_processing_time_seconds = research_config.get('max_processing_time_seconds', config.research.max_processing_time_seconds)
            
            # Update performance settings
            if 'performance' in config_dict:
                performance_config = config_dict['performance']
                config.performance.max_concurrent_api_calls = performance_config.get('max_concurrent_api_calls', config.performance.max_concurrent_api_calls)
                config.performance.enable_rate_limiting = performance_config.get('enable_rate_limiting', config.performance.enable_rate_limiting)
                config.performance.max_retries = performance_config.get('max_retries', config.performance.max_retries)
            
            # Update environment settings
            config.environment = config_dict.get('environment', config.environment)
            config.debug_mode = config_dict.get('debug_mode', config.debug_mode)
            
            logger.info(f"Configuration loaded from {file_path}")
            return config
            
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {file_path}, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}, using defaults")
            return cls()

# Global configuration instance
_global_config: Optional[ResearchPipelineConfig] = None

def get_config() -> ResearchPipelineConfig:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = ResearchPipelineConfig()
    return _global_config

def set_config(config: ResearchPipelineConfig):
    """Set global configuration instance"""
    global _global_config
    _global_config = config

def load_config_from_file(file_path: str) -> ResearchPipelineConfig:
    """Load and set global configuration from file"""
    config = ResearchPipelineConfig.load_from_file(file_path)
    set_config(config)
    return config

def create_default_config_file(file_path: str = 'config/research_config.json'):
    """Create default configuration file"""
    config = ResearchPipelineConfig()
    config.save_to_file(file_path)
    return config

# Configuration presets for different environments
def get_development_config() -> ResearchPipelineConfig:
    """Get development environment configuration"""
    config = ResearchPipelineConfig()
    config.environment = 'development'
    config.debug_mode = True
    config.logging.level = 'DEBUG'
    config.logging.log_cache_operations = True
    config.performance.max_concurrent_api_calls = 2
    config.cache.max_size = 500
    return config

def get_production_config() -> ResearchPipelineConfig:
    """Get production environment configuration"""
    config = ResearchPipelineConfig()
    config.environment = 'production'
    config.debug_mode = False
    config.logging.level = 'WARNING'
    config.logging.log_cache_operations = False
    config.performance.max_concurrent_api_calls = 5
    config.cache.max_size = 2000
    config.validation.min_confidence_score = 6.0  # Higher quality requirement
    return config

def get_testing_config() -> ResearchPipelineConfig:
    """Get testing environment configuration"""
    config = ResearchPipelineConfig()
    config.environment = 'testing'
    config.debug_mode = True
    config.api.use_fallback_on_error = True  # Always use fallback for consistent testing
    config.cache.enabled = False  # Disable cache for testing
    config.validation.min_sources = 3  # Lower requirements for testing
    config.performance.max_concurrent_api_calls = 1
    return config

# Test function for configuration system
def test_configuration_system():
    """Test the configuration system"""
    print("Testing Configuration System...")
    
    # Test default configuration
    print("\n1. Testing default configuration...")
    config = ResearchPipelineConfig()
    print(f"✅ Default environment: {config.environment}")
    print(f"✅ API enabled: {config.api.google_trends_enabled}")
    print(f"✅ Cache enabled: {config.cache.enabled}")
    print(f"✅ Validation enabled: {config.validation.enabled}")
    
    # Test configuration presets
    print("\n2. Testing configuration presets...")
    dev_config = get_development_config()
    prod_config = get_production_config()
    test_config = get_testing_config()
    
    print(f"✅ Development config: {dev_config.environment}")
    print(f"✅ Production config: {prod_config.environment}")
    print(f"✅ Testing config: {test_config.environment}")
    
    # Test configuration serialization
    print("\n3. Testing configuration serialization...")
    config_dict = config.to_dict()
    print(f"✅ Configuration serialized: {len(config_dict)} sections")
    
    # Test file operations
    print("\n4. Testing file operations...")
    test_file = '/tmp/test_config.json'
    config.save_to_file(test_file)
    loaded_config = ResearchPipelineConfig.load_from_file(test_file)
    print(f"✅ Configuration saved and loaded successfully")
    print(f"✅ Loaded environment: {loaded_config.environment}")
    
    # Test global configuration
    print("\n5. Testing global configuration...")
    set_config(dev_config)
    global_config = get_config()
    print(f"✅ Global config environment: {global_config.environment}")
    
    print("\n🎉 Configuration system test completed successfully!")
    
    return {
        'default_config': config,
        'dev_config': dev_config,
        'prod_config': prod_config,
        'test_config': test_config
    }

if __name__ == "__main__":
    # Run configuration tests
    test_configuration_system()

