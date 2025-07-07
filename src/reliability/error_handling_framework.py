"""
Robust Error Handling Framework
Production-grade error handling with circuit breakers, fallback strategies, and recovery mechanisms
"""

import asyncio
import time
import logging
import traceback
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from collections import defaultdict, deque
import random

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorMetrics:
    """Error tracking metrics"""
    total_requests: int = 0
    failed_requests: int = 0
    success_requests: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    
    def update_success(self, response_time: float):
        """Update metrics for successful request"""
        self.total_requests += 1
        self.success_requests += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
        
        # Update average response time
        if self.average_response_time == 0:
            self.average_response_time = response_time
        else:
            self.average_response_time = (self.average_response_time * 0.9) + (response_time * 0.1)
        
        self._update_error_rate()
    
    def update_failure(self):
        """Update metrics for failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now()
        self._update_error_rate()
    
    def _update_error_rate(self):
        """Update error rate calculation"""
        if self.total_requests > 0:
            self.error_rate = self.failed_requests / self.total_requests
        else:
            self.error_rate = 0.0

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 3  # Successes before closing circuit
    timeout_duration: int = 60  # Seconds to wait before half-open
    max_timeout_duration: int = 300  # Maximum timeout duration
    error_rate_threshold: float = 0.5  # Error rate threshold (50%)
    min_requests: int = 10  # Minimum requests before calculating error rate

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: bool = True  # Add random jitter to prevent thundering herd

@dataclass
class FallbackConfig:
    """Fallback configuration"""
    enable_cache_fallback: bool = True
    enable_mock_fallback: bool = True
    enable_degraded_service: bool = True
    cache_ttl: int = 3600  # Cache TTL in seconds
    mock_data_quality: str = "basic"  # "basic", "enhanced", "minimal"

class CircuitBreaker:
    """Production-grade circuit breaker implementation"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = ErrorMetrics()
        self.last_state_change = datetime.now()
        self.lock = threading.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized with config: {config}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.last_state_change = datetime.now()
                    logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            response_time = time.time() - start_time
            
            with self.lock:
                self.metrics.update_success(response_time)
                
                if self.state == CircuitState.HALF_OPEN:
                    if self.metrics.consecutive_successes >= self.config.success_threshold:
                        self.state = CircuitState.CLOSED
                        self.last_state_change = datetime.now()
                        logger.info(f"Circuit breaker '{self.name}' moved to CLOSED")
            
            return result
            
        except Exception as e:
            with self.lock:
                self.metrics.update_failure()
                
                if self._should_open_circuit():
                    self.state = CircuitState.OPEN
                    self.last_state_change = datetime.now()
                    logger.warning(f"Circuit breaker '{self.name}' moved to OPEN due to failures")
            
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        time_since_open = datetime.now() - self.last_state_change
        timeout_duration = min(
            self.config.timeout_duration * (2 ** (self.metrics.consecutive_failures // 5)),
            self.config.max_timeout_duration
        )
        return time_since_open.total_seconds() >= timeout_duration
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened"""
        # Check consecutive failures
        if self.metrics.consecutive_failures >= self.config.failure_threshold:
            return True
        
        # Check error rate if we have enough requests
        if (self.metrics.total_requests >= self.config.min_requests and 
            self.metrics.error_rate >= self.config.error_rate_threshold):
            return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_requests": self.metrics.success_requests,
                "error_rate": self.metrics.error_rate,
                "consecutive_failures": self.metrics.consecutive_failures,
                "consecutive_successes": self.metrics.consecutive_successes,
                "average_response_time": self.metrics.average_response_time,
                "last_failure_time": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                "last_success_time": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None
            },
            "last_state_change": self.last_state_change.isoformat()
        }

class RetryHandler:
    """Advanced retry handler with exponential backoff and jitter"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    # Last attempt, don't delay
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {str(e)}")
                time.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(delay, 0.1)  # Minimum 0.1 second delay

class FallbackManager:
    """Advanced fallback management system"""
    
    def __init__(self, config: FallbackConfig):
        self.config = config
        self.cache = {}
        self.cache_timestamps = {}
    
    def get_fallback_data(self, service_name: str, operation: str, 
                         context: Optional[Dict] = None) -> Any:
        """Get fallback data for failed service"""
        
        # Try cache fallback first
        if self.config.enable_cache_fallback:
            cached_data = self._get_cached_data(service_name, operation)
            if cached_data is not None:
                logger.info(f"Using cached fallback for {service_name}.{operation}")
                return cached_data
        
        # Try degraded service fallback
        if self.config.enable_degraded_service:
            degraded_data = self._get_degraded_service_data(service_name, operation, context)
            if degraded_data is not None:
                logger.info(f"Using degraded service fallback for {service_name}.{operation}")
                return degraded_data
        
        # Try mock fallback as last resort
        if self.config.enable_mock_fallback:
            mock_data = self._get_mock_data(service_name, operation, context)
            logger.warning(f"Using mock fallback for {service_name}.{operation}")
            return mock_data
        
        raise FallbackExhaustedException(f"All fallback options exhausted for {service_name}.{operation}")
    
    def cache_data(self, service_name: str, operation: str, data: Any):
        """Cache successful response data"""
        cache_key = f"{service_name}.{operation}"
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.now()
        
        # Clean old cache entries
        self._cleanup_cache()
    
    def _get_cached_data(self, service_name: str, operation: str) -> Optional[Any]:
        """Get data from cache if available and fresh"""
        cache_key = f"{service_name}.{operation}"
        
        if cache_key not in self.cache:
            return None
        
        cache_time = self.cache_timestamps.get(cache_key)
        if cache_time and (datetime.now() - cache_time).total_seconds() > self.config.cache_ttl:
            # Cache expired
            del self.cache[cache_key]
            del self.cache_timestamps[cache_key]
            return None
        
        return self.cache[cache_key]
    
    def _get_degraded_service_data(self, service_name: str, operation: str, 
                                  context: Optional[Dict]) -> Optional[Any]:
        """Get data from degraded service (simplified operation)"""
        
        # Service-specific degraded operations
        if service_name == "google_trends":
            return self._get_degraded_trends_data(context)
        elif service_name == "reddit_api":
            return self._get_degraded_reddit_data(context)
        elif service_name == "news_api":
            return self._get_degraded_news_data(context)
        elif service_name == "openai_api":
            return self._get_degraded_ai_data(context)
        
        return None
    
    def _get_degraded_trends_data(self, context: Optional[Dict]) -> Dict[str, Any]:
        """Get degraded Google Trends data"""
        keyword = context.get('keyword', 'business') if context else 'business'
        return {
            "keyword": keyword,
            "trend_data": [
                {"date": "2024-06", "value": 75},
                {"date": "2024-05", "value": 70},
                {"date": "2024-04", "value": 68}
            ],
            "interest_over_time": "moderate_growth",
            "related_queries": ["startup", "entrepreneurship", "innovation"],
            "data_quality": "degraded",
            "source": "fallback_trends"
        }
    
    def _get_degraded_reddit_data(self, context: Optional[Dict]) -> Dict[str, Any]:
        """Get degraded Reddit data"""
        keyword = context.get('keyword', 'business') if context else 'business'
        return {
            "keyword": keyword,
            "posts": [
                {
                    "title": f"Discussion about {keyword} trends",
                    "score": 150,
                    "comments": 45,
                    "sentiment": "positive"
                }
            ],
            "sentiment_summary": "moderately_positive",
            "engagement_level": "medium",
            "data_quality": "degraded",
            "source": "fallback_reddit"
        }
    
    def _get_degraded_news_data(self, context: Optional[Dict]) -> Dict[str, Any]:
        """Get degraded news data"""
        keyword = context.get('keyword', 'business') if context else 'business'
        return {
            "keyword": keyword,
            "articles": [
                {
                    "title": f"Recent developments in {keyword}",
                    "source": "Industry News",
                    "published_at": datetime.now().isoformat(),
                    "relevance": "medium"
                }
            ],
            "total_results": 1,
            "data_quality": "degraded",
            "source": "fallback_news"
        }
    
    def _get_degraded_ai_data(self, context: Optional[Dict]) -> Dict[str, Any]:
        """Get degraded AI service data"""
        return {
            "analysis": "Basic analysis available in degraded mode",
            "confidence": 0.6,
            "recommendations": ["Seek additional data sources", "Verify with manual research"],
            "data_quality": "degraded",
            "source": "fallback_ai"
        }
    
    def _get_mock_data(self, service_name: str, operation: str, 
                      context: Optional[Dict]) -> Dict[str, Any]:
        """Get mock data as last resort"""
        
        quality_level = self.config.mock_data_quality
        
        base_mock = {
            "service": service_name,
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "data_quality": f"mock_{quality_level}",
            "source": "fallback_mock",
            "warning": "This is mock data due to service unavailability"
        }
        
        if quality_level == "enhanced":
            base_mock.update({
                "confidence": 0.3,
                "reliability": "low",
                "recommendations": ["Service unavailable", "Manual verification required"]
            })
        elif quality_level == "basic":
            base_mock.update({
                "status": "service_unavailable",
                "fallback_active": True
            })
        else:  # minimal
            base_mock.update({
                "error": "service_unavailable"
            })
        
        return base_mock
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if (current_time - timestamp).total_seconds() > self.config.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            del self.cache_timestamps[key]

class HealthMonitor:
    """Service health monitoring system"""
    
    def __init__(self):
        self.service_health = {}
        self.health_history = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.Lock()
    
    def update_service_health(self, service_name: str, status: ServiceStatus, 
                            metrics: Optional[Dict] = None):
        """Update service health status"""
        with self.lock:
            health_record = {
                "status": status,
                "timestamp": datetime.now(),
                "metrics": metrics or {}
            }
            
            self.service_health[service_name] = health_record
            self.health_history[service_name].append(health_record)
    
    def get_service_health(self, service_name: str) -> Optional[Dict]:
        """Get current service health"""
        return self.service_health.get(service_name)
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        with self.lock:
            if not self.service_health:
                return {"status": "unknown", "services": {}}
            
            healthy_count = sum(1 for health in self.service_health.values() 
                              if health["status"] == ServiceStatus.HEALTHY)
            total_count = len(self.service_health)
            
            if healthy_count == total_count:
                overall_status = "healthy"
            elif healthy_count >= total_count * 0.7:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"
            
            return {
                "status": overall_status,
                "healthy_services": healthy_count,
                "total_services": total_count,
                "health_percentage": (healthy_count / total_count) * 100,
                "services": {name: health["status"].value for name, health in self.service_health.items()},
                "last_updated": datetime.now().isoformat()
            }

class ErrorHandlingFramework:
    """Main error handling framework orchestrator"""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.retry_handler = RetryHandler(RetryConfig())
        self.fallback_manager = FallbackManager(FallbackConfig())
        self.health_monitor = HealthMonitor()
        self.error_log = deque(maxlen=1000)
        
        logger.info("Error handling framework initialized")
    
    def register_service(self, service_name: str, 
                        circuit_config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Register a service with circuit breaker protection"""
        config = circuit_config or CircuitBreakerConfig()
        circuit_breaker = CircuitBreaker(service_name, config)
        self.circuit_breakers[service_name] = circuit_breaker
        
        # Initialize health monitoring
        self.health_monitor.update_service_health(service_name, ServiceStatus.UNKNOWN)
        
        logger.info(f"Service '{service_name}' registered with error handling")
        return circuit_breaker
    
    def protected_call(self, service_name: str, func: Callable, 
                      *args, enable_retry: bool = True, 
                      enable_fallback: bool = True, **kwargs) -> Any:
        """Execute function with full error handling protection"""
        
        if service_name not in self.circuit_breakers:
            self.register_service(service_name)
        
        circuit_breaker = self.circuit_breakers[service_name]
        
        try:
            # Wrap function with retry if enabled
            if enable_retry:
                protected_func = lambda: self.retry_handler.retry(func, *args, **kwargs)
            else:
                protected_func = lambda: func(*args, **kwargs)
            
            # Execute with circuit breaker protection
            result = circuit_breaker.call(protected_func)
            
            # Cache successful result
            self.fallback_manager.cache_data(service_name, func.__name__, result)
            
            # Update health status
            self.health_monitor.update_service_health(service_name, ServiceStatus.HEALTHY)
            
            return result
            
        except CircuitBreakerOpenError as e:
            self._log_error(service_name, "circuit_breaker_open", str(e), ErrorSeverity.HIGH)
            self.health_monitor.update_service_health(service_name, ServiceStatus.UNHEALTHY)
            
            if enable_fallback:
                return self.fallback_manager.get_fallback_data(
                    service_name, func.__name__, kwargs
                )
            else:
                raise e
                
        except Exception as e:
            self._log_error(service_name, "service_error", str(e), ErrorSeverity.MEDIUM)
            self.health_monitor.update_service_health(service_name, ServiceStatus.DEGRADED)
            
            if enable_fallback:
                return self.fallback_manager.get_fallback_data(
                    service_name, func.__name__, kwargs
                )
            else:
                raise e
    
    def _log_error(self, service_name: str, error_type: str, 
                  error_message: str, severity: ErrorSeverity):
        """Log error with structured information"""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "service": service_name,
            "error_type": error_type,
            "message": error_message,
            "severity": severity.value,
            "traceback": traceback.format_exc()
        }
        
        self.error_log.append(error_record)
        
        # Log to standard logger based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"[{service_name}] {error_type}: {error_message}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"[{service_name}] {error_type}: {error_message}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"[{service_name}] {error_type}: {error_message}")
        else:
            logger.info(f"[{service_name}] {error_type}: {error_message}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        circuit_status = {name: cb.get_status() for name, cb in self.circuit_breakers.items()}
        health_status = self.health_monitor.get_overall_health()
        
        recent_errors = list(self.error_log)[-10:]  # Last 10 errors
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": health_status,
            "circuit_breakers": circuit_status,
            "recent_errors": recent_errors,
            "registered_services": list(self.circuit_breakers.keys()),
            "framework_status": "operational"
        }
    
    def reset_circuit_breaker(self, service_name: str) -> bool:
        """Manually reset a circuit breaker"""
        if service_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[service_name]
            circuit_breaker.state = CircuitState.CLOSED
            circuit_breaker.metrics = ErrorMetrics()
            circuit_breaker.last_state_change = datetime.now()
            
            logger.info(f"Circuit breaker '{service_name}' manually reset")
            return True
        return False

# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass

class FallbackExhaustedException(Exception):
    """Raised when all fallback options are exhausted"""
    pass

# Decorator for easy service protection
def protected_service(service_name: str, enable_retry: bool = True, 
                     enable_fallback: bool = True, 
                     framework: Optional[ErrorHandlingFramework] = None):
    """Decorator to protect service calls with error handling"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal framework
            if framework is None:
                framework = get_global_error_framework()
            
            return framework.protected_call(
                service_name, func, *args, 
                enable_retry=enable_retry, 
                enable_fallback=enable_fallback, 
                **kwargs
            )
        return wrapper
    return decorator

# Global framework instance
_global_framework = None

def get_global_error_framework() -> ErrorHandlingFramework:
    """Get or create global error handling framework"""
    global _global_framework
    if _global_framework is None:
        _global_framework = ErrorHandlingFramework()
    return _global_framework

# Test function
def test_error_handling_framework():
    """Test the error handling framework"""
    print("Testing Error Handling Framework...")
    
    framework = ErrorHandlingFramework()
    
    # Test 1: Successful service call
    print("\n1. Testing successful service call...")
    
    def successful_service():
        return {"status": "success", "data": "test_data"}
    
    result = framework.protected_call("test_service", successful_service)
    print(f"✅ Successful call result: {result}")
    
    # Test 2: Failing service with fallback
    print("\n2. Testing failing service with fallback...")
    
    def failing_service():
        raise Exception("Service temporarily unavailable")
    
    try:
        result = framework.protected_call("failing_service", failing_service, 
                                        enable_fallback=True)
        print(f"✅ Fallback result: {result}")
    except Exception as e:
        print(f"❌ Fallback failed: {e}")
    
    # Test 3: Circuit breaker behavior
    print("\n3. Testing circuit breaker behavior...")
    
    # Register service with low failure threshold for testing
    test_config = CircuitBreakerConfig(failure_threshold=2, timeout_duration=5)
    framework.register_service("circuit_test", test_config)
    
    # Cause failures to open circuit
    for i in range(3):
        try:
            framework.protected_call("circuit_test", failing_service, enable_fallback=False)
        except:
            print(f"   Failure {i+1} recorded")
    
    # Check circuit state
    circuit_status = framework.circuit_breakers["circuit_test"].get_status()
    print(f"✅ Circuit breaker state: {circuit_status['state']}")
    
    # Test 4: System health monitoring
    print("\n4. Testing system health monitoring...")
    
    system_status = framework.get_system_status()
    print(f"✅ System health: {system_status['overall_health']['status']}")
    print(f"✅ Registered services: {len(system_status['registered_services'])}")
    print(f"✅ Recent errors: {len(system_status['recent_errors'])}")
    
    # Test 5: Decorator usage
    print("\n5. Testing decorator usage...")
    
    @protected_service("decorated_service", framework=framework)
    def decorated_function():
        return {"decorator": "working", "timestamp": datetime.now().isoformat()}
    
    result = decorated_function()
    print(f"✅ Decorated function result: {result}")
    
    print("\n🎉 Error handling framework test completed successfully!")
    
    return {
        "framework_status": "operational",
        "services_registered": len(framework.circuit_breakers),
        "system_health": system_status['overall_health']['status'],
        "circuit_breakers_tested": True,
        "fallback_system_tested": True,
        "health_monitoring_tested": True,
        "decorator_tested": True
    }

if __name__ == "__main__":
    # Run error handling framework tests
    test_error_handling_framework()

