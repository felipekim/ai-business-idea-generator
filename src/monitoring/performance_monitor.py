"""
Performance Monitoring and Logging System
Comprehensive monitoring for research pipeline performance
"""

import time
import psutil
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str = 'general'
    metadata: Dict = field(default_factory=dict)

@dataclass
class APICallMetric:
    """API call performance metric"""
    api_name: str
    endpoint: str
    duration: float
    success: bool
    timestamp: datetime
    response_size: int = 0
    error_message: Optional[str] = None

@dataclass
class CacheMetric:
    """Cache operation metric"""
    operation: str  # 'hit', 'miss', 'set', 'evict'
    key: str
    duration: float
    timestamp: datetime
    cache_size: int = 0

@dataclass
class ResearchMetric:
    """Research operation metric"""
    idea_title: str
    total_duration: float
    sources_collected: int
    confidence_score: float
    validation_passed: bool
    timestamp: datetime
    processing_stages: Dict[str, float] = field(default_factory=dict)

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, max_metrics_history: int = 1000):
        self.max_metrics_history = max_metrics_history
        
        # Metric storage
        self.performance_metrics = deque(maxlen=max_metrics_history)
        self.api_metrics = deque(maxlen=max_metrics_history)
        self.cache_metrics = deque(maxlen=max_metrics_history)
        self.research_metrics = deque(maxlen=max_metrics_history)
        
        # Real-time statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'peak_memory_usage': 0.0,
            'cache_hit_rate': 0.0,
            'api_success_rate': 0.0
        }
        
        # Performance thresholds
        self.thresholds = {
            'max_response_time': 30.0,  # seconds
            'max_memory_usage': 500.0,  # MB
            'min_cache_hit_rate': 50.0,  # percentage
            'min_api_success_rate': 90.0  # percentage
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.start_time = datetime.now()
        
        logger.info("Performance Monitor initialized")
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring_active = True
        self.start_time = datetime.now()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")
    
    def record_performance_metric(self, name: str, value: float, unit: str, 
                                category: str = 'general', metadata: Dict = None):
        """Record a performance metric"""
        if not self.monitoring_active:
            return
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            category=category,
            metadata=metadata or {}
        )
        
        self.performance_metrics.append(metric)
        self._update_stats()
    
    def record_api_call(self, api_name: str, endpoint: str, duration: float, 
                       success: bool, response_size: int = 0, error_message: str = None):
        """Record an API call metric"""
        if not self.monitoring_active:
            return
        
        metric = APICallMetric(
            api_name=api_name,
            endpoint=endpoint,
            duration=duration,
            success=success,
            timestamp=datetime.now(),
            response_size=response_size,
            error_message=error_message
        )
        
        self.api_metrics.append(metric)
        self._update_api_stats()
    
    def record_cache_operation(self, operation: str, key: str, duration: float, cache_size: int = 0):
        """Record a cache operation metric"""
        if not self.monitoring_active:
            return
        
        metric = CacheMetric(
            operation=operation,
            key=key,
            duration=duration,
            timestamp=datetime.now(),
            cache_size=cache_size
        )
        
        self.cache_metrics.append(metric)
        self._update_cache_stats()
    
    def record_research_operation(self, idea_title: str, total_duration: float, 
                                sources_collected: int, confidence_score: float,
                                validation_passed: bool, processing_stages: Dict[str, float] = None):
        """Record a research operation metric"""
        if not self.monitoring_active:
            return
        
        metric = ResearchMetric(
            idea_title=idea_title,
            total_duration=total_duration,
            sources_collected=sources_collected,
            confidence_score=confidence_score,
            validation_passed=validation_passed,
            timestamp=datetime.now(),
            processing_stages=processing_stages or {}
        )
        
        self.research_metrics.append(metric)
        self._update_research_stats()
    
    def _update_stats(self):
        """Update general statistics"""
        if not self.performance_metrics:
            return
        
        # Update memory usage
        memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
        self.stats['peak_memory_usage'] = max(self.stats['peak_memory_usage'], memory_usage)
    
    def _update_api_stats(self):
        """Update API statistics"""
        if not self.api_metrics:
            return
        
        # Calculate API success rate
        total_calls = len(self.api_metrics)
        successful_calls = sum(1 for metric in self.api_metrics if metric.success)
        self.stats['api_success_rate'] = (successful_calls / total_calls) * 100
        
        # Calculate average response time
        total_duration = sum(metric.duration for metric in self.api_metrics)
        self.stats['average_response_time'] = total_duration / total_calls
    
    def _update_cache_stats(self):
        """Update cache statistics"""
        if not self.cache_metrics:
            return
        
        # Calculate cache hit rate
        hits = sum(1 for metric in self.cache_metrics if metric.operation == 'hit')
        misses = sum(1 for metric in self.cache_metrics if metric.operation == 'miss')
        total_requests = hits + misses
        
        if total_requests > 0:
            self.stats['cache_hit_rate'] = (hits / total_requests) * 100
    
    def _update_research_stats(self):
        """Update research statistics"""
        if not self.research_metrics:
            return
        
        # Update request counts
        self.stats['total_requests'] = len(self.research_metrics)
        self.stats['successful_requests'] = sum(1 for metric in self.research_metrics if metric.validation_passed)
        self.stats['failed_requests'] = self.stats['total_requests'] - self.stats['successful_requests']
    
    def get_current_stats(self) -> Dict:
        """Get current performance statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        current_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        cpu_usage = psutil.cpu_percent()
        
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'current_memory_usage_mb': current_memory,
            'cpu_usage_percent': cpu_usage,
            'monitoring_active': self.monitoring_active,
            'metrics_collected': {
                'performance': len(self.performance_metrics),
                'api_calls': len(self.api_metrics),
                'cache_operations': len(self.cache_metrics),
                'research_operations': len(self.research_metrics)
            }
        }
    
    def get_performance_report(self, time_window_minutes: int = 60) -> Dict:
        """Get detailed performance report for specified time window"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        # Filter metrics by time window
        recent_api_metrics = [m for m in self.api_metrics if m.timestamp >= cutoff_time]
        recent_cache_metrics = [m for m in self.cache_metrics if m.timestamp >= cutoff_time]
        recent_research_metrics = [m for m in self.research_metrics if m.timestamp >= cutoff_time]
        
        # API performance analysis
        api_analysis = self._analyze_api_performance(recent_api_metrics)
        
        # Cache performance analysis
        cache_analysis = self._analyze_cache_performance(recent_cache_metrics)
        
        # Research performance analysis
        research_analysis = self._analyze_research_performance(recent_research_metrics)
        
        # System health check
        health_status = self._check_system_health()
        
        return {
            'time_window_minutes': time_window_minutes,
            'report_timestamp': datetime.now().isoformat(),
            'api_performance': api_analysis,
            'cache_performance': cache_analysis,
            'research_performance': research_analysis,
            'system_health': health_status,
            'recommendations': self._generate_performance_recommendations()
        }
    
    def _analyze_api_performance(self, metrics: List[APICallMetric]) -> Dict:
        """Analyze API performance metrics"""
        if not metrics:
            return {'status': 'no_data'}
        
        # Group by API
        api_groups = defaultdict(list)
        for metric in metrics:
            api_groups[metric.api_name].append(metric)
        
        api_analysis = {}
        for api_name, api_metrics in api_groups.items():
            total_calls = len(api_metrics)
            successful_calls = sum(1 for m in api_metrics if m.success)
            avg_duration = sum(m.duration for m in api_metrics) / total_calls
            
            api_analysis[api_name] = {
                'total_calls': total_calls,
                'success_rate': (successful_calls / total_calls) * 100,
                'average_duration': avg_duration,
                'max_duration': max(m.duration for m in api_metrics),
                'min_duration': min(m.duration for m in api_metrics)
            }
        
        return api_analysis
    
    def _analyze_cache_performance(self, metrics: List[CacheMetric]) -> Dict:
        """Analyze cache performance metrics"""
        if not metrics:
            return {'status': 'no_data'}
        
        operations = defaultdict(int)
        for metric in metrics:
            operations[metric.operation] += 1
        
        total_requests = operations['hit'] + operations['miss']
        hit_rate = (operations['hit'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_operations': len(metrics),
            'hit_rate': hit_rate,
            'operations_breakdown': dict(operations),
            'average_cache_size': sum(m.cache_size for m in metrics if m.cache_size > 0) / len(metrics)
        }
    
    def _analyze_research_performance(self, metrics: List[ResearchMetric]) -> Dict:
        """Analyze research performance metrics"""
        if not metrics:
            return {'status': 'no_data'}
        
        total_requests = len(metrics)
        successful_requests = sum(1 for m in metrics if m.validation_passed)
        avg_duration = sum(m.total_duration for m in metrics) / total_requests
        avg_sources = sum(m.sources_collected for m in metrics) / total_requests
        avg_confidence = sum(m.confidence_score for m in metrics) / total_requests
        
        return {
            'total_requests': total_requests,
            'success_rate': (successful_requests / total_requests) * 100,
            'average_duration': avg_duration,
            'average_sources_collected': avg_sources,
            'average_confidence_score': avg_confidence,
            'max_duration': max(m.total_duration for m in metrics),
            'min_duration': min(m.total_duration for m in metrics)
        }
    
    def _check_system_health(self) -> Dict:
        """Check overall system health"""
        current_stats = self.get_current_stats()
        
        health_issues = []
        health_score = 100.0
        
        # Check response time
        if current_stats['average_response_time'] > self.thresholds['max_response_time']:
            health_issues.append(f"High response time: {current_stats['average_response_time']:.2f}s")
            health_score -= 20
        
        # Check memory usage
        if current_stats['current_memory_usage_mb'] > self.thresholds['max_memory_usage']:
            health_issues.append(f"High memory usage: {current_stats['current_memory_usage_mb']:.1f}MB")
            health_score -= 15
        
        # Check cache hit rate
        if current_stats['cache_hit_rate'] < self.thresholds['min_cache_hit_rate']:
            health_issues.append(f"Low cache hit rate: {current_stats['cache_hit_rate']:.1f}%")
            health_score -= 10
        
        # Check API success rate
        if current_stats['api_success_rate'] < self.thresholds['min_api_success_rate']:
            health_issues.append(f"Low API success rate: {current_stats['api_success_rate']:.1f}%")
            health_score -= 25
        
        # Determine health status
        if health_score >= 90:
            health_status = 'excellent'
        elif health_score >= 70:
            health_status = 'good'
        elif health_score >= 50:
            health_status = 'fair'
        else:
            health_status = 'poor'
        
        return {
            'health_score': max(health_score, 0),
            'health_status': health_status,
            'issues': health_issues,
            'system_metrics': {
                'cpu_usage': current_stats['cpu_usage_percent'],
                'memory_usage': current_stats['current_memory_usage_mb'],
                'uptime': current_stats['uptime_seconds']
            }
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        current_stats = self.get_current_stats()
        
        # Response time recommendations
        if current_stats['average_response_time'] > 10.0:
            recommendations.append("Consider increasing API timeout or optimizing data collection")
        
        # Memory usage recommendations
        if current_stats['current_memory_usage_mb'] > 300:
            recommendations.append("Consider reducing cache size or implementing memory cleanup")
        
        # Cache recommendations
        if current_stats['cache_hit_rate'] < 60:
            recommendations.append("Improve cache warming strategy or increase cache TTL")
        
        # API recommendations
        if current_stats['api_success_rate'] < 95:
            recommendations.append("Investigate API failures and improve error handling")
        
        return recommendations
    
    def export_metrics(self, file_path: str, format: str = 'json'):
        """Export metrics to file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'stats': self.get_current_stats(),
                'performance_metrics': [
                    {
                        'name': m.name,
                        'value': m.value,
                        'unit': m.unit,
                        'timestamp': m.timestamp.isoformat(),
                        'category': m.category,
                        'metadata': m.metadata
                    } for m in self.performance_metrics
                ],
                'api_metrics': [
                    {
                        'api_name': m.api_name,
                        'endpoint': m.endpoint,
                        'duration': m.duration,
                        'success': m.success,
                        'timestamp': m.timestamp.isoformat(),
                        'response_size': m.response_size,
                        'error_message': m.error_message
                    } for m in self.api_metrics
                ],
                'research_metrics': [
                    {
                        'idea_title': m.idea_title,
                        'total_duration': m.total_duration,
                        'sources_collected': m.sources_collected,
                        'confidence_score': m.confidence_score,
                        'validation_passed': m.validation_passed,
                        'timestamp': m.timestamp.isoformat(),
                        'processing_stages': m.processing_stages
                    } for m in self.research_metrics
                ]
            }
            
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

# Context manager for performance monitoring
class PerformanceContext:
    """Context manager for monitoring specific operations"""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str, category: str = 'general'):
        self.monitor = monitor
        self.operation_name = operation_name
        self.category = category
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            success = exc_type is None
            
            self.monitor.record_performance_metric(
                name=self.operation_name,
                value=duration,
                unit='seconds',
                category=self.category,
                metadata={'success': success}
            )

# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor

def monitor_operation(operation_name: str, category: str = 'general'):
    """Decorator for monitoring function performance"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            with PerformanceContext(monitor, operation_name, category):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            with PerformanceContext(monitor, operation_name, category):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Test function for performance monitoring
def test_performance_monitoring():
    """Test the performance monitoring system"""
    print("Testing Performance Monitoring System...")
    
    # Create monitor
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Test metric recording
    print("\n1. Testing metric recording...")
    monitor.record_performance_metric('test_metric', 1.5, 'seconds', 'test')
    monitor.record_api_call('test_api', '/test', 0.5, True, 1024)
    monitor.record_cache_operation('hit', 'test_key', 0.001, 100)
    monitor.record_research_operation('Test Idea', 5.0, 6, 8.5, True)
    
    print("✅ Metrics recorded successfully")
    
    # Test statistics
    print("\n2. Testing statistics...")
    stats = monitor.get_current_stats()
    print(f"✅ Current stats: {len(stats)} metrics")
    print(f"✅ Memory usage: {stats['current_memory_usage_mb']:.1f}MB")
    print(f"✅ Total requests: {stats['total_requests']}")
    
    # Test performance report
    print("\n3. Testing performance report...")
    report = monitor.get_performance_report(60)
    print(f"✅ Performance report generated")
    print(f"✅ System health: {report['system_health']['health_status']}")
    
    # Test context manager
    print("\n4. Testing context manager...")
    with PerformanceContext(monitor, 'test_operation', 'test'):
        time.sleep(0.1)  # Simulate work
    print("✅ Context manager worked")
    
    # Test export
    print("\n5. Testing metrics export...")
    export_path = '/tmp/test_metrics.json'
    monitor.export_metrics(export_path)
    print(f"✅ Metrics exported to {export_path}")
    
    monitor.stop_monitoring()
    print("\n🎉 Performance monitoring test completed successfully!")
    
    return monitor

if __name__ == "__main__":
    # Run performance monitoring tests
    test_performance_monitoring()

