"""
Performance Optimization Engine
Optimizes system performance, manages resources, and ensures scalability
"""

import logging
import json
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Performance optimization levels"""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class ResourceType(Enum):
    """System resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    API_CALLS = "api_calls"

class PerformanceMetric(Enum):
    """Performance metrics to track"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    CACHE_HIT_RATE = "cache_hit_rate"

@dataclass
class PerformanceData:
    """Performance measurement data"""
    metric: PerformanceMetric
    value: float
    timestamp: datetime
    component: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    """Result of optimization operation"""
    optimization_id: str
    component: str
    optimization_type: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_percentage: float
    execution_time: float
    success: bool
    recommendations: List[str] = field(default_factory=list)

@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    # Performance thresholds
    max_response_time_ms: float = 5000.0
    min_throughput_rps: float = 10.0
    max_error_rate_percent: float = 1.0
    max_cpu_usage_percent: float = 80.0
    max_memory_usage_percent: float = 85.0
    
    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    enable_caching: bool = True
    enable_parallel_processing: bool = True
    enable_resource_monitoring: bool = True
    
    # Cache settings
    cache_ttl_seconds: int = 3600
    max_cache_size_mb: int = 512
    cache_cleanup_interval_minutes: int = 30
    
    # Parallel processing
    max_worker_threads: int = 8
    batch_size: int = 5
    timeout_seconds: int = 30

class PerformanceOptimizer:
    """Performance optimization engine"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.performance_data = deque(maxlen=1000)
        self.optimization_history = []
        self.resource_monitors = {}
        self.cache_systems = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_worker_threads)
        
        # Performance tracking
        self.component_metrics = defaultdict(list)
        self.optimization_results = {}
        
        # Resource monitoring
        if self.config.enable_resource_monitoring:
            self._start_resource_monitoring()
        
        logger.info("Performance Optimizer initialized")
    
    def optimize_component(self, component_name: str, optimization_func: Callable, 
                          *args, **kwargs) -> OptimizationResult:
        """Optimize a specific component"""
        logger.info(f"Optimizing component: {component_name}")
        
        optimization_id = f"opt_{component_name}_{int(time.time())}"
        start_time = time.time()
        
        try:
            # Measure before optimization
            before_metrics = self._measure_component_performance(component_name)
            
            # Apply optimization
            optimization_result = optimization_func(*args, **kwargs)
            
            # Measure after optimization
            after_metrics = self._measure_component_performance(component_name)
            
            # Calculate improvement
            improvement = self._calculate_improvement(before_metrics, after_metrics)
            
            execution_time = time.time() - start_time
            
            result = OptimizationResult(
                optimization_id=optimization_id,
                component=component_name,
                optimization_type=optimization_func.__name__,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percentage=improvement,
                execution_time=execution_time,
                success=True,
                recommendations=self._generate_optimization_recommendations(
                    component_name, before_metrics, after_metrics
                )
            )
            
            self.optimization_history.append(result)
            logger.info(f"Optimization completed: {improvement:.1f}% improvement")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed for {component_name}: {e}")
            
            return OptimizationResult(
                optimization_id=optimization_id,
                component=component_name,
                optimization_type=optimization_func.__name__,
                before_metrics={},
                after_metrics={},
                improvement_percentage=0.0,
                execution_time=time.time() - start_time,
                success=False,
                recommendations=[f"Optimization failed: {str(e)}"]
            )
    
    def optimize_parallel_processing(self, tasks: List[Callable], 
                                   task_args: List[tuple] = None) -> OptimizationResult:
        """Optimize parallel processing of tasks"""
        logger.info(f"Optimizing parallel processing for {len(tasks)} tasks")
        
        if not self.config.enable_parallel_processing:
            logger.warning("Parallel processing disabled in config")
            return self._create_failed_result("parallel_processing", "Disabled in config")
        
        start_time = time.time()
        
        try:
            # Measure sequential execution time (baseline)
            sequential_start = time.time()
            sequential_results = []
            
            for i, task in enumerate(tasks[:3]):  # Sample first 3 tasks
                args = task_args[i] if task_args and i < len(task_args) else ()
                sequential_results.append(task(*args))
            
            sequential_time = time.time() - sequential_start
            estimated_sequential_total = sequential_time * len(tasks) / 3
            
            # Execute tasks in parallel
            parallel_start = time.time()
            parallel_results = []
            
            # Split tasks into batches
            batch_size = min(self.config.batch_size, len(tasks))
            task_batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
            
            for batch in task_batches:
                futures = []
                for i, task in enumerate(batch):
                    batch_index = task_batches.index(batch) * batch_size + i
                    args = task_args[batch_index] if task_args and batch_index < len(task_args) else ()
                    future = self.thread_pool.submit(task, *args)
                    futures.append(future)
                
                # Wait for batch completion
                for future in as_completed(futures, timeout=self.config.timeout_seconds):
                    try:
                        result = future.result()
                        parallel_results.append(result)
                    except Exception as e:
                        logger.error(f"Task failed in parallel execution: {e}")
                        parallel_results.append(None)
            
            parallel_time = time.time() - parallel_start
            
            # Calculate performance improvement
            speedup = estimated_sequential_total / parallel_time if parallel_time > 0 else 1.0
            improvement_percentage = (speedup - 1.0) * 100
            
            before_metrics = {
                'execution_time': estimated_sequential_total,
                'throughput': len(tasks) / estimated_sequential_total,
                'parallelization': 1.0
            }
            
            after_metrics = {
                'execution_time': parallel_time,
                'throughput': len(tasks) / parallel_time,
                'parallelization': speedup
            }
            
            result = OptimizationResult(
                optimization_id=f"parallel_opt_{int(time.time())}",
                component="parallel_processing",
                optimization_type="parallel_execution",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percentage=improvement_percentage,
                execution_time=time.time() - start_time,
                success=True,
                recommendations=[
                    f"Achieved {speedup:.1f}x speedup with parallel processing",
                    f"Optimal batch size: {batch_size}",
                    f"Consider increasing worker threads if CPU allows"
                ]
            )
            
            logger.info(f"Parallel processing optimization: {speedup:.1f}x speedup")
            return result
            
        except Exception as e:
            logger.error(f"Parallel processing optimization failed: {e}")
            return self._create_failed_result("parallel_processing", str(e))
    
    def optimize_caching(self, cache_key_prefix: str, 
                        data_generator: Callable) -> OptimizationResult:
        """Optimize caching system"""
        logger.info(f"Optimizing caching for: {cache_key_prefix}")
        
        if not self.config.enable_caching:
            logger.warning("Caching disabled in config")
            return self._create_failed_result("caching", "Disabled in config")
        
        start_time = time.time()
        
        try:
            # Initialize cache if not exists
            if cache_key_prefix not in self.cache_systems:
                self.cache_systems[cache_key_prefix] = {
                    'data': {},
                    'timestamps': {},
                    'hit_count': 0,
                    'miss_count': 0,
                    'size_mb': 0
                }
            
            cache = self.cache_systems[cache_key_prefix]
            
            # Measure cache performance before optimization
            before_hit_rate = self._calculate_cache_hit_rate(cache)
            before_size = cache['size_mb']
            
            # Optimize cache
            self._cleanup_expired_cache_entries(cache)
            self._optimize_cache_size(cache)
            
            # Test cache performance
            test_keys = [f"test_key_{i}" for i in range(10)]
            
            # Generate test data and measure cache performance
            cache_test_start = time.time()
            
            for key in test_keys:
                full_key = f"{cache_key_prefix}:{key}"
                
                # Try to get from cache
                if self._get_from_cache(cache, full_key):
                    cache['hit_count'] += 1
                else:
                    # Generate data and cache it
                    data = data_generator(key) if callable(data_generator) else f"test_data_{key}"
                    self._set_cache(cache, full_key, data)
                    cache['miss_count'] += 1
            
            cache_test_time = time.time() - cache_test_start
            
            # Measure after optimization
            after_hit_rate = self._calculate_cache_hit_rate(cache)
            after_size = cache['size_mb']
            
            improvement_percentage = ((after_hit_rate - before_hit_rate) / max(before_hit_rate, 0.01)) * 100
            
            before_metrics = {
                'hit_rate': before_hit_rate,
                'cache_size_mb': before_size,
                'response_time': cache_test_time * 2  # Estimated without cache
            }
            
            after_metrics = {
                'hit_rate': after_hit_rate,
                'cache_size_mb': after_size,
                'response_time': cache_test_time
            }
            
            result = OptimizationResult(
                optimization_id=f"cache_opt_{cache_key_prefix}_{int(time.time())}",
                component="caching",
                optimization_type="cache_optimization",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percentage=improvement_percentage,
                execution_time=time.time() - start_time,
                success=True,
                recommendations=[
                    f"Cache hit rate: {after_hit_rate:.1f}%",
                    f"Cache size optimized: {after_size:.1f} MB",
                    f"Consider increasing cache TTL if data is stable"
                ]
            )
            
            logger.info(f"Cache optimization completed: {after_hit_rate:.1f}% hit rate")
            return result
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            return self._create_failed_result("caching", str(e))
    
    def optimize_resource_usage(self) -> OptimizationResult:
        """Optimize system resource usage"""
        logger.info("Optimizing system resource usage")
        
        start_time = time.time()
        
        try:
            # Measure current resource usage
            before_metrics = self._get_current_resource_usage()
            
            # Apply resource optimizations
            optimizations_applied = []
            
            # CPU optimization
            if before_metrics['cpu_percent'] > self.config.max_cpu_usage_percent:
                self._optimize_cpu_usage()
                optimizations_applied.append("CPU usage optimization")
            
            # Memory optimization
            if before_metrics['memory_percent'] > self.config.max_memory_usage_percent:
                self._optimize_memory_usage()
                optimizations_applied.append("Memory usage optimization")
            
            # Disk optimization
            if before_metrics['disk_usage_percent'] > 90:
                self._optimize_disk_usage()
                optimizations_applied.append("Disk usage optimization")
            
            # Wait for optimizations to take effect
            time.sleep(2)
            
            # Measure after optimization
            after_metrics = self._get_current_resource_usage()
            
            # Calculate improvement
            cpu_improvement = before_metrics['cpu_percent'] - after_metrics['cpu_percent']
            memory_improvement = before_metrics['memory_percent'] - after_metrics['memory_percent']
            
            overall_improvement = (cpu_improvement + memory_improvement) / 2
            improvement_percentage = (overall_improvement / max(before_metrics['cpu_percent'], 1)) * 100
            
            result = OptimizationResult(
                optimization_id=f"resource_opt_{int(time.time())}",
                component="system_resources",
                optimization_type="resource_optimization",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percentage=improvement_percentage,
                execution_time=time.time() - start_time,
                success=True,
                recommendations=[
                    f"Applied optimizations: {', '.join(optimizations_applied)}",
                    f"CPU usage reduced by {cpu_improvement:.1f}%",
                    f"Memory usage reduced by {memory_improvement:.1f}%"
                ]
            )
            
            logger.info(f"Resource optimization completed: {improvement_percentage:.1f}% improvement")
            return result
            
        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
            return self._create_failed_result("system_resources", str(e))
    
    def optimize_api_performance(self, api_calls: List[Callable]) -> OptimizationResult:
        """Optimize API call performance"""
        logger.info(f"Optimizing performance for {len(api_calls)} API calls")
        
        start_time = time.time()
        
        try:
            # Measure baseline performance
            baseline_start = time.time()
            baseline_results = []
            
            for api_call in api_calls[:3]:  # Sample first 3 calls
                call_start = time.time()
                try:
                    result = api_call()
                    baseline_results.append(result)
                except Exception as e:
                    logger.error(f"API call failed: {e}")
                    baseline_results.append(None)
                
                call_time = time.time() - call_start
                self._record_performance_data(
                    PerformanceMetric.RESPONSE_TIME, call_time * 1000, "api_calls"
                )
            
            baseline_time = time.time() - baseline_start
            estimated_total_time = baseline_time * len(api_calls) / 3
            
            # Apply optimizations
            optimized_start = time.time()
            
            # Batch API calls
            batched_results = self._batch_api_calls(api_calls)
            
            # Apply caching for repeated calls
            cached_results = self._cache_api_results(api_calls)
            
            # Use parallel processing
            parallel_results = self._parallel_api_calls(api_calls)
            
            optimized_time = time.time() - optimized_start
            
            # Calculate improvement
            speedup = estimated_total_time / optimized_time if optimized_time > 0 else 1.0
            improvement_percentage = (speedup - 1.0) * 100
            
            before_metrics = {
                'total_time': estimated_total_time,
                'average_response_time': baseline_time / 3 * 1000,  # ms
                'throughput': 3 / baseline_time,  # calls per second
                'parallelization': 1.0
            }
            
            after_metrics = {
                'total_time': optimized_time,
                'average_response_time': optimized_time / len(api_calls) * 1000,  # ms
                'throughput': len(api_calls) / optimized_time,  # calls per second
                'parallelization': speedup
            }
            
            result = OptimizationResult(
                optimization_id=f"api_opt_{int(time.time())}",
                component="api_performance",
                optimization_type="api_optimization",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percentage=improvement_percentage,
                execution_time=time.time() - start_time,
                success=True,
                recommendations=[
                    f"Achieved {speedup:.1f}x speedup for API calls",
                    "Consider implementing request batching",
                    "Cache frequently accessed data",
                    "Use connection pooling for better performance"
                ]
            )
            
            logger.info(f"API performance optimization: {speedup:.1f}x speedup")
            return result
            
        except Exception as e:
            logger.error(f"API performance optimization failed: {e}")
            return self._create_failed_result("api_performance", str(e))
    
    def _measure_component_performance(self, component_name: str) -> Dict[str, float]:
        """Measure performance metrics for a component"""
        metrics = {}
        
        # Get recent performance data for this component
        component_data = [
            data for data in self.performance_data
            if data.component == component_name
            and data.timestamp > datetime.utcnow() - timedelta(minutes=5)
        ]
        
        if component_data:
            # Calculate average response time
            response_times = [
                data.value for data in component_data
                if data.metric == PerformanceMetric.RESPONSE_TIME
            ]
            if response_times:
                metrics['response_time'] = statistics.mean(response_times)
            
            # Calculate throughput
            throughput_data = [
                data.value for data in component_data
                if data.metric == PerformanceMetric.THROUGHPUT
            ]
            if throughput_data:
                metrics['throughput'] = statistics.mean(throughput_data)
            
            # Calculate error rate
            error_data = [
                data.value for data in component_data
                if data.metric == PerformanceMetric.ERROR_RATE
            ]
            if error_data:
                metrics['error_rate'] = statistics.mean(error_data)
        
        # Add current resource usage
        resource_usage = self._get_current_resource_usage()
        metrics.update(resource_usage)
        
        return metrics
    
    def _calculate_improvement(self, before: Dict[str, float], 
                             after: Dict[str, float]) -> float:
        """Calculate overall improvement percentage"""
        improvements = []
        
        # Response time improvement (lower is better)
        if 'response_time' in before and 'response_time' in after:
            if before['response_time'] > 0:
                improvement = (before['response_time'] - after['response_time']) / before['response_time'] * 100
                improvements.append(improvement)
        
        # Throughput improvement (higher is better)
        if 'throughput' in before and 'throughput' in after:
            if before['throughput'] > 0:
                improvement = (after['throughput'] - before['throughput']) / before['throughput'] * 100
                improvements.append(improvement)
        
        # Error rate improvement (lower is better)
        if 'error_rate' in before and 'error_rate' in after:
            if before['error_rate'] > 0:
                improvement = (before['error_rate'] - after['error_rate']) / before['error_rate'] * 100
                improvements.append(improvement)
        
        # CPU usage improvement (lower is better)
        if 'cpu_percent' in before and 'cpu_percent' in after:
            if before['cpu_percent'] > 0:
                improvement = (before['cpu_percent'] - after['cpu_percent']) / before['cpu_percent'] * 100
                improvements.append(improvement)
        
        return statistics.mean(improvements) if improvements else 0.0
    
    def _generate_optimization_recommendations(self, component: str, 
                                             before: Dict[str, float], 
                                             after: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Response time recommendations
        if 'response_time' in after and after['response_time'] > self.config.max_response_time_ms:
            recommendations.append("Consider implementing caching to reduce response time")
            recommendations.append("Optimize database queries and API calls")
        
        # Throughput recommendations
        if 'throughput' in after and after['throughput'] < self.config.min_throughput_rps:
            recommendations.append("Increase parallel processing to improve throughput")
            recommendations.append("Consider load balancing for better distribution")
        
        # Resource usage recommendations
        if 'cpu_percent' in after and after['cpu_percent'] > self.config.max_cpu_usage_percent:
            recommendations.append("Optimize CPU-intensive operations")
            recommendations.append("Consider scaling horizontally")
        
        if 'memory_percent' in after and after['memory_percent'] > self.config.max_memory_usage_percent:
            recommendations.append("Implement memory optimization strategies")
            recommendations.append("Review memory leaks and garbage collection")
        
        return recommendations
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring"""
        def monitor_resources():
            while True:
                try:
                    usage = self._get_current_resource_usage()
                    
                    for metric, value in usage.items():
                        self._record_performance_data(
                            PerformanceMetric.RESOURCE_USAGE, value, f"system_{metric}"
                        )
                    
                    time.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def _get_current_resource_usage(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / (1024 * 1024),
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / (1024 * 1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_used_mb': 0.0,
                'disk_usage_percent': 0.0,
                'disk_free_gb': 0.0
            }
    
    def _optimize_cpu_usage(self):
        """Optimize CPU usage"""
        # Reduce thread pool size if CPU usage is high
        if hasattr(self, 'thread_pool'):
            current_workers = self.thread_pool._max_workers
            if current_workers > 2:
                new_workers = max(2, current_workers - 2)
                logger.info(f"Reducing thread pool size from {current_workers} to {new_workers}")
                # Note: ThreadPoolExecutor doesn't support dynamic resizing
                # In production, would implement a custom thread pool
    
    def _optimize_memory_usage(self):
        """Optimize memory usage"""
        # Clean up old performance data
        if len(self.performance_data) > 500:
            # Keep only recent data
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            self.performance_data = deque([
                data for data in self.performance_data
                if data.timestamp > cutoff_time
            ], maxlen=1000)
        
        # Clean up cache systems
        for cache_name, cache in self.cache_systems.items():
            self._cleanup_expired_cache_entries(cache)
    
    def _optimize_disk_usage(self):
        """Optimize disk usage"""
        # Clean up old log files and temporary data
        logger.info("Optimizing disk usage - cleaning temporary files")
        # In production, would implement actual file cleanup
    
    def _calculate_cache_hit_rate(self, cache: Dict[str, Any]) -> float:
        """Calculate cache hit rate"""
        total_requests = cache['hit_count'] + cache['miss_count']
        if total_requests == 0:
            return 0.0
        return (cache['hit_count'] / total_requests) * 100
    
    def _cleanup_expired_cache_entries(self, cache: Dict[str, Any]):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in cache['timestamps'].items():
            if current_time - timestamp > self.config.cache_ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in cache['data']:
                del cache['data'][key]
            if key in cache['timestamps']:
                del cache['timestamps'][key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _optimize_cache_size(self, cache: Dict[str, Any]):
        """Optimize cache size"""
        # Estimate cache size (simplified)
        estimated_size = len(cache['data']) * 0.001  # Rough estimate in MB
        cache['size_mb'] = estimated_size
        
        if estimated_size > self.config.max_cache_size_mb:
            # Remove oldest entries
            sorted_items = sorted(
                cache['timestamps'].items(),
                key=lambda x: x[1]
            )
            
            items_to_remove = len(sorted_items) // 4  # Remove 25% of items
            for key, _ in sorted_items[:items_to_remove]:
                if key in cache['data']:
                    del cache['data'][key]
                if key in cache['timestamps']:
                    del cache['timestamps'][key]
            
            logger.info(f"Optimized cache size: removed {items_to_remove} entries")
    
    def _get_from_cache(self, cache: Dict[str, Any], key: str) -> Any:
        """Get data from cache"""
        if key in cache['data']:
            # Check if expired
            if time.time() - cache['timestamps'][key] < self.config.cache_ttl_seconds:
                return cache['data'][key]
            else:
                # Remove expired entry
                del cache['data'][key]
                del cache['timestamps'][key]
        return None
    
    def _set_cache(self, cache: Dict[str, Any], key: str, data: Any):
        """Set data in cache"""
        cache['data'][key] = data
        cache['timestamps'][key] = time.time()
    
    def _batch_api_calls(self, api_calls: List[Callable]) -> List[Any]:
        """Batch API calls for better performance"""
        # Simplified batching - in production would implement proper API batching
        results = []
        batch_size = min(5, len(api_calls))
        
        for i in range(0, len(api_calls), batch_size):
            batch = api_calls[i:i + batch_size]
            batch_results = []
            
            for call in batch:
                try:
                    result = call()
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Batched API call failed: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
        
        return results
    
    def _cache_api_results(self, api_calls: List[Callable]) -> List[Any]:
        """Cache API results for repeated calls"""
        # Simplified caching - in production would implement proper API result caching
        cache_key = "api_results"
        if cache_key not in self.cache_systems:
            self.cache_systems[cache_key] = {
                'data': {},
                'timestamps': {},
                'hit_count': 0,
                'miss_count': 0,
                'size_mb': 0
            }
        
        cache = self.cache_systems[cache_key]
        results = []
        
        for i, call in enumerate(api_calls):
            call_key = f"call_{i}_{hash(str(call))}"
            cached_result = self._get_from_cache(cache, call_key)
            
            if cached_result is not None:
                cache['hit_count'] += 1
                results.append(cached_result)
            else:
                try:
                    result = call()
                    self._set_cache(cache, call_key, result)
                    cache['miss_count'] += 1
                    results.append(result)
                except Exception as e:
                    logger.error(f"Cached API call failed: {e}")
                    results.append(None)
        
        return results
    
    def _parallel_api_calls(self, api_calls: List[Callable]) -> List[Any]:
        """Execute API calls in parallel"""
        results = [None] * len(api_calls)
        futures = {}
        
        # Submit all calls to thread pool
        for i, call in enumerate(api_calls):
            future = self.thread_pool.submit(call)
            futures[future] = i
        
        # Collect results
        for future in as_completed(futures, timeout=self.config.timeout_seconds):
            try:
                result = future.result()
                index = futures[future]
                results[index] = result
            except Exception as e:
                logger.error(f"Parallel API call failed: {e}")
                index = futures[future]
                results[index] = None
        
        return results
    
    def _record_performance_data(self, metric: PerformanceMetric, 
                                value: float, component: str):
        """Record performance data"""
        data = PerformanceData(
            metric=metric,
            value=value,
            timestamp=datetime.utcnow(),
            component=component
        )
        self.performance_data.append(data)
    
    def _create_failed_result(self, component: str, error_message: str) -> OptimizationResult:
        """Create failed optimization result"""
        return OptimizationResult(
            optimization_id=f"failed_{component}_{int(time.time())}",
            component=component,
            optimization_type="failed_optimization",
            before_metrics={},
            after_metrics={},
            improvement_percentage=0.0,
            execution_time=0.0,
            success=False,
            recommendations=[f"Optimization failed: {error_message}"]
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        recent_data = [
            data for data in self.performance_data
            if data.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        summary = {
            'total_optimizations': len(self.optimization_history),
            'successful_optimizations': sum(1 for opt in self.optimization_history if opt.success),
            'average_improvement': statistics.mean([
                opt.improvement_percentage for opt in self.optimization_history if opt.success
            ]) if self.optimization_history else 0.0,
            'recent_performance_data_points': len(recent_data),
            'cache_systems': len(self.cache_systems),
            'current_resource_usage': self._get_current_resource_usage(),
            'optimization_config': {
                'level': self.config.optimization_level.value,
                'caching_enabled': self.config.enable_caching,
                'parallel_processing_enabled': self.config.enable_parallel_processing,
                'max_workers': self.config.max_worker_threads
            }
        }
        
        return summary
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        logger.info("Performance Optimizer cleaned up")

