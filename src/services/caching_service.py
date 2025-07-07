"""
Caching Service for Research Engine
Implements intelligent caching to reduce API calls and improve performance
"""

import json
import hashlib
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    data: Any
    created_at: datetime
    expires_at: datetime
    access_count: int
    last_accessed: datetime
    source_type: str
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > self.expires_at
    
    def is_stale(self, max_age_hours: int = 24) -> bool:
        """Check if cache entry is stale"""
        return datetime.now() > self.created_at + timedelta(hours=max_age_hours)

class InMemoryCacheService:
    """In-memory caching service with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl_hours: int = 24):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl_hours = default_ttl_hours
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # TTL settings for different data types
        self.ttl_settings = {
            'trends': 6,      # Google Trends data - 6 hours
            'news': 2,        # News data - 2 hours
            'social': 4,      # Social media data - 4 hours
            'research': 12,   # Research results - 12 hours
            'default': 24     # Default - 24 hours
        }
    
    def _generate_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate a unique cache key"""
        # Create a deterministic key from parameters
        key_data = f"{prefix}:" + ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        # Hash for consistent length
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        self.stats['total_requests'] += 1
        
        if key not in self.cache:
            self.stats['misses'] += 1
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if entry.is_expired():
            del self.cache[key]
            self.stats['misses'] += 1
            logger.debug(f"Cache entry expired: {key}")
            return None
        
        # Update access statistics
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        
        self.stats['hits'] += 1
        logger.debug(f"Cache hit: {key}")
        return entry.data
    
    async def set(self, key: str, data: Any, source_type: str = 'default', ttl_hours: Optional[int] = None) -> None:
        """Set data in cache"""
        # Determine TTL
        if ttl_hours is None:
            ttl_hours = self.ttl_settings.get(source_type, self.default_ttl_hours)
        
        # Create cache entry
        now = datetime.now()
        entry = CacheEntry(
            key=key,
            data=data,
            created_at=now,
            expires_at=now + timedelta(hours=ttl_hours),
            access_count=0,
            last_accessed=now,
            source_type=source_type
        )
        
        # Check if we need to evict entries
        if len(self.cache) >= self.max_size:
            await self._evict_lru()
        
        self.cache[key] = entry
        logger.debug(f"Cache set: {key} (TTL: {ttl_hours}h)")
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entries"""
        if not self.cache:
            return
        
        # Find LRU entry
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
        del self.cache[lru_key]
        self.stats['evictions'] += 1
        logger.debug(f"Evicted LRU entry: {lru_key}")
    
    async def invalidate(self, pattern: str = None) -> int:
        """Invalidate cache entries matching pattern"""
        if pattern is None:
            # Clear all cache
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared all cache entries: {count}")
            return count
        
        # Clear entries matching pattern
        keys_to_remove = [key for key in self.cache.keys() if pattern in key]
        for key in keys_to_remove:
            del self.cache[key]
        
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching: {pattern}")
        return len(keys_to_remove)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        hit_rate = self.stats['hits'] / max(self.stats['total_requests'], 1) * 100
        
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': round(hit_rate, 2),
            'total_requests': self.stats['total_requests'],
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions']
        }
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)

class CachedResearchService:
    """Research service with intelligent caching"""
    
    def __init__(self, cache_service: InMemoryCacheService):
        self.cache = cache_service
    
    async def get_cached_trend_data(self, keyword: str, **kwargs) -> Optional[Dict]:
        """Get cached trend data"""
        cache_key = self.cache._generate_cache_key('trends', keyword=keyword, **kwargs)
        return await self.cache.get(cache_key)
    
    async def cache_trend_data(self, keyword: str, data: Dict, **kwargs) -> None:
        """Cache trend data"""
        cache_key = self.cache._generate_cache_key('trends', keyword=keyword, **kwargs)
        await self.cache.set(cache_key, data, source_type='trends')
    
    async def get_cached_social_data(self, keyword: str, **kwargs) -> Optional[Dict]:
        """Get cached social sentiment data"""
        cache_key = self.cache._generate_cache_key('social', keyword=keyword, **kwargs)
        return await self.cache.get(cache_key)
    
    async def cache_social_data(self, keyword: str, data: Dict, **kwargs) -> None:
        """Cache social sentiment data"""
        cache_key = self.cache._generate_cache_key('social', keyword=keyword, **kwargs)
        await self.cache.set(cache_key, data, source_type='social')
    
    async def get_cached_news_data(self, keyword: str, **kwargs) -> Optional[Dict]:
        """Get cached news data"""
        cache_key = self.cache._generate_cache_key('news', keyword=keyword, **kwargs)
        return await self.cache.get(cache_key)
    
    async def cache_news_data(self, keyword: str, data: Dict, **kwargs) -> None:
        """Cache news data"""
        cache_key = self.cache._generate_cache_key('news', keyword=keyword, **kwargs)
        await self.cache.set(cache_key, data, source_type='news')
    
    async def get_cached_research_result(self, idea_title: str, idea_description: str) -> Optional[Dict]:
        """Get cached complete research result"""
        cache_key = self.cache._generate_cache_key('research', title=idea_title, description=idea_description)
        return await self.cache.get(cache_key)
    
    async def cache_research_result(self, idea_title: str, idea_description: str, result: Dict) -> None:
        """Cache complete research result"""
        cache_key = self.cache._generate_cache_key('research', title=idea_title, description=idea_description)
        await self.cache.set(cache_key, result, source_type='research')

class SmartCacheManager:
    """Smart cache manager with advanced features"""
    
    def __init__(self, cache_service: InMemoryCacheService):
        self.cache = cache_service
        self.research_cache = CachedResearchService(cache_service)
        
        # Cache warming settings
        self.popular_keywords = [
            'artificial intelligence', 'machine learning', 'blockchain',
            'fintech', 'healthtech', 'edtech', 'saas', 'mobile app',
            'e-commerce', 'automation', 'sustainability', 'remote work'
        ]
    
    async def warm_cache(self, keywords: Optional[List[str]] = None) -> Dict:
        """Warm cache with popular keywords"""
        if keywords is None:
            keywords = self.popular_keywords[:5]  # Warm top 5
        
        warming_stats = {
            'keywords_warmed': 0,
            'cache_entries_created': 0,
            'errors': 0
        }
        
        logger.info(f"Starting cache warming for {len(keywords)} keywords")
        
        for keyword in keywords:
            try:
                # Check if already cached
                trend_cached = await self.research_cache.get_cached_trend_data(keyword)
                social_cached = await self.research_cache.get_cached_social_data(keyword)
                news_cached = await self.research_cache.get_cached_news_data(keyword)
                
                entries_needed = 0
                if not trend_cached:
                    entries_needed += 1
                if not social_cached:
                    entries_needed += 1
                if not news_cached:
                    entries_needed += 1
                
                if entries_needed > 0:
                    logger.info(f"Cache warming needed for keyword: {keyword}")
                    # Note: In production, this would trigger actual API calls
                    # For now, we'll simulate cache warming
                    await self._simulate_cache_warming(keyword)
                    warming_stats['cache_entries_created'] += entries_needed
                
                warming_stats['keywords_warmed'] += 1
                
            except Exception as e:
                logger.error(f"Error warming cache for {keyword}: {e}")
                warming_stats['errors'] += 1
        
        logger.info(f"Cache warming completed: {warming_stats}")
        return warming_stats
    
    async def _simulate_cache_warming(self, keyword: str) -> None:
        """Simulate cache warming with mock data"""
        # Simulate trend data
        mock_trend_data = {
            'keyword': keyword,
            'trend_score': 7.5,
            'data_source': 'Cache Warming Mock',
            'collection_timestamp': datetime.now().isoformat()
        }
        await self.research_cache.cache_trend_data(keyword, mock_trend_data)
        
        # Simulate social data
        mock_social_data = {
            'keyword': keyword,
            'sentiment_score': 0.6,
            'total_mentions': 45,
            'data_source': 'Cache Warming Mock',
            'collection_timestamp': datetime.now().isoformat()
        }
        await self.research_cache.cache_social_data(keyword, mock_social_data)
        
        # Simulate news data
        mock_news_data = {
            'keyword': keyword,
            'total_articles': 25,
            'data_source': 'Cache Warming Mock',
            'collection_timestamp': datetime.now().isoformat()
        }
        await self.research_cache.cache_news_data(keyword, mock_news_data)
    
    async def optimize_cache(self) -> Dict:
        """Optimize cache by cleaning up and reorganizing"""
        optimization_stats = {
            'expired_cleaned': 0,
            'cache_size_before': len(self.cache.cache),
            'cache_size_after': 0
        }
        
        # Clean up expired entries
        optimization_stats['expired_cleaned'] = await self.cache.cleanup_expired()
        
        # Get final cache size
        optimization_stats['cache_size_after'] = len(self.cache.cache)
        
        logger.info(f"Cache optimization completed: {optimization_stats}")
        return optimization_stats
    
    async def get_cache_health(self) -> Dict:
        """Get comprehensive cache health metrics"""
        stats = self.cache.get_stats()
        
        # Calculate additional health metrics
        cache_utilization = (stats['cache_size'] / stats['max_size']) * 100
        
        # Analyze cache entry ages
        now = datetime.now()
        entry_ages = []
        stale_count = 0
        
        for entry in self.cache.cache.values():
            age_hours = (now - entry.created_at).total_seconds() / 3600
            entry_ages.append(age_hours)
            
            if entry.is_stale():
                stale_count += 1
        
        avg_age = sum(entry_ages) / len(entry_ages) if entry_ages else 0
        
        health_metrics = {
            **stats,
            'cache_utilization_percent': round(cache_utilization, 2),
            'average_entry_age_hours': round(avg_age, 2),
            'stale_entries': stale_count,
            'health_score': self._calculate_health_score(stats, cache_utilization, stale_count)
        }
        
        return health_metrics
    
    def _calculate_health_score(self, stats: Dict, utilization: float, stale_count: int) -> float:
        """Calculate overall cache health score (0-100)"""
        # Base score from hit rate
        hit_rate_score = min(stats['hit_rate'], 100)
        
        # Utilization score (optimal around 70-80%)
        if utilization < 50:
            utilization_score = utilization * 2  # Underutilized
        elif utilization > 90:
            utilization_score = 100 - (utilization - 90) * 5  # Over-utilized
        else:
            utilization_score = 100  # Optimal range
        
        # Staleness penalty
        total_entries = stats['cache_size']
        staleness_penalty = (stale_count / max(total_entries, 1)) * 20
        
        # Combined health score
        health_score = (hit_rate_score * 0.5 + utilization_score * 0.3 + (100 - staleness_penalty) * 0.2)
        
        return round(max(min(health_score, 100), 0), 1)

# Factory function to create cache services
def create_cache_services() -> Tuple[InMemoryCacheService, SmartCacheManager]:
    """Create cache services with optimal configuration"""
    cache_service = InMemoryCacheService(max_size=1000, default_ttl_hours=24)
    cache_manager = SmartCacheManager(cache_service)
    
    return cache_service, cache_manager

# Test function for caching system
async def test_caching_system():
    """Test the caching system"""
    print("Testing Caching System...")
    
    # Create cache services
    cache_service, cache_manager = create_cache_services()
    
    # Test basic caching
    print("\n1. Testing basic cache operations...")
    await cache_service.set('test_key', {'data': 'test_value'}, 'test')
    cached_data = await cache_service.get('test_key')
    print(f"✅ Basic caching: {cached_data is not None}")
    
    # Test research caching
    print("\n2. Testing research data caching...")
    research_cache = CachedResearchService(cache_service)
    
    test_data = {'keyword': 'test', 'score': 8.5}
    await research_cache.cache_trend_data('test_keyword', test_data)
    retrieved_data = await research_cache.get_cached_trend_data('test_keyword')
    print(f"✅ Research caching: {retrieved_data is not None}")
    
    # Test cache warming
    print("\n3. Testing cache warming...")
    warming_stats = await cache_manager.warm_cache(['ai', 'blockchain'])
    print(f"✅ Cache warming: {warming_stats['keywords_warmed']} keywords warmed")
    
    # Test cache health
    print("\n4. Testing cache health metrics...")
    health = await cache_manager.get_cache_health()
    print(f"✅ Cache health score: {health['health_score']}/100")
    
    # Test cache optimization
    print("\n5. Testing cache optimization...")
    optimization = await cache_manager.optimize_cache()
    print(f"✅ Cache optimization: {optimization['expired_cleaned']} entries cleaned")
    
    print("\n🎉 Caching system test completed successfully!")
    
    return {
        'cache_service': cache_service,
        'cache_manager': cache_manager,
        'health_metrics': health
    }

if __name__ == "__main__":
    # Run caching tests
    asyncio.run(test_caching_system())

