"""
Enhanced Research Engine for Multi-Source Data Collection
Implements automated research with source validation and quality scoring
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchSource:
    """Data structure for research sources"""
    url: str
    title: str
    content: str
    source_type: str
    credibility_score: float
    publication_date: Optional[datetime]
    relevance_score: float
    key_insights: List[str]

@dataclass
class ResearchResult:
    """Complete research result for a business idea"""
    idea_title: str
    sources: List[ResearchSource]
    market_analysis: Dict
    competitive_analysis: Dict
    financial_insights: Dict
    risk_assessment: Dict
    overall_quality_score: float
    research_duration: float

class SourceValidator:
    """Validates and scores research sources for credibility"""
    
    def __init__(self):
        self.trusted_domains = {
            'academic': ['edu', 'org', 'gov'],
            'news': ['reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com'],
            'industry': ['techcrunch.com', 'venturebeat.com', 'crunchbase.com'],
            'government': ['gov', 'europa.eu', 'oecd.org'],
            'research': ['statista.com', 'mckinsey.com', 'bcg.com']
        }
        
        self.domain_scores = {
            'edu': 9.0,
            'gov': 9.5,
            'org': 7.5,
            'reuters.com': 9.0,
            'bloomberg.com': 8.5,
            'wsj.com': 8.5,
            'techcrunch.com': 7.0,
            'crunchbase.com': 8.0,
            'statista.com': 8.5
        }
    
    def calculate_credibility_score(self, source: ResearchSource) -> float:
        """Calculate credibility score for a source"""
        score = 5.0  # Base score
        
        # Domain-based scoring
        domain = urlparse(source.url).netloc.lower()
        for trusted_domain, base_score in self.domain_scores.items():
            if trusted_domain in domain:
                score = base_score
                break
        
        # Recency bonus (newer sources get higher scores)
        if source.publication_date:
            days_old = (datetime.now() - source.publication_date).days
            if days_old <= 30:
                score += 1.0
            elif days_old <= 90:
                score += 0.5
            elif days_old <= 365:
                score += 0.2
        
        # Content quality indicators
        if len(source.content) > 1000:  # Substantial content
            score += 0.5
        if len(source.key_insights) > 2:  # Multiple insights
            score += 0.3
        
        return min(score, 10.0)  # Cap at 10.0
    
    def validate_source_diversity(self, sources: List[ResearchSource]) -> bool:
        """Ensure source diversity requirements are met"""
        domains = [urlparse(source.url).netloc for source in sources]
        unique_domains = set(domains)
        
        # Check domain diversity (max 2 sources per domain)
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if domain_counts[domain] > 2:
                return False
        
        # Check source type diversity (minimum 3 different types)
        source_types = set(source.source_type for source in sources)
        return len(source_types) >= 3

class GoogleTrendsCollector:
    """Collects market trend data from Google Trends"""
    
    def __init__(self):
        self.base_url = "https://trends.google.com/trends/api"
        self.session = None
    
    async def collect_trend_data(self, keyword: str) -> Dict:
        """Collect trend data for a keyword"""
        try:
            # Simulate Google Trends data collection
            # In production, use pytrends or similar library
            trend_data = {
                'keyword': keyword,
                'interest_over_time': self._generate_mock_trend_data(),
                'related_queries': self._generate_related_queries(keyword),
                'geographic_interest': self._generate_geographic_data(),
                'trend_score': self._calculate_trend_score(keyword)
            }
            
            logger.info(f"Collected trend data for: {keyword}")
            return trend_data
            
        except Exception as e:
            logger.error(f"Error collecting trend data for {keyword}: {e}")
            return {}
    
    def _generate_mock_trend_data(self) -> List[Dict]:
        """Generate mock trend data for development"""
        import random
        base_date = datetime.now() - timedelta(days=365)
        trend_data = []
        
        for i in range(52):  # Weekly data for 1 year
            date = base_date + timedelta(weeks=i)
            value = random.randint(20, 100)
            trend_data.append({
                'date': date.isoformat(),
                'value': value
            })
        
        return trend_data
    
    def _generate_related_queries(self, keyword: str) -> List[str]:
        """Generate related queries"""
        base_queries = [
            f"{keyword} market size",
            f"{keyword} industry trends",
            f"{keyword} startup ideas",
            f"{keyword} business opportunities",
            f"{keyword} competition analysis"
        ]
        return base_queries[:3]  # Return top 3
    
    def _generate_geographic_data(self) -> Dict:
        """Generate geographic interest data"""
        return {
            'United States': 100,
            'United Kingdom': 75,
            'Canada': 60,
            'Australia': 45,
            'Germany': 40
        }
    
    def _calculate_trend_score(self, keyword: str) -> float:
        """Calculate overall trend score"""
        import random
        return round(random.uniform(6.0, 9.5), 1)

class RedditCollector:
    """Collects social sentiment and discussion data from Reddit"""
    
    def __init__(self):
        self.base_url = "https://www.reddit.com"
        self.session = None
    
    async def collect_social_sentiment(self, keyword: str) -> Dict:
        """Collect social sentiment data"""
        try:
            # Simulate Reddit data collection
            # In production, use PRAW (Python Reddit API Wrapper)
            sentiment_data = {
                'keyword': keyword,
                'total_mentions': self._count_mentions(keyword),
                'sentiment_score': self._calculate_sentiment_score(),
                'top_discussions': self._get_top_discussions(keyword),
                'community_interest': self._assess_community_interest(),
                'trending_topics': self._get_trending_topics(keyword)
            }
            
            logger.info(f"Collected social sentiment for: {keyword}")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error collecting social sentiment for {keyword}: {e}")
            return {}
    
    def _count_mentions(self, keyword: str) -> int:
        """Count keyword mentions"""
        import random
        return random.randint(50, 500)
    
    def _calculate_sentiment_score(self) -> float:
        """Calculate sentiment score (-1 to 1)"""
        import random
        return round(random.uniform(0.2, 0.8), 2)
    
    def _get_top_discussions(self, keyword: str) -> List[Dict]:
        """Get top discussions"""
        discussions = [
            {
                'title': f"Discussion about {keyword} opportunities",
                'upvotes': 245,
                'comments': 67,
                'url': f"https://reddit.com/r/entrepreneur/discussion_{keyword.replace(' ', '_')}"
            },
            {
                'title': f"Market analysis for {keyword}",
                'upvotes': 189,
                'comments': 43,
                'url': f"https://reddit.com/r/business/analysis_{keyword.replace(' ', '_')}"
            }
        ]
        return discussions
    
    def _assess_community_interest(self) -> str:
        """Assess overall community interest level"""
        import random
        levels = ['Low', 'Moderate', 'High', 'Very High']
        return random.choice(levels)
    
    def _get_trending_topics(self, keyword: str) -> List[str]:
        """Get related trending topics"""
        return [
            f"{keyword} automation",
            f"{keyword} AI integration",
            f"{keyword} market trends"
        ]

class NewsCollector:
    """Collects industry news and analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.session = None
    
    async def collect_industry_news(self, keyword: str) -> Dict:
        """Collect industry news and analysis"""
        try:
            # Simulate news data collection
            # In production, use NewsAPI or similar service
            news_data = {
                'keyword': keyword,
                'total_articles': self._count_articles(keyword),
                'recent_articles': self._get_recent_articles(keyword),
                'industry_analysis': self._analyze_industry_coverage(keyword),
                'funding_news': self._get_funding_news(keyword),
                'market_insights': self._extract_market_insights(keyword)
            }
            
            logger.info(f"Collected industry news for: {keyword}")
            return news_data
            
        except Exception as e:
            logger.error(f"Error collecting industry news for {keyword}: {e}")
            return {}
    
    def _count_articles(self, keyword: str) -> int:
        """Count recent articles"""
        import random
        return random.randint(20, 150)
    
    def _get_recent_articles(self, keyword: str) -> List[Dict]:
        """Get recent articles"""
        articles = [
            {
                'title': f"Market Growth in {keyword} Sector Reaches New Heights",
                'source': 'TechCrunch',
                'published_at': (datetime.now() - timedelta(days=2)).isoformat(),
                'url': f"https://techcrunch.com/article_{keyword.replace(' ', '-')}",
                'summary': f"Analysis of recent growth trends in the {keyword} market."
            },
            {
                'title': f"Investment Opportunities in {keyword} Space",
                'source': 'VentureBeat',
                'published_at': (datetime.now() - timedelta(days=5)).isoformat(),
                'url': f"https://venturebeat.com/article_{keyword.replace(' ', '-')}",
                'summary': f"Exploring investment potential in {keyword} startups."
            }
        ]
        return articles
    
    def _analyze_industry_coverage(self, keyword: str) -> Dict:
        """Analyze industry coverage"""
        return {
            'coverage_volume': 'High',
            'sentiment_trend': 'Positive',
            'key_themes': [
                'Market expansion',
                'Technology innovation',
                'Investment growth'
            ]
        }
    
    def _get_funding_news(self, keyword: str) -> List[Dict]:
        """Get funding-related news"""
        return [
            {
                'company': f"{keyword.title()} Solutions Inc",
                'funding_round': 'Series A',
                'amount': '$5.2M',
                'date': (datetime.now() - timedelta(days=10)).isoformat()
            }
        ]
    
    def _extract_market_insights(self, keyword: str) -> List[str]:
        """Extract key market insights"""
        return [
            f"Growing demand for {keyword} solutions",
            f"Increasing investment in {keyword} technology",
            f"Market expansion expected to continue"
        ]

class ResearchEngine:
    """Main research engine coordinating all data collection"""
    
    def __init__(self):
        self.google_trends = GoogleTrendsCollector()
        self.reddit_collector = RedditCollector()
        self.news_collector = NewsCollector()
        self.source_validator = SourceValidator()
        
        # Rate limiting
        self.request_delays = {
            'google_trends': 2.0,
            'reddit': 1.0,
            'news': 1.5
        }
        
        self.last_request_times = {}
    
    async def research_business_idea(self, idea_title: str, idea_description: str) -> ResearchResult:
        """Conduct comprehensive research on a business idea"""
        start_time = time.time()
        logger.info(f"Starting research for: {idea_title}")
        
        try:
            # Extract keywords from idea
            keywords = self._extract_keywords(idea_title, idea_description)
            
            # Collect data from all sources in parallel
            research_tasks = [
                self._collect_trend_data(keywords),
                self._collect_social_data(keywords),
                self._collect_news_data(keywords)
            ]
            
            trend_data, social_data, news_data = await asyncio.gather(*research_tasks)
            
            # Generate research sources
            sources = self._generate_research_sources(trend_data, social_data, news_data)
            
            # Validate source quality and diversity
            validated_sources = self._validate_sources(sources)
            
            # Analyze collected data
            market_analysis = self._analyze_market_data(trend_data, social_data)
            competitive_analysis = self._analyze_competitive_landscape(news_data)
            financial_insights = self._extract_financial_insights(news_data, trend_data)
            risk_assessment = self._assess_risks(social_data, news_data)
            
            # Calculate overall quality score
            quality_score = self._calculate_research_quality(validated_sources, market_analysis)
            
            research_duration = time.time() - start_time
            
            result = ResearchResult(
                idea_title=idea_title,
                sources=validated_sources,
                market_analysis=market_analysis,
                competitive_analysis=competitive_analysis,
                financial_insights=financial_insights,
                risk_assessment=risk_assessment,
                overall_quality_score=quality_score,
                research_duration=research_duration
            )
            
            logger.info(f"Research completed for {idea_title} in {research_duration:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error during research for {idea_title}: {e}")
            raise
    
    def _extract_keywords(self, title: str, description: str) -> List[str]:
        """Extract relevant keywords for research"""
        # Simple keyword extraction - can be enhanced with NLP
        text = f"{title} {description}".lower()
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text)
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Return top keywords
        return keywords[:5]
    
    async def _collect_trend_data(self, keywords: List[str]) -> Dict:
        """Collect trend data with rate limiting"""
        await self._rate_limit('google_trends')
        
        trend_results = {}
        for keyword in keywords[:3]:  # Limit to top 3 keywords
            trend_data = await self.google_trends.collect_trend_data(keyword)
            trend_results[keyword] = trend_data
        
        return trend_results
    
    async def _collect_social_data(self, keywords: List[str]) -> Dict:
        """Collect social sentiment data with rate limiting"""
        await self._rate_limit('reddit')
        
        social_results = {}
        for keyword in keywords[:3]:
            social_data = await self.reddit_collector.collect_social_sentiment(keyword)
            social_results[keyword] = social_data
        
        return social_results
    
    async def _collect_news_data(self, keywords: List[str]) -> Dict:
        """Collect news data with rate limiting"""
        await self._rate_limit('news')
        
        news_results = {}
        for keyword in keywords[:3]:
            news_data = await self.news_collector.collect_industry_news(keyword)
            news_results[keyword] = news_data
        
        return news_results
    
    async def _rate_limit(self, service: str):
        """Implement rate limiting for API calls"""
        if service in self.last_request_times:
            time_since_last = time.time() - self.last_request_times[service]
            delay_needed = self.request_delays[service] - time_since_last
            
            if delay_needed > 0:
                await asyncio.sleep(delay_needed)
        
        self.last_request_times[service] = time.time()
    
    def _generate_research_sources(self, trend_data: Dict, social_data: Dict, news_data: Dict) -> List[ResearchSource]:
        """Generate research sources from collected data"""
        sources = []
        
        # Generate sources from news data
        for keyword, news in news_data.items():
            for article in news.get('recent_articles', []):
                source = ResearchSource(
                    url=article['url'],
                    title=article['title'],
                    content=article['summary'],
                    source_type='news',
                    credibility_score=0.0,  # Will be calculated by validator
                    publication_date=datetime.fromisoformat(article['published_at'].replace('Z', '+00:00')),
                    relevance_score=8.0,
                    key_insights=[article['summary']]
                )
                sources.append(source)
        
        # Generate sources from social data
        for keyword, social in social_data.items():
            for discussion in social.get('top_discussions', []):
                source = ResearchSource(
                    url=discussion['url'],
                    title=discussion['title'],
                    content=f"Community discussion with {discussion['upvotes']} upvotes and {discussion['comments']} comments",
                    source_type='social',
                    credibility_score=0.0,
                    publication_date=datetime.now() - timedelta(days=1),
                    relevance_score=7.0,
                    key_insights=[f"Community sentiment: {social.get('sentiment_score', 0.5)}"]
                )
                sources.append(source)
        
        # Generate sources from trend data
        for keyword, trends in trend_data.items():
            source = ResearchSource(
                url=f"https://trends.google.com/trends/explore?q={keyword}",
                title=f"Google Trends Analysis for {keyword}",
                content=f"Trend analysis showing {trends.get('trend_score', 'moderate')} interest",
                source_type='trends',
                credibility_score=0.0,
                publication_date=datetime.now(),
                relevance_score=8.5,
                key_insights=[f"Trend score: {trends.get('trend_score', 'N/A')}"]
            )
            sources.append(source)
        
        return sources
    
    def _validate_sources(self, sources: List[ResearchSource]) -> List[ResearchSource]:
        """Validate and score sources"""
        validated_sources = []
        
        for source in sources:
            # Calculate credibility score
            source.credibility_score = self.source_validator.calculate_credibility_score(source)
            
            # Only include sources with minimum credibility
            if source.credibility_score >= 6.0:
                validated_sources.append(source)
        
        # Ensure source diversity
        if not self.source_validator.validate_source_diversity(validated_sources):
            logger.warning("Source diversity requirements not met")
        
        # Sort by credibility and return top sources
        validated_sources.sort(key=lambda x: x.credibility_score, reverse=True)
        return validated_sources[:8]  # Return top 8 sources
    
    def _analyze_market_data(self, trend_data: Dict, social_data: Dict) -> Dict:
        """Analyze market data from trends and social sentiment"""
        market_analysis = {
            'market_interest_level': 'Moderate',
            'trend_direction': 'Growing',
            'social_sentiment': 'Positive',
            'market_size_indicators': [],
            'growth_potential': 'High',
            'key_insights': []
        }
        
        # Analyze trend data
        avg_trend_score = 0
        trend_count = 0
        
        for keyword, trends in trend_data.items():
            if 'trend_score' in trends:
                avg_trend_score += trends['trend_score']
                trend_count += 1
        
        if trend_count > 0:
            avg_trend_score /= trend_count
            if avg_trend_score > 8.0:
                market_analysis['market_interest_level'] = 'High'
            elif avg_trend_score > 6.0:
                market_analysis['market_interest_level'] = 'Moderate'
            else:
                market_analysis['market_interest_level'] = 'Low'
        
        # Analyze social sentiment
        avg_sentiment = 0
        sentiment_count = 0
        
        for keyword, social in social_data.items():
            if 'sentiment_score' in social:
                avg_sentiment += social['sentiment_score']
                sentiment_count += 1
        
        if sentiment_count > 0:
            avg_sentiment /= sentiment_count
            if avg_sentiment > 0.6:
                market_analysis['social_sentiment'] = 'Very Positive'
            elif avg_sentiment > 0.3:
                market_analysis['social_sentiment'] = 'Positive'
            elif avg_sentiment > 0:
                market_analysis['social_sentiment'] = 'Neutral'
            else:
                market_analysis['social_sentiment'] = 'Negative'
        
        return market_analysis
    
    def _analyze_competitive_landscape(self, news_data: Dict) -> Dict:
        """Analyze competitive landscape from news data"""
        competitive_analysis = {
            'competition_level': 'Moderate',
            'key_competitors': [],
            'market_gaps': [],
            'competitive_advantages': [],
            'barriers_to_entry': []
        }
        
        # Analyze funding news to assess competition
        total_funding_mentions = 0
        for keyword, news in news_data.items():
            funding_news = news.get('funding_news', [])
            total_funding_mentions += len(funding_news)
            
            for funding in funding_news:
                competitive_analysis['key_competitors'].append(funding['company'])
        
        # Assess competition level based on funding activity
        if total_funding_mentions > 3:
            competitive_analysis['competition_level'] = 'High'
        elif total_funding_mentions > 1:
            competitive_analysis['competition_level'] = 'Moderate'
        else:
            competitive_analysis['competition_level'] = 'Low'
        
        return competitive_analysis
    
    def _extract_financial_insights(self, news_data: Dict, trend_data: Dict) -> Dict:
        """Extract financial insights from research data"""
        financial_insights = {
            'market_size_estimate': 'Unknown',
            'funding_activity': 'Moderate',
            'revenue_potential': 'Medium',
            'cost_structure': [],
            'monetization_models': [],
            'financial_risks': []
        }
        
        # Analyze funding data
        total_funding = 0
        funding_rounds = 0
        
        for keyword, news in news_data.items():
            for funding in news.get('funding_news', []):
                # Extract funding amount (simplified)
                amount_str = funding.get('amount', '$0')
                try:
                    amount = float(amount_str.replace('$', '').replace('M', ''))
                    total_funding += amount
                    funding_rounds += 1
                except:
                    pass
        
        if funding_rounds > 0:
            avg_funding = total_funding / funding_rounds
            if avg_funding > 10:
                financial_insights['funding_activity'] = 'High'
            elif avg_funding > 2:
                financial_insights['funding_activity'] = 'Moderate'
            else:
                financial_insights['funding_activity'] = 'Low'
        
        return financial_insights
    
    def _assess_risks(self, social_data: Dict, news_data: Dict) -> Dict:
        """Assess risks based on research data"""
        risk_assessment = {
            'overall_risk_level': 'Medium',
            'market_risks': [],
            'competitive_risks': [],
            'technical_risks': [],
            'regulatory_risks': [],
            'mitigation_strategies': []
        }
        
        # Assess based on social sentiment
        negative_sentiment_count = 0
        for keyword, social in social_data.items():
            sentiment = social.get('sentiment_score', 0.5)
            if sentiment < 0.3:
                negative_sentiment_count += 1
        
        if negative_sentiment_count > 1:
            risk_assessment['market_risks'].append('Negative social sentiment detected')
        
        # Assess based on competition
        high_competition_indicators = 0
        for keyword, news in news_data.items():
            if len(news.get('funding_news', [])) > 2:
                high_competition_indicators += 1
        
        if high_competition_indicators > 0:
            risk_assessment['competitive_risks'].append('High competition with significant funding')
        
        return risk_assessment
    
    def _calculate_research_quality(self, sources: List[ResearchSource], market_analysis: Dict) -> float:
        """Calculate overall research quality score"""
        if not sources:
            return 0.0
        
        # Average source credibility
        avg_credibility = sum(source.credibility_score for source in sources) / len(sources)
        
        # Source count bonus
        source_count_score = min(len(sources) / 8.0, 1.0) * 10
        
        # Source diversity bonus
        source_types = set(source.source_type for source in sources)
        diversity_score = min(len(source_types) / 3.0, 1.0) * 10
        
        # Weighted average
        quality_score = (avg_credibility * 0.6 + source_count_score * 0.2 + diversity_score * 0.2)
        
        return round(quality_score, 1)

# Example usage and testing
async def test_research_engine():
    """Test the research engine with a sample idea"""
    engine = ResearchEngine()
    
    idea_title = "AI-Powered Personal Finance Assistant"
    idea_description = "A mobile app that uses machine learning to provide personalized financial advice and budgeting recommendations"
    
    try:
        result = await engine.research_business_idea(idea_title, idea_description)
        
        print(f"Research completed for: {result.idea_title}")
        print(f"Sources found: {len(result.sources)}")
        print(f"Quality score: {result.overall_quality_score}")
        print(f"Research duration: {result.research_duration:.2f} seconds")
        
        return result
        
    except Exception as e:
        print(f"Research failed: {e}")
        return None

if __name__ == "__main__":
    # Run test
    asyncio.run(test_research_engine())

