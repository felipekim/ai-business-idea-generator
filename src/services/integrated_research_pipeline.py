"""
Integrated Research Pipeline
Combines all Phase 3 components into a unified research automation system
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Import our custom services
from .real_api_integrations import create_real_api_collectors
from .caching_service import create_cache_services
from .validation_engine import create_validation_services

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchRequest:
    """Research request parameters"""
    idea_title: str
    idea_description: str
    target_sources: int = 6
    research_depth: str = 'moderate'  # 'light', 'moderate', 'deep'
    priority_areas: List[str] = None
    custom_keywords: List[str] = None

@dataclass
class ResearchResult:
    """Complete research result"""
    idea_title: str
    idea_description: str
    research_summary: Dict
    sources: List[Dict]
    validation_report: Dict
    confidence_score: float
    research_quality: str
    recommendations: List[str]
    processing_time: float
    cache_hit_rate: float
    metadata: Dict

class IntegratedResearchPipeline:
    """Main research pipeline orchestrating all components"""
    
    def __init__(self):
        # Initialize all services
        self.api_collectors = create_real_api_collectors()
        self.cache_service, self.cache_manager = create_cache_services()
        self.validation_engine = create_validation_services()
        
        # Pipeline statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0,
            'cache_hit_rate': 0.0
        }
        
        logger.info("Integrated Research Pipeline initialized successfully")
    
    async def conduct_research(self, request: ResearchRequest) -> ResearchResult:
        """Conduct comprehensive research for a business idea"""
        start_time = datetime.now()
        self.stats['total_requests'] += 1
        
        try:
            logger.info(f"Starting research for: {request.idea_title}")
            
            # Step 1: Check cache for existing research
            cached_result = await self._check_cache(request)
            if cached_result:
                logger.info("Found cached research result")
                return cached_result
            
            # Step 2: Generate research keywords
            keywords = await self._generate_research_keywords(request)
            
            # Step 3: Collect data from multiple sources
            research_data = await self._collect_research_data(keywords, request.target_sources)
            
            # Step 4: Validate and enhance data quality
            validation_report = await self._validate_research_quality(research_data)
            
            # Step 5: Synthesize research findings
            research_summary = await self._synthesize_research_findings(research_data, request)
            
            # Step 6: Generate recommendations
            recommendations = await self._generate_recommendations(research_data, validation_report)
            
            # Step 7: Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            cache_stats = self.cache_service.get_stats()
            
            # Create result
            result = ResearchResult(
                idea_title=request.idea_title,
                idea_description=request.idea_description,
                research_summary=research_summary,
                sources=research_data.get('sources', []),
                validation_report=asdict(validation_report),
                confidence_score=validation_report.confidence_score,
                research_quality=self._determine_quality_level(validation_report.confidence_score),
                recommendations=recommendations,
                processing_time=processing_time,
                cache_hit_rate=cache_stats['hit_rate'],
                metadata={
                    'keywords_used': keywords,
                    'research_depth': request.research_depth,
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_version': '3.0'
                }
            )
            
            # Step 8: Cache the result
            await self._cache_result(request, result)
            
            self.stats['successful_requests'] += 1
            logger.info(f"Research completed successfully in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Research failed for {request.idea_title}: {e}")
            raise
    
    async def _check_cache(self, request: ResearchRequest) -> Optional[ResearchResult]:
        """Check cache for existing research"""
        cache_key = f"research:{request.idea_title}:{request.idea_description}"
        cached_data = await self.cache_service.get(cache_key)
        
        if cached_data:
            # Convert cached data back to ResearchResult
            return ResearchResult(**cached_data)
        
        return None
    
    async def _generate_research_keywords(self, request: ResearchRequest) -> List[str]:
        """Generate research keywords from the business idea"""
        keywords = []
        
        # Extract keywords from title and description
        title_words = request.idea_title.lower().split()
        desc_words = request.idea_description.lower().split()
        
        # Filter and combine
        all_words = title_words + desc_words
        filtered_words = [word for word in all_words if len(word) > 3]
        
        # Add custom keywords if provided
        if request.custom_keywords:
            keywords.extend(request.custom_keywords)
        
        # Add extracted keywords
        keywords.extend(filtered_words[:5])  # Top 5 words
        
        # Add business-related keywords
        business_keywords = [
            f"{request.idea_title} market",
            f"{request.idea_title} business",
            f"{request.idea_title} industry",
            f"{request.idea_title} startup"
        ]
        keywords.extend(business_keywords)
        
        # Remove duplicates and limit
        unique_keywords = list(set(keywords))[:8]
        
        logger.info(f"Generated {len(unique_keywords)} research keywords")
        return unique_keywords
    
    async def _collect_research_data(self, keywords: List[str], target_sources: int) -> Dict:
        """Collect research data from multiple sources"""
        research_data = {
            'sources': [],
            'trends_data': {},
            'social_data': {},
            'news_data': {},
            'collection_metadata': {
                'keywords_used': keywords,
                'target_sources': target_sources,
                'collection_timestamp': datetime.now().isoformat()
            }
        }
        
        # Collect data for primary keywords (limit to 3 for performance)
        primary_keywords = keywords[:3]
        
        for keyword in primary_keywords:
            try:
                # Collect trends data
                trends_data = await self.api_collectors['google_trends'].collect_trend_data(keyword)
                research_data['trends_data'][keyword] = trends_data
                
                # Add trends as sources
                if trends_data:
                    research_data['sources'].append({
                        'url': 'https://trends.google.com',
                        'title': f"Google Trends: {keyword}",
                        'content': f"Trend analysis for {keyword}",
                        'source_type': 'trends',
                        'credibility_score': 8.0,
                        'publication_date': datetime.now().isoformat(),
                        'data': trends_data
                    })
                
                # Collect social data
                social_data = await self.api_collectors['reddit'].collect_social_sentiment(keyword)
                research_data['social_data'][keyword] = social_data
                
                # Add social discussions as sources
                if social_data and social_data.get('top_discussions'):
                    for discussion in social_data['top_discussions'][:2]:  # Top 2 discussions
                        research_data['sources'].append({
                            'url': discussion.get('url', ''),
                            'title': discussion.get('title', ''),
                            'content': f"Social discussion about {keyword}",
                            'source_type': 'social',
                            'credibility_score': 6.0,
                            'publication_date': discussion.get('created_utc', datetime.now().isoformat()),
                            'data': discussion
                        })
                
                # Collect news data
                news_data = await self.api_collectors['news'].collect_industry_news(keyword)
                research_data['news_data'][keyword] = news_data
                
                # Add news articles as sources
                if news_data and news_data.get('recent_articles'):
                    for article in news_data['recent_articles'][:2]:  # Top 2 articles
                        research_data['sources'].append({
                            'url': article.get('url', ''),
                            'title': article.get('title', ''),
                            'content': article.get('summary', ''),
                            'source_type': 'news',
                            'credibility_score': 7.5,
                            'publication_date': article.get('published_at', datetime.now().isoformat()),
                            'data': article
                        })
                
                # Rate limiting between keywords
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.warning(f"Error collecting data for keyword '{keyword}': {e}")
                continue
        
        logger.info(f"Collected {len(research_data['sources'])} sources from {len(primary_keywords)} keywords")
        return research_data
    
    async def _validate_research_quality(self, research_data: Dict) -> object:
        """Validate research quality using validation engine"""
        try:
            validation_result = self.validation_engine.validate_research_quality(research_data)
            logger.info(f"Research validation completed: {validation_result.confidence_score}/10")
            return validation_result
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            # Return default validation result
            from .validation_engine import ValidationResult
            return ValidationResult(
                is_valid=True,
                confidence_score=5.0,
                validation_issues=['Validation system error'],
                validation_details={},
                recommendations=['Manual review recommended']
            )
    
    async def _synthesize_research_findings(self, research_data: Dict, request: ResearchRequest) -> Dict:
        """Synthesize research findings into structured summary"""
        summary = {
            'market_analysis': self._analyze_market_trends(research_data),
            'competitive_landscape': self._analyze_competition(research_data),
            'social_sentiment': self._analyze_social_sentiment(research_data),
            'industry_insights': self._analyze_industry_news(research_data),
            'key_findings': self._extract_key_findings(research_data),
            'risk_factors': self._identify_risk_factors(research_data),
            'opportunities': self._identify_opportunities(research_data)
        }
        
        logger.info("Research findings synthesized successfully")
        return summary
    
    def _analyze_market_trends(self, research_data: Dict) -> Dict:
        """Analyze market trends from collected data"""
        trends_data = research_data.get('trends_data', {})
        
        analysis = {
            'trend_direction': 'stable',
            'growth_indicators': [],
            'market_interest': 'moderate',
            'geographic_distribution': {},
            'trend_score': 5.0
        }
        
        if trends_data:
            # Aggregate trend scores
            trend_scores = []
            for keyword, data in trends_data.items():
                if 'trend_score' in data:
                    trend_scores.append(data['trend_score'])
                
                # Extract growth indicators
                if 'related_queries' in data:
                    analysis['growth_indicators'].extend(data['related_queries'][:3])
            
            if trend_scores:
                analysis['trend_score'] = sum(trend_scores) / len(trend_scores)
                
                if analysis['trend_score'] > 7.0:
                    analysis['trend_direction'] = 'growing'
                    analysis['market_interest'] = 'high'
                elif analysis['trend_score'] > 5.0:
                    analysis['trend_direction'] = 'stable'
                    analysis['market_interest'] = 'moderate'
                else:
                    analysis['trend_direction'] = 'declining'
                    analysis['market_interest'] = 'low'
        
        return analysis
    
    def _analyze_competition(self, research_data: Dict) -> Dict:
        """Analyze competitive landscape"""
        news_data = research_data.get('news_data', {})
        
        analysis = {
            'competition_level': 'moderate',
            'key_players': [],
            'funding_activity': [],
            'market_gaps': [],
            'competitive_intensity': 5.0
        }
        
        if news_data:
            # Extract funding information
            for keyword, data in news_data.items():
                if 'funding_news' in data:
                    analysis['funding_activity'].extend(data['funding_news'])
                
                # Extract key themes as potential competitors
                if 'industry_analysis' in data and 'key_themes' in data['industry_analysis']:
                    analysis['key_players'].extend(data['industry_analysis']['key_themes'][:3])
        
        # Determine competition level based on funding activity
        if len(analysis['funding_activity']) > 3:
            analysis['competition_level'] = 'high'
            analysis['competitive_intensity'] = 8.0
        elif len(analysis['funding_activity']) > 1:
            analysis['competition_level'] = 'moderate'
            analysis['competitive_intensity'] = 6.0
        else:
            analysis['competition_level'] = 'low'
            analysis['competitive_intensity'] = 3.0
        
        return analysis
    
    def _analyze_social_sentiment(self, research_data: Dict) -> Dict:
        """Analyze social sentiment"""
        social_data = research_data.get('social_data', {})
        
        analysis = {
            'overall_sentiment': 'neutral',
            'community_interest': 'moderate',
            'discussion_volume': 0,
            'sentiment_score': 0.5,
            'key_topics': []
        }
        
        if social_data:
            sentiment_scores = []
            total_mentions = 0
            
            for keyword, data in social_data.items():
                if 'sentiment_score' in data:
                    sentiment_scores.append(data['sentiment_score'])
                
                if 'total_mentions' in data:
                    total_mentions += data['total_mentions']
                
                if 'trending_topics' in data:
                    analysis['key_topics'].extend(data['trending_topics'])
            
            if sentiment_scores:
                analysis['sentiment_score'] = sum(sentiment_scores) / len(sentiment_scores)
                
                if analysis['sentiment_score'] > 0.6:
                    analysis['overall_sentiment'] = 'positive'
                elif analysis['sentiment_score'] > 0.4:
                    analysis['overall_sentiment'] = 'neutral'
                else:
                    analysis['overall_sentiment'] = 'negative'
            
            analysis['discussion_volume'] = total_mentions
            
            if total_mentions > 100:
                analysis['community_interest'] = 'high'
            elif total_mentions > 30:
                analysis['community_interest'] = 'moderate'
            else:
                analysis['community_interest'] = 'low'
        
        return analysis
    
    def _analyze_industry_news(self, research_data: Dict) -> Dict:
        """Analyze industry news and insights"""
        news_data = research_data.get('news_data', {})
        
        analysis = {
            'industry_momentum': 'stable',
            'recent_developments': [],
            'market_signals': [],
            'coverage_volume': 'moderate'
        }
        
        if news_data:
            total_articles = 0
            positive_signals = 0
            
            for keyword, data in news_data.items():
                if 'total_articles' in data:
                    total_articles += data['total_articles']
                
                if 'recent_articles' in data:
                    analysis['recent_developments'].extend([
                        article['title'] for article in data['recent_articles'][:2]
                    ])
                
                if 'market_insights' in data:
                    analysis['market_signals'].extend(data['market_insights'][:3])
                
                # Check industry analysis sentiment
                if 'industry_analysis' in data:
                    industry_analysis = data['industry_analysis']
                    if industry_analysis.get('sentiment_trend') == 'Positive':
                        positive_signals += 1
            
            # Determine coverage volume
            if total_articles > 50:
                analysis['coverage_volume'] = 'high'
            elif total_articles > 20:
                analysis['coverage_volume'] = 'moderate'
            else:
                analysis['coverage_volume'] = 'low'
            
            # Determine industry momentum
            if positive_signals > len(news_data) * 0.6:
                analysis['industry_momentum'] = 'accelerating'
            elif positive_signals > len(news_data) * 0.3:
                analysis['industry_momentum'] = 'stable'
            else:
                analysis['industry_momentum'] = 'slowing'
        
        return analysis
    
    def _extract_key_findings(self, research_data: Dict) -> List[str]:
        """Extract key findings from all data sources"""
        findings = []
        
        # From trends data
        trends_data = research_data.get('trends_data', {})
        for keyword, data in trends_data.items():
            if data.get('trend_score', 0) > 7.0:
                findings.append(f"Strong market interest in {keyword}")
        
        # From social data
        social_data = research_data.get('social_data', {})
        for keyword, data in social_data.items():
            if data.get('sentiment_score', 0) > 0.6:
                findings.append(f"Positive community sentiment around {keyword}")
        
        # From news data
        news_data = research_data.get('news_data', {})
        for keyword, data in news_data.items():
            if data.get('total_articles', 0) > 30:
                findings.append(f"High media coverage for {keyword}")
        
        return findings[:5]  # Top 5 findings
    
    def _identify_risk_factors(self, research_data: Dict) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        # Check for negative sentiment
        social_data = research_data.get('social_data', {})
        for keyword, data in social_data.items():
            if data.get('sentiment_score', 0.5) < 0.4:
                risks.append(f"Negative sentiment detected for {keyword}")
        
        # Check for declining trends
        trends_data = research_data.get('trends_data', {})
        for keyword, data in trends_data.items():
            if data.get('trend_score', 5.0) < 4.0:
                risks.append(f"Declining interest trend for {keyword}")
        
        # Check for high competition
        news_data = research_data.get('news_data', {})
        total_funding = sum(len(data.get('funding_news', [])) for data in news_data.values())
        if total_funding > 5:
            risks.append("High competition with significant funding activity")
        
        return risks[:3]  # Top 3 risks
    
    def _identify_opportunities(self, research_data: Dict) -> List[str]:
        """Identify potential opportunities"""
        opportunities = []
        
        # Check for growing trends
        trends_data = research_data.get('trends_data', {})
        for keyword, data in trends_data.items():
            if data.get('trend_score', 5.0) > 7.0:
                opportunities.append(f"Growing market opportunity in {keyword}")
        
        # Check for positive sentiment with low competition
        social_data = research_data.get('social_data', {})
        news_data = research_data.get('news_data', {})
        
        for keyword in social_data.keys():
            social_score = social_data[keyword].get('sentiment_score', 0.5)
            funding_count = len(news_data.get(keyword, {}).get('funding_news', []))
            
            if social_score > 0.6 and funding_count < 2:
                opportunities.append(f"Positive sentiment with low competition in {keyword}")
        
        return opportunities[:3]  # Top 3 opportunities
    
    async def _generate_recommendations(self, research_data: Dict, validation_report: object) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Add validation recommendations
        recommendations.extend(validation_report.recommendations[:3])
        
        # Add research-based recommendations
        market_analysis = research_data.get('market_analysis', {})
        
        # Source diversity recommendations
        source_count = len(research_data.get('sources', []))
        if source_count < 5:
            recommendations.append("Gather additional sources for more comprehensive analysis")
        
        # Market trend recommendations
        trends_data = research_data.get('trends_data', {})
        avg_trend_score = 5.0
        if trends_data:
            trend_scores = [data.get('trend_score', 5.0) for data in trends_data.values()]
            avg_trend_score = sum(trend_scores) / len(trend_scores)
        
        if avg_trend_score > 7.0:
            recommendations.append("Consider rapid market entry due to strong trend indicators")
        elif avg_trend_score < 4.0:
            recommendations.append("Reassess market timing due to declining trend indicators")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _determine_quality_level(self, confidence_score: float) -> str:
        """Determine research quality level"""
        if confidence_score >= 9.0:
            return 'excellent'
        elif confidence_score >= 7.0:
            return 'good'
        elif confidence_score >= 5.0:
            return 'acceptable'
        else:
            return 'poor'
    
    async def _cache_result(self, request: ResearchRequest, result: ResearchResult) -> None:
        """Cache research result"""
        cache_key = f"research:{request.idea_title}:{request.idea_description}"
        await self.cache_service.set(cache_key, asdict(result), 'research')
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics"""
        cache_stats = self.cache_service.get_stats()
        
        return {
            **self.stats,
            'cache_stats': cache_stats,
            'success_rate': (self.stats['successful_requests'] / max(self.stats['total_requests'], 1)) * 100
        }

# Factory function to create research pipeline
def create_research_pipeline() -> IntegratedResearchPipeline:
    """Create integrated research pipeline"""
    return IntegratedResearchPipeline()

# Test function for integrated pipeline
async def test_integrated_pipeline():
    """Test the integrated research pipeline"""
    print("Testing Integrated Research Pipeline...")
    
    # Create pipeline
    pipeline = create_research_pipeline()
    
    # Create test request
    test_request = ResearchRequest(
        idea_title="AI-Powered Personal Finance Assistant",
        idea_description="An AI application that helps users manage their personal finances, track expenses, and provide investment recommendations.",
        target_sources=6,
        research_depth='moderate',
        custom_keywords=['fintech', 'personal finance', 'AI assistant']
    )
    
    print(f"\n🔍 Researching: {test_request.idea_title}")
    
    # Conduct research
    result = await pipeline.conduct_research(test_request)
    
    # Display results
    print(f"\n📊 Research Results:")
    print(f"✅ Confidence Score: {result.confidence_score}/10")
    print(f"✅ Research Quality: {result.research_quality}")
    print(f"✅ Sources Collected: {len(result.sources)}")
    print(f"✅ Processing Time: {result.processing_time:.2f}s")
    print(f"✅ Cache Hit Rate: {result.cache_hit_rate}%")
    print(f"✅ Validation Passed: {result.validation_report['is_valid']}")
    print(f"✅ Issues Found: {len(result.validation_report['validation_issues'])}")
    print(f"✅ Recommendations: {len(result.recommendations)}")
    
    # Display key findings
    if result.research_summary.get('key_findings'):
        print(f"\n🔍 Key Findings:")
        for finding in result.research_summary['key_findings']:
            print(f"  • {finding}")
    
    # Display recommendations
    if result.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in result.recommendations[:3]:
            print(f"  • {rec}")
    
    # Get pipeline stats
    stats = pipeline.get_pipeline_stats()
    print(f"\n📈 Pipeline Statistics:")
    print(f"✅ Success Rate: {stats['success_rate']:.1f}%")
    print(f"✅ Cache Hit Rate: {stats['cache_stats']['hit_rate']:.1f}%")
    
    print("\n🎉 Integrated pipeline test completed successfully!")
    
    return result

if __name__ == "__main__":
    # Run integrated pipeline test
    asyncio.run(test_integrated_pipeline())

