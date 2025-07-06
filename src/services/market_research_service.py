import os
import json
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import re
from urllib.parse import quote_plus, urljoin
import random

class MarketResearchService:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1  # seconds between requests
        
        # Market data sources
        self.data_sources = {
            "search_trends": "https://trends.google.com/trends/api/explore",
            "news_api": "https://newsapi.org/v2/everything",
            "reddit_api": "https://www.reddit.com/search.json",
            "crunchbase": "https://www.crunchbase.com/discover/organization.companies",
            "product_hunt": "https://www.producthunt.com/search"
        }
    
    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def research_market_opportunity(self, business_idea: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive market research for a business idea"""
        try:
            keywords = self._extract_keywords(business_idea)
            
            research_data = {
                "search_analysis": self.analyze_search_trends(keywords),
                "competitive_landscape": self.analyze_competition(keywords, business_idea.get('name', '')),
                "market_sentiment": self.analyze_market_sentiment(keywords),
                "funding_landscape": self.analyze_funding_trends(keywords),
                "news_analysis": self.analyze_news_coverage(keywords),
                "social_validation": self.analyze_social_signals(keywords),
                "market_size_estimation": self.estimate_market_size(business_idea),
                "trend_analysis": self.analyze_market_trends(keywords),
                "research_timestamp": datetime.utcnow().isoformat()
            }
            
            # Calculate overall market score
            research_data["market_score"] = self._calculate_market_score(research_data)
            
            return research_data
            
        except Exception as e:
            print(f"Error in market research: {e}")
            return self._get_fallback_research_data()
    
    def _extract_keywords(self, business_idea: Dict[str, Any]) -> List[str]:
        """Extract relevant keywords from business idea"""
        keywords = []
        
        # Extract from various fields
        text_fields = [
            business_idea.get('name', ''),
            business_idea.get('tagline', ''),
            business_idea.get('problem_statement', ''),
            business_idea.get('ai_solution', ''),
            business_idea.get('target_audience', '')
        ]
        
        # Combine all text
        combined_text = ' '.join(text_fields).lower()
        
        # Extract meaningful keywords (simple approach)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
        
        # Filter out common words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Remove duplicates and take top keywords
        keywords = list(dict.fromkeys(keywords))[:10]
        
        return keywords
    
    def analyze_search_trends(self, keywords: List[str]) -> Dict[str, Any]:
        """Analyze search trends for keywords"""
        try:
            # Simulate Google Trends analysis
            # In production, use pytrends library
            
            trend_data = {}
            for keyword in keywords[:5]:  # Limit to top 5 keywords
                # Simulate trend data
                interest_over_time = [random.randint(20, 100) for _ in range(12)]  # 12 months
                
                trend_data[keyword] = {
                    "interest_score": sum(interest_over_time) / len(interest_over_time),
                    "trend_direction": self._calculate_trend_direction(interest_over_time),
                    "peak_month": interest_over_time.index(max(interest_over_time)) + 1,
                    "monthly_data": interest_over_time,
                    "related_queries": self._get_related_queries(keyword),
                    "geographic_interest": self._get_geographic_data(keyword)
                }
            
            return {
                "keyword_trends": trend_data,
                "overall_trend": self._calculate_overall_trend(trend_data),
                "search_volume_estimate": self._estimate_search_volume(keywords),
                "seasonality": self._detect_seasonality(trend_data)
            }
            
        except Exception as e:
            print(f"Error analyzing search trends: {e}")
            return self._get_fallback_search_data()
    
    def analyze_competition(self, keywords: List[str], business_name: str) -> Dict[str, Any]:
        """Analyze competitive landscape"""
        try:
            # Simulate competitive analysis
            # In production, scrape search results, analyze competitors
            
            competitors = []
            for i in range(random.randint(2, 8)):
                competitor = {
                    "name": f"Competitor {i+1}",
                    "website": f"https://competitor{i+1}.com",
                    "description": f"AI-powered solution for {random.choice(keywords)}",
                    "estimated_traffic": random.randint(1000, 100000),
                    "funding_status": random.choice(["Bootstrapped", "Seed", "Series A", "Series B", "Unknown"]),
                    "team_size": random.randint(1, 50),
                    "key_features": [f"Feature {j+1}" for j in range(random.randint(2, 5))],
                    "pricing_model": random.choice(["Freemium", "Subscription", "One-time", "Usage-based"]),
                    "strengths": [f"Strength {j+1}" for j in range(random.randint(1, 3))],
                    "weaknesses": [f"Weakness {j+1}" for j in range(random.randint(1, 3))]
                }
                competitors.append(competitor)
            
            return {
                "direct_competitors": competitors[:3],
                "indirect_competitors": competitors[3:6],
                "market_leaders": competitors[:2],
                "competitive_intensity": self._calculate_competitive_intensity(competitors),
                "market_gaps": self._identify_market_gaps(competitors, keywords),
                "competitive_advantages": self._suggest_competitive_advantages(competitors),
                "barrier_to_entry": random.choice(["Low", "Medium", "High"]),
                "market_saturation": random.choice(["Low", "Medium", "High"])
            }
            
        except Exception as e:
            print(f"Error analyzing competition: {e}")
            return self._get_fallback_competition_data()
    
    def analyze_market_sentiment(self, keywords: List[str]) -> Dict[str, Any]:
        """Analyze market sentiment from various sources"""
        try:
            # Simulate sentiment analysis
            # In production, use Twitter API, Reddit API, news sentiment
            
            sentiment_data = {}
            for keyword in keywords[:3]:
                sentiment_data[keyword] = {
                    "overall_sentiment": random.uniform(0.3, 0.9),
                    "positive_mentions": random.randint(100, 1000),
                    "negative_mentions": random.randint(10, 200),
                    "neutral_mentions": random.randint(200, 800),
                    "sentiment_trend": random.choice(["improving", "stable", "declining"]),
                    "key_themes": [f"Theme {i+1}" for i in range(random.randint(2, 4))]
                }
            
            return {
                "keyword_sentiment": sentiment_data,
                "overall_market_sentiment": sum(data["overall_sentiment"] for data in sentiment_data.values()) / len(sentiment_data),
                "sentiment_drivers": ["Market growth", "Technology advancement", "User adoption"],
                "risk_indicators": ["Regulatory concerns", "Competition", "Technology limitations"][:random.randint(1, 3)]
            }
            
        except Exception as e:
            print(f"Error analyzing market sentiment: {e}")
            return self._get_fallback_sentiment_data()
    
    def analyze_funding_trends(self, keywords: List[str]) -> Dict[str, Any]:
        """Analyze funding and investment trends"""
        try:
            # Simulate funding analysis
            # In production, use Crunchbase API, PitchBook data
            
            funding_data = {
                "recent_rounds": [],
                "total_funding_6months": random.randint(50, 500) * 1000000,
                "average_round_size": random.randint(1, 20) * 1000000,
                "number_of_deals": random.randint(10, 100),
                "top_investors": ["Andreessen Horowitz", "Sequoia Capital", "Y Combinator", "Accel", "GV"][:random.randint(2, 5)],
                "funding_stages": {
                    "seed": random.randint(5, 30),
                    "series_a": random.randint(2, 15),
                    "series_b": random.randint(1, 8),
                    "later_stage": random.randint(0, 5)
                },
                "geographic_distribution": {
                    "north_america": random.randint(40, 70),
                    "europe": random.randint(15, 30),
                    "asia": random.randint(10, 25),
                    "other": random.randint(5, 15)
                },
                "funding_trend": random.choice(["increasing", "stable", "decreasing"]),
                "investor_confidence": random.uniform(0.6, 0.9)
            }
            
            # Generate recent funding rounds
            for i in range(random.randint(3, 8)):
                round_data = {
                    "company": f"AI Startup {i+1}",
                    "amount": random.randint(1, 50) * 1000000,
                    "stage": random.choice(["Seed", "Series A", "Series B", "Series C"]),
                    "date": (datetime.now() - timedelta(days=random.randint(1, 180))).strftime("%Y-%m-%d"),
                    "investors": random.sample(funding_data["top_investors"], random.randint(1, 3))
                }
                funding_data["recent_rounds"].append(round_data)
            
            return funding_data
            
        except Exception as e:
            print(f"Error analyzing funding trends: {e}")
            return self._get_fallback_funding_data()
    
    def analyze_news_coverage(self, keywords: List[str]) -> Dict[str, Any]:
        """Analyze news coverage and media attention"""
        try:
            # Simulate news analysis
            # In production, use NewsAPI, Google News
            
            news_data = {
                "total_articles": random.randint(50, 500),
                "articles_last_30days": random.randint(10, 100),
                "media_sentiment": random.uniform(0.4, 0.8),
                "top_publications": ["TechCrunch", "VentureBeat", "Forbes", "Wired", "MIT Technology Review"][:random.randint(2, 5)],
                "trending_topics": keywords[:3],
                "coverage_trend": random.choice(["increasing", "stable", "decreasing"]),
                "key_narratives": [
                    "AI transformation in industry",
                    "Startup innovation",
                    "Market disruption",
                    "Technology adoption"
                ][:random.randint(2, 4)]
            }
            
            return news_data
            
        except Exception as e:
            print(f"Error analyzing news coverage: {e}")
            return self._get_fallback_news_data()
    
    def analyze_social_signals(self, keywords: List[str]) -> Dict[str, Any]:
        """Analyze social media signals and community interest"""
        try:
            # Simulate social media analysis
            # In production, use Twitter API, Reddit API, LinkedIn API
            
            social_data = {}
            platforms = ["twitter", "reddit", "linkedin", "hackernews"]
            
            for platform in platforms:
                social_data[platform] = {
                    "mentions": random.randint(100, 5000),
                    "engagement_rate": random.uniform(0.02, 0.15),
                    "sentiment_score": random.uniform(0.4, 0.8),
                    "trending_discussions": [f"Discussion {i+1}" for i in range(random.randint(2, 5))],
                    "influencer_mentions": random.randint(5, 50),
                    "community_size": random.randint(1000, 100000)
                }
            
            return {
                "platform_data": social_data,
                "overall_social_score": sum(data["sentiment_score"] for data in social_data.values()) / len(social_data),
                "viral_potential": random.uniform(0.3, 0.8),
                "community_engagement": random.choice(["High", "Medium", "Low"]),
                "social_proof_indicators": ["User testimonials", "Case studies", "Community discussions"][:random.randint(1, 3)]
            }
            
        except Exception as e:
            print(f"Error analyzing social signals: {e}")
            return self._get_fallback_social_data()
    
    def estimate_market_size(self, business_idea: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate total addressable market (TAM), serviceable addressable market (SAM), and serviceable obtainable market (SOM)"""
        try:
            # Simulate market size estimation
            # In production, use industry reports, government data, market research
            
            # Base TAM on industry and target audience
            target_audience = business_idea.get('target_audience', '').lower()
            
            if 'small business' in target_audience:
                tam_base = random.randint(50, 200) * 1000000000  # $50B-$200B
            elif 'enterprise' in target_audience:
                tam_base = random.randint(100, 500) * 1000000000  # $100B-$500B
            elif 'consumer' in target_audience:
                tam_base = random.randint(20, 100) * 1000000000  # $20B-$100B
            else:
                tam_base = random.randint(10, 50) * 1000000000  # $10B-$50B
            
            sam = tam_base * random.uniform(0.05, 0.20)  # 5-20% of TAM
            som = sam * random.uniform(0.01, 0.10)  # 1-10% of SAM
            
            return {
                "tam": int(tam_base),
                "sam": int(sam),
                "som": int(som),
                "market_growth_rate": random.uniform(0.05, 0.30),  # 5-30% annual growth
                "market_maturity": random.choice(["Emerging", "Growing", "Mature", "Declining"]),
                "geographic_breakdown": {
                    "north_america": random.uniform(0.30, 0.50),
                    "europe": random.uniform(0.20, 0.35),
                    "asia_pacific": random.uniform(0.15, 0.30),
                    "other": random.uniform(0.05, 0.15)
                },
                "customer_segments": [
                    {"segment": "Early Adopters", "size_percentage": random.uniform(0.10, 0.20)},
                    {"segment": "Mainstream", "size_percentage": random.uniform(0.60, 0.80)},
                    {"segment": "Laggards", "size_percentage": random.uniform(0.10, 0.20)}
                ]
            }
            
        except Exception as e:
            print(f"Error estimating market size: {e}")
            return self._get_fallback_market_size()
    
    def analyze_market_trends(self, keywords: List[str]) -> Dict[str, Any]:
        """Analyze current and emerging market trends"""
        try:
            # Simulate trend analysis
            # In production, use trend analysis tools, industry reports
            
            trends = {
                "emerging_trends": [
                    "AI automation adoption",
                    "No-code/low-code platforms",
                    "Remote work solutions",
                    "Sustainability focus",
                    "Personalization at scale"
                ][:random.randint(3, 5)],
                "declining_trends": [
                    "Manual processes",
                    "One-size-fits-all solutions",
                    "Desktop-only applications"
                ][:random.randint(1, 3)],
                "technology_trends": [
                    "Machine learning integration",
                    "API-first architecture",
                    "Cloud-native solutions",
                    "Mobile-first design"
                ][:random.randint(2, 4)],
                "market_drivers": [
                    "Digital transformation",
                    "Cost reduction pressure",
                    "Efficiency demands",
                    "Competitive advantage"
                ][:random.randint(2, 4)],
                "regulatory_trends": [
                    "Data privacy regulations",
                    "AI governance frameworks",
                    "Industry compliance requirements"
                ][:random.randint(1, 3)]
            }
            
            return trends
            
        except Exception as e:
            print(f"Error analyzing market trends: {e}")
            return self._get_fallback_trends()
    
    def _calculate_trend_direction(self, data_points: List[int]) -> str:
        """Calculate trend direction from data points"""
        if len(data_points) < 2:
            return "stable"
        
        recent_avg = sum(data_points[-3:]) / len(data_points[-3:])
        earlier_avg = sum(data_points[:3]) / len(data_points[:3])
        
        if recent_avg > earlier_avg * 1.1:
            return "rising"
        elif recent_avg < earlier_avg * 0.9:
            return "declining"
        else:
            return "stable"
    
    def _get_related_queries(self, keyword: str) -> List[str]:
        """Get related search queries"""
        # Simulate related queries
        prefixes = ["best", "how to", "free", "top", "cheap"]
        suffixes = ["software", "tool", "platform", "solution", "service"]
        
        related = []
        for prefix in prefixes[:2]:
            related.append(f"{prefix} {keyword}")
        for suffix in suffixes[:2]:
            related.append(f"{keyword} {suffix}")
        
        return related
    
    def _get_geographic_data(self, keyword: str) -> Dict[str, int]:
        """Get geographic interest data"""
        countries = ["United States", "United Kingdom", "Canada", "Germany", "Australia"]
        return {country: random.randint(20, 100) for country in countries}
    
    def _calculate_overall_trend(self, trend_data: Dict[str, Any]) -> str:
        """Calculate overall trend from keyword trends"""
        if not trend_data:
            return "stable"
        
        rising_count = sum(1 for data in trend_data.values() if data.get("trend_direction") == "rising")
        total_count = len(trend_data)
        
        if rising_count / total_count > 0.6:
            return "rising"
        elif rising_count / total_count < 0.3:
            return "declining"
        else:
            return "stable"
    
    def _estimate_search_volume(self, keywords: List[str]) -> Dict[str, int]:
        """Estimate search volume for keywords"""
        return {keyword: random.randint(1000, 50000) for keyword in keywords[:5]}
    
    def _detect_seasonality(self, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect seasonal patterns in trends"""
        return {
            "has_seasonality": random.choice([True, False]),
            "peak_months": random.sample(range(1, 13), random.randint(1, 3)),
            "seasonal_factor": random.uniform(1.1, 2.0)
        }
    
    def _calculate_competitive_intensity(self, competitors: List[Dict[str, Any]]) -> str:
        """Calculate competitive intensity"""
        num_competitors = len(competitors)
        if num_competitors <= 3:
            return "Low"
        elif num_competitors <= 7:
            return "Medium"
        else:
            return "High"
    
    def _identify_market_gaps(self, competitors: List[Dict[str, Any]], keywords: List[str]) -> List[str]:
        """Identify potential market gaps"""
        gaps = [
            "Affordable pricing tier",
            "Better user experience",
            "Industry-specific features",
            "Integration capabilities",
            "Mobile-first approach",
            "AI-powered automation",
            "Real-time analytics",
            "Compliance features"
        ]
        return random.sample(gaps, random.randint(2, 4))
    
    def _suggest_competitive_advantages(self, competitors: List[Dict[str, Any]]) -> List[str]:
        """Suggest potential competitive advantages"""
        advantages = [
            "Superior AI technology",
            "Better pricing model",
            "Faster implementation",
            "Industry expertise",
            "Better customer support",
            "More integrations",
            "Easier to use",
            "Better performance"
        ]
        return random.sample(advantages, random.randint(2, 4))
    
    def _calculate_market_score(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall market opportunity score"""
        try:
            # Weight different factors
            weights = {
                "search_trends": 0.20,
                "competition": 0.20,
                "sentiment": 0.15,
                "funding": 0.15,
                "news": 0.10,
                "social": 0.10,
                "market_size": 0.10
            }
            
            scores = {}
            
            # Search trends score
            search_data = research_data.get("search_analysis", {})
            overall_trend = search_data.get("overall_trend", "stable")
            scores["search_trends"] = 8.0 if overall_trend == "rising" else 6.0 if overall_trend == "stable" else 4.0
            
            # Competition score (lower competition = higher score)
            comp_data = research_data.get("competitive_landscape", {})
            comp_intensity = comp_data.get("competitive_intensity", "Medium")
            scores["competition"] = 8.0 if comp_intensity == "Low" else 6.0 if comp_intensity == "Medium" else 4.0
            
            # Sentiment score
            sentiment_data = research_data.get("market_sentiment", {})
            sentiment_score = sentiment_data.get("overall_market_sentiment", 0.6)
            scores["sentiment"] = sentiment_score * 10
            
            # Funding score
            funding_data = research_data.get("funding_landscape", {})
            funding_trend = funding_data.get("funding_trend", "stable")
            scores["funding"] = 8.0 if funding_trend == "increasing" else 6.0 if funding_trend == "stable" else 4.0
            
            # News score
            news_data = research_data.get("news_analysis", {})
            coverage_trend = news_data.get("coverage_trend", "stable")
            scores["news"] = 8.0 if coverage_trend == "increasing" else 6.0 if coverage_trend == "stable" else 4.0
            
            # Social score
            social_data = research_data.get("social_validation", {})
            social_score = social_data.get("overall_social_score", 0.6)
            scores["social"] = social_score * 10
            
            # Market size score
            market_data = research_data.get("market_size_estimation", {})
            tam = market_data.get("tam", 0)
            if tam > 10000000000:  # $10B+
                scores["market_size"] = 9.0
            elif tam > 1000000000:  # $1B+
                scores["market_size"] = 7.0
            elif tam > 100000000:  # $100M+
                scores["market_size"] = 5.0
            else:
                scores["market_size"] = 3.0
            
            # Calculate weighted total
            total_score = sum(scores[key] * weights[key] for key in scores if key in weights)
            
            return {
                "total_score": round(total_score, 2),
                "component_scores": scores,
                "grade": "A" if total_score >= 8.0 else "B" if total_score >= 7.0 else "C" if total_score >= 6.0 else "D",
                "recommendation": self._get_recommendation(total_score)
            }
            
        except Exception as e:
            print(f"Error calculating market score: {e}")
            return {"total_score": 6.0, "component_scores": {}, "grade": "C", "recommendation": "Needs more research"}
    
    def _get_recommendation(self, score: float) -> str:
        """Get recommendation based on market score"""
        if score >= 8.0:
            return "Strong market opportunity - proceed with confidence"
        elif score >= 7.0:
            return "Good market opportunity - validate key assumptions"
        elif score >= 6.0:
            return "Moderate opportunity - conduct deeper research"
        else:
            return "Weak market signals - consider pivoting"
    
    # Fallback data methods
    def _get_fallback_research_data(self) -> Dict[str, Any]:
        """Fallback research data if APIs fail"""
        return {
            "search_analysis": self._get_fallback_search_data(),
            "competitive_landscape": self._get_fallback_competition_data(),
            "market_sentiment": self._get_fallback_sentiment_data(),
            "funding_landscape": self._get_fallback_funding_data(),
            "news_analysis": self._get_fallback_news_data(),
            "social_validation": self._get_fallback_social_data(),
            "market_size_estimation": self._get_fallback_market_size(),
            "trend_analysis": self._get_fallback_trends(),
            "market_score": {"total_score": 6.0, "grade": "C", "recommendation": "Data unavailable - manual research needed"},
            "research_timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_fallback_search_data(self) -> Dict[str, Any]:
        return {
            "keyword_trends": {"general": {"interest_score": 50, "trend_direction": "stable"}},
            "overall_trend": "stable",
            "search_volume_estimate": {"general": 10000},
            "seasonality": {"has_seasonality": False}
        }
    
    def _get_fallback_competition_data(self) -> Dict[str, Any]:
        return {
            "direct_competitors": [],
            "indirect_competitors": [],
            "market_leaders": [],
            "competitive_intensity": "Medium",
            "market_gaps": ["Research needed"],
            "competitive_advantages": ["To be determined"],
            "barrier_to_entry": "Medium",
            "market_saturation": "Medium"
        }
    
    def _get_fallback_sentiment_data(self) -> Dict[str, Any]:
        return {
            "keyword_sentiment": {},
            "overall_market_sentiment": 0.6,
            "sentiment_drivers": ["Market research needed"],
            "risk_indicators": ["Data unavailable"]
        }
    
    def _get_fallback_funding_data(self) -> Dict[str, Any]:
        return {
            "recent_rounds": [],
            "total_funding_6months": 0,
            "average_round_size": 0,
            "number_of_deals": 0,
            "top_investors": ["Data unavailable"],
            "funding_stages": {"seed": 0, "series_a": 0, "series_b": 0, "later_stage": 0},
            "geographic_distribution": {},
            "funding_trend": "unknown",
            "investor_confidence": 0.5
        }
    
    def _get_fallback_news_data(self) -> Dict[str, Any]:
        return {
            "total_articles": 0,
            "articles_last_30days": 0,
            "media_sentiment": 0.5,
            "top_publications": ["Data unavailable"],
            "trending_topics": [],
            "coverage_trend": "unknown",
            "key_narratives": ["Research needed"]
        }
    
    def _get_fallback_social_data(self) -> Dict[str, Any]:
        return {
            "platform_data": {},
            "overall_social_score": 0.5,
            "viral_potential": 0.5,
            "community_engagement": "Unknown",
            "social_proof_indicators": ["Research needed"]
        }
    
    def _get_fallback_market_size(self) -> Dict[str, Any]:
        return {
            "tam": 1000000000,
            "sam": 100000000,
            "som": 10000000,
            "market_growth_rate": 0.10,
            "market_maturity": "Unknown",
            "geographic_breakdown": {},
            "customer_segments": []
        }
    
    def _get_fallback_trends(self) -> Dict[str, Any]:
        return {
            "emerging_trends": ["Research needed"],
            "declining_trends": ["Analysis pending"],
            "technology_trends": ["Data unavailable"],
            "market_drivers": ["Investigation required"],
            "regulatory_trends": ["Assessment needed"]
        }

