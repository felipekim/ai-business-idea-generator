import os
import json
import random
from datetime import datetime
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class AIService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4"
        self.backup_model = "gpt-3.5-turbo"
        
        # Niche rotation for diverse ideas
        self.niches = [
            "healthcare and wellness",
            "education and learning",
            "productivity and automation",
            "e-commerce and retail",
            "content creation and marketing",
            "finance and fintech",
            "real estate and property",
            "food and nutrition",
            "fitness and sports",
            "travel and hospitality",
            "sustainability and environment",
            "remote work and collaboration",
            "elderly care and aging",
            "pet care and animals",
            "mental health and therapy",
            "small business operations",
            "freelancing and gig economy",
            "home improvement and DIY",
            "parenting and childcare",
            "personal finance and budgeting"
        ]
        
        # Markets to avoid (too saturated)
        self.avoid_markets = [
            "social media platforms",
            "dating apps",
            "food delivery",
            "ride sharing",
            "cryptocurrency trading",
            "NFT marketplaces",
            "generic chatbots",
            "basic todo apps"
        ]
    
    def get_current_trends(self) -> str:
        """Get current trends to inform idea generation"""
        # In a real implementation, this would fetch from APIs
        # For now, we'll use static trends that are relevant
        trends = [
            "AI automation for small businesses",
            "No-code/low-code solutions",
            "Remote work productivity tools",
            "Sustainability and eco-friendly solutions",
            "Mental health and wellness apps",
            "Elder care technology",
            "Micro-SaaS for niche markets",
            "Voice-first applications",
            "Personalized learning platforms",
            "Local community building tools"
        ]
        return ", ".join(random.sample(trends, 5))
    
    def generate_business_ideas(self, count: int = 5) -> List[Dict[str, Any]]:
        """Generate business ideas using OpenAI API"""
        try:
            # Select today's niche
            today_niche = random.choice(self.niches)
            current_trends = self.get_current_trends()
            avoid_markets = ", ".join(self.avoid_markets)
            
            prompt = f"""You are an expert business strategist and venture capitalist with 20+ years of experience identifying profitable AI business opportunities. Your task is to generate exactly {count} innovative AI business ideas that meet these strict criteria:

REQUIREMENTS:
- Solves a clear, validated problem using AI
- Can be built and launched by a solo, non-technical founder
- Total startup cost under $10,000
- Uses no-code/AI tools for implementation
- Low competition or underserved niche
- Addresses a daily pain point for a specific audience

For each idea, provide a JSON object with these exact fields:
- "name": Catchy, memorable business name
- "summary": One-line value proposition (max 100 chars)
- "target_audience": Specific demographic/psychographic
- "problem_solved": Exact pain point addressed
- "ai_solution": How AI solves the problem
- "implementation": No-code tools and approach needed
- "revenue_model": How money is made
- "launch_cost": Detailed cost breakdown under $10K
- "revenue_1_year": Conservative estimate in dollars
- "revenue_5_year": Growth potential in dollars

Focus on niche: {today_niche}
Current trends to consider: {current_trends}
Avoid these saturated markets: {avoid_markets}

Return ONLY a valid JSON array of {count} business idea objects. No additional text or formatting."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a business idea generation expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                ideas = json.loads(content)
                if not isinstance(ideas, list):
                    raise ValueError("Response is not a list")
                
                # Add niche and generation metadata
                for idea in ideas:
                    idea['niche'] = today_niche
                    idea['generated_at'] = datetime.utcnow().isoformat()
                
                return ideas
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw response: {content}")
                return self._generate_fallback_ideas(count, today_niche)
                
        except Exception as e:
            print(f"Error generating ideas: {e}")
            return self._generate_fallback_ideas(count, "general business")
    
    def score_business_idea(self, idea: Dict[str, Any]) -> Dict[str, float]:
        """Score a business idea across 6 dimensions"""
        try:
            scoring_prompt = f"""As a business evaluation expert, score this AI business idea across 6 dimensions on a 1-10 scale:

BUSINESS IDEA:
Name: {idea.get('name', '')}
Summary: {idea.get('summary', '')}
Problem: {idea.get('problem_solved', '')}
Solution: {idea.get('ai_solution', '')}
Implementation: {idea.get('implementation', '')}
Launch Cost: {idea.get('launch_cost', '')}

Score each dimension (1-10, where 10 is best):

1. COST TO BUILD (based on launch cost):
   - Under $1K = 10 points
   - $1K-$3K = 8 points  
   - $3K-$5K = 6 points
   - $5K-$7K = 4 points
   - $7K-$10K = 2 points
   - Over $10K = 1 point

2. EASE OF IMPLEMENTATION (for non-technical solo founder):
   - No-code/AI tools only = 10 points
   - Basic integrations needed = 8 points
   - Some technical setup = 6 points
   - Moderate complexity = 4 points
   - High complexity = 2 points
   - Requires coding = 1 point

3. MARKET SIZE (total addressable market):
   - $10B+ TAM = 10 points
   - $1B-$10B TAM = 8 points
   - $100M-$1B TAM = 6 points
   - $10M-$100M TAM = 4 points
   - $1M-$10M TAM = 2 points
   - Under $1M TAM = 1 point

4. COMPETITION LEVEL:
   - Blue ocean/no competition = 10 points
   - 1-2 weak competitors = 8 points
   - 3-5 competitors = 6 points
   - 6-10 competitors = 4 points
   - 10+ competitors = 2 points
   - Saturated market = 1 point

5. PROBLEM SEVERITY (how urgent/painful):
   - Critical daily pain point = 10 points
   - Significant weekly issue = 8 points
   - Moderate monthly problem = 6 points
   - Minor occasional issue = 4 points
   - Nice-to-have improvement = 2 points
   - Luxury/convenience = 1 point

6. FOUNDER FIT (solo, non-technical):
   - Perfect fit for solo founder = 10 points
   - Good fit with minimal help = 8 points
   - Manageable with outsourcing = 6 points
   - Challenging but possible = 4 points
   - Requires significant support = 2 points
   - Not suitable for solo founder = 1 point

Return ONLY a JSON object with scores:
{{"cost_to_build": X.X, "ease_of_implementation": X.X, "market_size": X.X, "competition_level": X.X, "problem_severity": X.X, "founder_fit": X.X}}"""

            response = self.client.chat.completions.create(
                model=self.backup_model,  # Use cheaper model for scoring
                messages=[
                    {"role": "system", "content": "You are a business scoring expert. Always respond with valid JSON only."},
                    {"role": "user", "content": scoring_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            scores = json.loads(content)
            
            # Calculate total score (weighted average)
            weights = {
                'cost_to_build': 0.20,
                'ease_of_implementation': 0.20,
                'market_size': 0.15,
                'competition_level': 0.15,
                'problem_severity': 0.15,
                'founder_fit': 0.15
            }
            
            total_score = sum(scores[key] * weights[key] for key in scores if key in weights)
            scores['total'] = round(total_score, 2)
            
            return scores
            
        except Exception as e:
            print(f"Error scoring idea: {e}")
            # Return default scores if API fails
            return {
                'cost_to_build': 5.0,
                'ease_of_implementation': 5.0,
                'market_size': 5.0,
                'competition_level': 5.0,
                'problem_severity': 5.0,
                'founder_fit': 5.0,
                'total': 5.0
            }
    
    def generate_validation_evidence(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation evidence for a business idea"""
        try:
            validation_prompt = f"""As a market research expert, provide validation evidence for this business idea:

BUSINESS IDEA:
Name: {idea.get('name', '')}
Summary: {idea.get('summary', '')}
Target Audience: {idea.get('target_audience', '')}
Problem: {idea.get('problem_solved', '')}

Provide realistic validation evidence in JSON format:

{{
  "market_demand": {{
    "search_volume": "estimated monthly searches for related keywords",
    "trend_direction": "growing/stable/declining",
    "social_mentions": "estimated social media mentions per month"
  }},
  "competition_analysis": {{
    "direct_competitors": ["competitor1", "competitor2", "competitor3"],
    "market_saturation": "low/medium/high",
    "competitive_advantage": "key differentiator"
  }},
  "financial_indicators": {{
    "market_size_estimate": "TAM in dollars",
    "average_pricing": "typical pricing in market",
    "customer_acquisition_cost": "estimated CAC"
  }},
  "risk_factors": ["risk1", "risk2", "risk3"],
  "success_indicators": ["indicator1", "indicator2", "indicator3"]
}}

Base estimates on realistic market data and industry standards."""

            response = self.client.chat.completions.create(
                model=self.backup_model,
                messages=[
                    {"role": "system", "content": "You are a market research expert. Always respond with valid JSON only."},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.4,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            return json.loads(content)
            
        except Exception as e:
            print(f"Error generating validation evidence: {e}")
            return {
                "market_demand": {
                    "search_volume": "Data unavailable",
                    "trend_direction": "unknown",
                    "social_mentions": "Data unavailable"
                },
                "competition_analysis": {
                    "direct_competitors": ["Analysis pending"],
                    "market_saturation": "unknown",
                    "competitive_advantage": "To be determined"
                },
                "financial_indicators": {
                    "market_size_estimate": "Analysis pending",
                    "average_pricing": "Market research needed",
                    "customer_acquisition_cost": "To be calculated"
                },
                "risk_factors": ["Market validation needed"],
                "success_indicators": ["Further research required"]
            }
    
    def _generate_fallback_ideas(self, count: int, niche: str) -> List[Dict[str, Any]]:
        """Generate fallback ideas if API fails"""
        fallback_ideas = []
        for i in range(count):
            idea = {
                "name": f"AI Solution {i+1}",
                "summary": f"AI-powered solution for {niche} market",
                "target_audience": "Small business owners",
                "problem_solved": "Manual processes taking too much time",
                "ai_solution": "Automated workflow using AI",
                "implementation": "No-code platform integration",
                "revenue_model": "Monthly subscription",
                "launch_cost": 5000,
                "revenue_1_year": 50000,
                "revenue_5_year": 500000,
                "niche": niche,
                "generated_at": datetime.utcnow().isoformat()
            }
            fallback_ideas.append(idea)
        return fallback_ideas

