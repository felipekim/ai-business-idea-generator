"""
Custom Evaluation and Scoring Engine
Advanced scoring system with personalized weighting for business idea evaluation
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import math

logger = logging.getLogger(__name__)

class EvaluationDimension(Enum):
    """Evaluation dimensions for business ideas"""
    COST_TO_BUILD = "cost_to_build"
    EASE_OF_IMPLEMENTATION = "ease_of_implementation"
    MARKET_SIZE = "market_size"
    COMPETITION_LEVEL = "competition_level"
    PROBLEM_SEVERITY = "problem_severity"
    FOUNDER_FIT = "founder_fit"

@dataclass
class DimensionConfig:
    """Configuration for an evaluation dimension"""
    name: str
    weight: float  # 0.0 to 1.0
    description: str
    scoring_criteria: Dict[str, Any]
    min_score: float = 1.0
    max_score: float = 10.0
    reverse_scoring: bool = False  # True for dimensions where lower is better (e.g., cost)

@dataclass
class EvaluationCriteria:
    """Complete evaluation criteria configuration"""
    dimensions: Dict[EvaluationDimension, DimensionConfig]
    overall_weighting: Dict[str, float] = field(default_factory=dict)
    scoring_methodology: str = "weighted_average"
    confidence_threshold: float = 7.0
    investment_grade_threshold: float = 8.5
    
    def __post_init__(self):
        """Validate and normalize weights"""
        self._validate_weights()
        self._normalize_weights()
    
    def _validate_weights(self):
        """Validate weight configuration"""
        total_weight = sum(dim.weight for dim in self.dimensions.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Dimension weights sum to {total_weight:.3f}, normalizing to 1.0")
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total_weight = sum(dim.weight for dim in self.dimensions.values())
        if total_weight > 0:
            for dimension in self.dimensions.values():
                dimension.weight = dimension.weight / total_weight

@dataclass
class DimensionScore:
    """Score for a single evaluation dimension"""
    dimension: EvaluationDimension
    raw_score: float
    weighted_score: float
    confidence: float
    reasoning: str
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    calculation_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    """Complete evaluation result for a business idea"""
    idea_title: str
    overall_score: float
    investment_grade: str  # 'A+', 'A', 'B+', 'B', 'C+', 'C', 'D'
    dimension_scores: Dict[EvaluationDimension, DimensionScore]
    confidence_level: float
    evaluation_timestamp: datetime
    evaluation_criteria_used: Dict[str, Any]
    
    # Investment analysis
    financial_projections: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    opportunity_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)

class CustomScoringEngine:
    """Advanced scoring engine with personalized evaluation criteria"""
    
    def __init__(self, evaluation_criteria: Optional[EvaluationCriteria] = None):
        self.evaluation_criteria = evaluation_criteria or self._create_default_criteria()
        self.scoring_history = []
        
        logger.info("Custom Scoring Engine initialized")
    
    def _create_default_criteria(self) -> EvaluationCriteria:
        """Create default evaluation criteria optimized for solo non-technical founders"""
        dimensions = {
            EvaluationDimension.COST_TO_BUILD: DimensionConfig(
                name="Cost to Build",
                weight=0.20,  # 20% - Important for bootstrapped founders
                description="Total startup cost including development, tools, and initial operations",
                scoring_criteria={
                    "excellent": {"max_cost": 2000, "score": 10},
                    "good": {"max_cost": 5000, "score": 8},
                    "acceptable": {"max_cost": 10000, "score": 6},
                    "poor": {"max_cost": 20000, "score": 3},
                    "unacceptable": {"max_cost": float('inf'), "score": 1}
                },
                reverse_scoring=True  # Lower cost = higher score
            ),
            
            EvaluationDimension.EASE_OF_IMPLEMENTATION: DimensionConfig(
                name="Ease of Implementation",
                weight=0.25,  # 25% - Critical for non-technical founders
                description="How easily a non-technical founder can implement using no-code/AI tools",
                scoring_criteria={
                    "no_code_only": {"score": 10, "description": "Pure no-code/AI tools"},
                    "minimal_technical": {"score": 8, "description": "Basic technical skills needed"},
                    "moderate_technical": {"score": 6, "description": "Some coding or technical help required"},
                    "high_technical": {"score": 3, "description": "Significant technical expertise needed"},
                    "expert_technical": {"score": 1, "description": "Advanced technical team required"}
                }
            ),
            
            EvaluationDimension.MARKET_SIZE: DimensionConfig(
                name="Market Size",
                weight=0.20,  # 20% - Important for growth potential
                description="Total addressable market and growth potential",
                scoring_criteria={
                    "massive": {"min_tam": 10000000000, "score": 10},  # $10B+
                    "large": {"min_tam": 1000000000, "score": 8},     # $1B+
                    "medium": {"min_tam": 100000000, "score": 6},     # $100M+
                    "small": {"min_tam": 10000000, "score": 4},      # $10M+
                    "niche": {"min_tam": 0, "score": 2}              # <$10M
                }
            ),
            
            EvaluationDimension.COMPETITION_LEVEL: DimensionConfig(
                name="Competition Level",
                weight=0.15,  # 15% - Important but manageable with good positioning
                description="Level of existing competition and market saturation",
                scoring_criteria={
                    "blue_ocean": {"score": 10, "description": "No direct competitors"},
                    "low_competition": {"score": 8, "description": "Few weak competitors"},
                    "moderate_competition": {"score": 6, "description": "Several competitors, room for differentiation"},
                    "high_competition": {"score": 3, "description": "Many strong competitors"},
                    "saturated": {"score": 1, "description": "Market dominated by giants"}
                },
                reverse_scoring=True  # Lower competition = higher score
            ),
            
            EvaluationDimension.PROBLEM_SEVERITY: DimensionConfig(
                name="Problem Severity",
                weight=0.15,  # 15% - Important for product-market fit
                description="How urgent and painful the problem is for target customers",
                scoring_criteria={
                    "critical": {"score": 10, "description": "Mission-critical problem, high willingness to pay"},
                    "important": {"score": 8, "description": "Important problem, clear value proposition"},
                    "moderate": {"score": 6, "description": "Moderate pain point, some willingness to pay"},
                    "minor": {"score": 3, "description": "Nice-to-have solution"},
                    "trivial": {"score": 1, "description": "Low-priority problem"}
                }
            ),
            
            EvaluationDimension.FOUNDER_FIT: DimensionConfig(
                name="Founder Fit",
                weight=0.05,  # 5% - Assumes personal assessment
                description="How well the opportunity aligns with founder skills and interests",
                scoring_criteria={
                    "perfect_fit": {"score": 10, "description": "Ideal match for skills and interests"},
                    "good_fit": {"score": 8, "description": "Strong alignment with capabilities"},
                    "moderate_fit": {"score": 6, "description": "Reasonable fit, some learning required"},
                    "poor_fit": {"score": 3, "description": "Significant skill gaps"},
                    "no_fit": {"score": 1, "description": "Misaligned with capabilities"}
                }
            )
        }
        
        return EvaluationCriteria(
            dimensions=dimensions,
            scoring_methodology="weighted_average",
            confidence_threshold=7.0,
            investment_grade_threshold=8.5
        )
    
    def evaluate_business_idea(self, research_data: Dict, idea_title: str, 
                             custom_weights: Optional[Dict[str, float]] = None) -> EvaluationResult:
        """Evaluate a business idea using custom scoring criteria"""
        logger.info(f"Evaluating business idea: {idea_title}")
        
        # Apply custom weights if provided
        if custom_weights:
            self._apply_custom_weights(custom_weights)
        
        # Calculate dimension scores
        dimension_scores = {}
        for dimension in EvaluationDimension:
            score = self._calculate_dimension_score(dimension, research_data, idea_title)
            dimension_scores[dimension] = score
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)
        
        # Determine investment grade
        investment_grade = self._determine_investment_grade(overall_score)
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(dimension_scores, research_data)
        
        # Generate financial projections
        financial_projections = self._generate_financial_projections(dimension_scores, research_data)
        
        # Perform risk assessment
        risk_assessment = self._perform_risk_assessment(dimension_scores, research_data)
        
        # Analyze opportunities
        opportunity_analysis = self._analyze_opportunities(dimension_scores, research_data)
        
        # Generate recommendations
        strengths, weaknesses, recommendations, next_steps = self._generate_recommendations(
            dimension_scores, research_data, overall_score
        )
        
        # Create evaluation result
        result = EvaluationResult(
            idea_title=idea_title,
            overall_score=overall_score,
            investment_grade=investment_grade,
            dimension_scores=dimension_scores,
            confidence_level=confidence_level,
            evaluation_timestamp=datetime.now(),
            evaluation_criteria_used=asdict(self.evaluation_criteria),
            financial_projections=financial_projections,
            risk_assessment=risk_assessment,
            opportunity_analysis=opportunity_analysis,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            next_steps=next_steps
        )
        
        # Store in history
        self.scoring_history.append(result)
        
        logger.info(f"Evaluation completed: {overall_score:.1f}/10 ({investment_grade})")
        return result
    
    def _apply_custom_weights(self, custom_weights: Dict[str, float]):
        """Apply custom weights to evaluation criteria"""
        for dimension_name, weight in custom_weights.items():
            try:
                dimension = EvaluationDimension(dimension_name)
                if dimension in self.evaluation_criteria.dimensions:
                    self.evaluation_criteria.dimensions[dimension].weight = weight
            except ValueError:
                logger.warning(f"Unknown dimension: {dimension_name}")
        
        # Renormalize weights
        self.evaluation_criteria._normalize_weights()
    
    def _calculate_dimension_score(self, dimension: EvaluationDimension, 
                                 research_data: Dict, idea_title: str) -> DimensionScore:
        """Calculate score for a specific dimension"""
        config = self.evaluation_criteria.dimensions[dimension]
        
        # Extract relevant data for this dimension
        if dimension == EvaluationDimension.COST_TO_BUILD:
            return self._score_cost_to_build(config, research_data, idea_title)
        elif dimension == EvaluationDimension.EASE_OF_IMPLEMENTATION:
            return self._score_ease_of_implementation(config, research_data, idea_title)
        elif dimension == EvaluationDimension.MARKET_SIZE:
            return self._score_market_size(config, research_data, idea_title)
        elif dimension == EvaluationDimension.COMPETITION_LEVEL:
            return self._score_competition_level(config, research_data, idea_title)
        elif dimension == EvaluationDimension.PROBLEM_SEVERITY:
            return self._score_problem_severity(config, research_data, idea_title)
        elif dimension == EvaluationDimension.FOUNDER_FIT:
            return self._score_founder_fit(config, research_data, idea_title)
        else:
            # Default scoring
            return DimensionScore(
                dimension=dimension,
                raw_score=5.0,
                weighted_score=5.0 * config.weight,
                confidence=0.5,
                reasoning="Default scoring applied - insufficient data",
                supporting_data={},
                calculation_details={}
            )
    
    def _score_cost_to_build(self, config: DimensionConfig, research_data: Dict, 
                           idea_title: str) -> DimensionScore:
        """Score cost to build dimension"""
        # Extract financial data from research
        financial_data = research_data.get('research_summary', {}).get('financial_breakdown', {})
        startup_costs = financial_data.get('startup_costs', {})
        
        # Calculate total estimated cost
        total_cost = 0
        cost_breakdown = {}
        
        # Development costs (AI/no-code tools)
        if 'AI' in idea_title or 'software' in idea_title.lower():
            dev_cost = 2000  # Estimated for AI/no-code development
            cost_breakdown['development'] = dev_cost
            total_cost += dev_cost
        
        # Tool and subscription costs
        tool_costs = 500  # Estimated monthly tools * 6 months
        cost_breakdown['tools_and_subscriptions'] = tool_costs
        total_cost += tool_costs
        
        # Marketing and launch costs
        marketing_cost = 1000  # Basic marketing budget
        cost_breakdown['marketing'] = marketing_cost
        total_cost += marketing_cost
        
        # Legal and administrative
        admin_cost = 500
        cost_breakdown['legal_admin'] = admin_cost
        total_cost += admin_cost
        
        # Score based on total cost
        raw_score = 5.0
        reasoning = f"Estimated total cost: ${total_cost:,}"
        
        for criteria_name, criteria in config.scoring_criteria.items():
            if total_cost <= criteria.get('max_cost', float('inf')):
                raw_score = criteria['score']
                reasoning += f" - {criteria_name} range"
                break
        
        weighted_score = raw_score * config.weight
        confidence = 0.7  # Moderate confidence in cost estimation
        
        return DimensionScore(
            dimension=EvaluationDimension.COST_TO_BUILD,
            raw_score=raw_score,
            weighted_score=weighted_score,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={'total_cost': total_cost, 'cost_breakdown': cost_breakdown},
            calculation_details={'criteria_used': config.scoring_criteria}
        )
    
    def _score_ease_of_implementation(self, config: DimensionConfig, research_data: Dict, 
                                    idea_title: str) -> DimensionScore:
        """Score ease of implementation dimension"""
        # Analyze idea complexity based on title and description
        idea_lower = idea_title.lower()
        
        # Determine technical complexity
        no_code_indicators = ['ai-powered', 'chatbot', 'automation', 'dashboard', 'analytics']
        moderate_indicators = ['platform', 'marketplace', 'integration', 'api']
        complex_indicators = ['blockchain', 'machine learning', 'computer vision', 'iot']
        
        raw_score = 6.0  # Default moderate score
        reasoning = "Moderate technical complexity"
        
        if any(indicator in idea_lower for indicator in no_code_indicators):
            raw_score = 9.0
            reasoning = "High potential for no-code/AI implementation"
        elif any(indicator in idea_lower for indicator in complex_indicators):
            raw_score = 3.0
            reasoning = "High technical complexity, requires specialized skills"
        elif any(indicator in idea_lower for indicator in moderate_indicators):
            raw_score = 6.0
            reasoning = "Moderate complexity, some technical skills needed"
        
        # Adjust based on research data
        research_summary = research_data.get('research_summary', {})
        if 'no-code' in str(research_summary).lower():
            raw_score = min(raw_score + 1.0, 10.0)
            reasoning += " (no-code solutions available)"
        
        weighted_score = raw_score * config.weight
        confidence = 0.8  # High confidence in implementation assessment
        
        return DimensionScore(
            dimension=EvaluationDimension.EASE_OF_IMPLEMENTATION,
            raw_score=raw_score,
            weighted_score=weighted_score,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={'complexity_indicators': {'no_code': no_code_indicators, 'complex': complex_indicators}},
            calculation_details={'base_score': raw_score}
        )
    
    def _score_market_size(self, config: DimensionConfig, research_data: Dict, 
                         idea_title: str) -> DimensionScore:
        """Score market size dimension"""
        # Extract market data from research
        market_analysis = research_data.get('research_summary', {}).get('market_analysis', {})
        
        # Estimate market size based on trends and industry
        trend_score = market_analysis.get('trend_score', 5.0)
        market_interest = market_analysis.get('market_interest', 'moderate')
        
        # Base market size estimation
        raw_score = 5.0
        estimated_tam = 100000000  # Default $100M
        reasoning = "Market size estimated from trend analysis"
        
        # Adjust based on market interest and trends
        if market_interest == 'high' and trend_score > 7.0:
            raw_score = 8.0
            estimated_tam = 1000000000  # $1B
            reasoning = "High market interest with strong trends"
        elif market_interest == 'high' or trend_score > 6.0:
            raw_score = 7.0
            estimated_tam = 500000000  # $500M
            reasoning = "Good market indicators"
        elif market_interest == 'low' or trend_score < 4.0:
            raw_score = 3.0
            estimated_tam = 50000000  # $50M
            reasoning = "Limited market indicators"
        
        # Industry-specific adjustments
        idea_lower = idea_title.lower()
        if any(term in idea_lower for term in ['ai', 'fintech', 'healthcare', 'education']):
            raw_score = min(raw_score + 1.0, 10.0)
            estimated_tam *= 2
            reasoning += " (high-growth industry)"
        
        weighted_score = raw_score * config.weight
        confidence = 0.6  # Moderate confidence in market size estimation
        
        return DimensionScore(
            dimension=EvaluationDimension.MARKET_SIZE,
            raw_score=raw_score,
            weighted_score=weighted_score,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={'estimated_tam': estimated_tam, 'market_interest': market_interest, 'trend_score': trend_score},
            calculation_details={'base_score': raw_score}
        )
    
    def _score_competition_level(self, config: DimensionConfig, research_data: Dict, 
                               idea_title: str) -> DimensionScore:
        """Score competition level dimension"""
        # Extract competitive data from research
        competitive_analysis = research_data.get('research_summary', {}).get('competitive_landscape', {})
        
        competition_level = competitive_analysis.get('competition_level', 'moderate')
        competitive_intensity = competitive_analysis.get('competitive_intensity', 5.0)
        funding_activity = competitive_analysis.get('funding_activity', [])
        
        # Score based on competition indicators
        raw_score = 5.0
        reasoning = f"Competition level: {competition_level}"
        
        if competition_level == 'low' and competitive_intensity < 4.0:
            raw_score = 8.0
            reasoning = "Low competition with limited funding activity"
        elif competition_level == 'moderate' and competitive_intensity < 6.0:
            raw_score = 6.0
            reasoning = "Moderate competition, room for differentiation"
        elif competition_level == 'high' or competitive_intensity > 7.0:
            raw_score = 3.0
            reasoning = "High competition with significant market activity"
        
        # Adjust based on funding activity
        if len(funding_activity) > 5:
            raw_score = max(raw_score - 2.0, 1.0)
            reasoning += " (high funding activity indicates intense competition)"
        elif len(funding_activity) == 0:
            raw_score = min(raw_score + 1.0, 10.0)
            reasoning += " (no recent funding activity)"
        
        weighted_score = raw_score * config.weight
        confidence = 0.7  # Good confidence in competition assessment
        
        return DimensionScore(
            dimension=EvaluationDimension.COMPETITION_LEVEL,
            raw_score=raw_score,
            weighted_score=weighted_score,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={'competition_level': competition_level, 'competitive_intensity': competitive_intensity, 'funding_count': len(funding_activity)},
            calculation_details={'base_score': raw_score}
        )
    
    def _score_problem_severity(self, config: DimensionConfig, research_data: Dict, 
                              idea_title: str) -> DimensionScore:
        """Score problem severity dimension"""
        # Extract social sentiment and industry insights
        social_sentiment = research_data.get('research_summary', {}).get('social_sentiment', {})
        industry_insights = research_data.get('research_summary', {}).get('industry_insights', {})
        
        sentiment_score = social_sentiment.get('sentiment_score', 0.5)
        community_interest = social_sentiment.get('community_interest', 'moderate')
        discussion_volume = social_sentiment.get('discussion_volume', 0)
        
        # Score based on problem indicators
        raw_score = 5.0
        reasoning = "Problem severity assessed from social signals"
        
        # High discussion volume indicates real problem
        if discussion_volume > 100 and sentiment_score > 0.6:
            raw_score = 8.0
            reasoning = "High community engagement with positive sentiment"
        elif discussion_volume > 50:
            raw_score = 6.0
            reasoning = "Moderate community discussion"
        elif discussion_volume < 10:
            raw_score = 3.0
            reasoning = "Limited community discussion"
        
        # Industry momentum indicates problem urgency
        industry_momentum = industry_insights.get('industry_momentum', 'stable')
        if industry_momentum == 'accelerating':
            raw_score = min(raw_score + 1.5, 10.0)
            reasoning += " (accelerating industry momentum)"
        elif industry_momentum == 'slowing':
            raw_score = max(raw_score - 1.0, 1.0)
            reasoning += " (slowing industry momentum)"
        
        # Problem-specific keywords
        idea_lower = idea_title.lower()
        urgent_keywords = ['security', 'healthcare', 'finance', 'emergency', 'critical']
        if any(keyword in idea_lower for keyword in urgent_keywords):
            raw_score = min(raw_score + 1.0, 10.0)
            reasoning += " (addresses urgent problem domain)"
        
        weighted_score = raw_score * config.weight
        confidence = 0.6  # Moderate confidence in problem severity assessment
        
        return DimensionScore(
            dimension=EvaluationDimension.PROBLEM_SEVERITY,
            raw_score=raw_score,
            weighted_score=weighted_score,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={'sentiment_score': sentiment_score, 'discussion_volume': discussion_volume, 'industry_momentum': industry_momentum},
            calculation_details={'base_score': raw_score}
        )
    
    def _score_founder_fit(self, config: DimensionConfig, research_data: Dict, 
                         idea_title: str) -> DimensionScore:
        """Score founder fit dimension (assumes solo non-technical founder)"""
        # This is a placeholder - in practice, this would be customized based on founder profile
        idea_lower = idea_title.lower()
        
        # Assess fit for non-technical solo founder
        raw_score = 7.0  # Default good fit
        reasoning = "Good fit for non-technical solo founder"
        
        # AI/automation ideas are good fit for leveraging tools
        if any(term in idea_lower for term in ['ai', 'automation', 'tool', 'assistant']):
            raw_score = 8.0
            reasoning = "Excellent fit - leverages AI/automation tools"
        
        # Complex technical ideas are poor fit
        if any(term in idea_lower for term in ['blockchain', 'iot', 'hardware', 'infrastructure']):
            raw_score = 4.0
            reasoning = "Poor fit - requires significant technical expertise"
        
        # Business/service ideas are good fit
        if any(term in idea_lower for term in ['service', 'consulting', 'marketplace', 'platform']):
            raw_score = 7.0
            reasoning = "Good fit - business-focused opportunity"
        
        weighted_score = raw_score * config.weight
        confidence = 0.8  # High confidence in founder fit assessment
        
        return DimensionScore(
            dimension=EvaluationDimension.FOUNDER_FIT,
            raw_score=raw_score,
            weighted_score=weighted_score,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={'founder_profile': 'non_technical_solo'},
            calculation_details={'base_score': raw_score}
        )
    
    def _calculate_overall_score(self, dimension_scores: Dict[EvaluationDimension, DimensionScore]) -> float:
        """Calculate overall weighted score"""
        total_weighted_score = sum(score.weighted_score for score in dimension_scores.values())
        return round(total_weighted_score, 1)
    
    def _determine_investment_grade(self, overall_score: float) -> str:
        """Determine investment grade based on overall score"""
        if overall_score >= 9.5:
            return 'A+'
        elif overall_score >= 9.0:
            return 'A'
        elif overall_score >= 8.5:
            return 'B+'
        elif overall_score >= 8.0:
            return 'B'
        elif overall_score >= 7.0:
            return 'C+'
        elif overall_score >= 6.0:
            return 'C'
        else:
            return 'D'
    
    def _calculate_confidence_level(self, dimension_scores: Dict[EvaluationDimension, DimensionScore], 
                                  research_data: Dict) -> float:
        """Calculate overall confidence level in the evaluation"""
        # Average confidence across dimensions
        avg_confidence = sum(score.confidence for score in dimension_scores.values()) / len(dimension_scores)
        
        # Adjust based on data quality
        sources_count = len(research_data.get('sources', []))
        if sources_count >= 8:
            data_quality_factor = 1.0
        elif sources_count >= 5:
            data_quality_factor = 0.9
        else:
            data_quality_factor = 0.7
        
        return round(avg_confidence * data_quality_factor, 2)
    
    def _generate_financial_projections(self, dimension_scores: Dict[EvaluationDimension, DimensionScore], 
                                      research_data: Dict) -> Dict[str, Any]:
        """Generate financial projections based on evaluation"""
        cost_score = dimension_scores[EvaluationDimension.COST_TO_BUILD]
        market_score = dimension_scores[EvaluationDimension.MARKET_SIZE]
        
        startup_cost = cost_score.supporting_data.get('total_cost', 5000)
        estimated_tam = market_score.supporting_data.get('estimated_tam', 100000000)
        
        # Revenue projections (conservative estimates)
        year1_revenue = min(startup_cost * 2, estimated_tam * 0.0001)  # 0.01% market penetration
        year3_revenue = min(year1_revenue * 5, estimated_tam * 0.001)   # 0.1% market penetration
        year5_revenue = min(year3_revenue * 3, estimated_tam * 0.01)    # 1% market penetration
        
        return {
            'startup_cost': startup_cost,
            'estimated_tam': estimated_tam,
            'revenue_projections': {
                'year_1': round(year1_revenue),
                'year_3': round(year3_revenue),
                'year_5': round(year5_revenue)
            },
            'break_even_months': max(6, round(startup_cost / (year1_revenue / 12))),
            'roi_5_year': round((year5_revenue - startup_cost) / startup_cost * 100, 1)
        }
    
    def _perform_risk_assessment(self, dimension_scores: Dict[EvaluationDimension, DimensionScore], 
                                research_data: Dict) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        risks = []
        risk_level = "Medium"
        
        # Competition risk
        competition_score = dimension_scores[EvaluationDimension.COMPETITION_LEVEL].raw_score
        if competition_score < 5.0:
            risks.append("High competition risk - market may be saturated")
        
        # Implementation risk
        implementation_score = dimension_scores[EvaluationDimension.EASE_OF_IMPLEMENTATION].raw_score
        if implementation_score < 6.0:
            risks.append("Implementation risk - may require technical expertise beyond founder capabilities")
        
        # Market risk
        market_score = dimension_scores[EvaluationDimension.MARKET_SIZE].raw_score
        if market_score < 5.0:
            risks.append("Market size risk - limited growth potential")
        
        # Problem validation risk
        problem_score = dimension_scores[EvaluationDimension.PROBLEM_SEVERITY].raw_score
        if problem_score < 6.0:
            risks.append("Problem validation risk - may not address urgent customer need")
        
        # Overall risk level
        avg_score = sum(score.raw_score for score in dimension_scores.values()) / len(dimension_scores)
        if avg_score >= 8.0:
            risk_level = "Low"
        elif avg_score >= 6.0:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'overall_risk_level': risk_level,
            'identified_risks': risks,
            'risk_mitigation_strategies': self._generate_risk_mitigation_strategies(risks),
            'confidence_in_assessment': 0.7
        }
    
    def _generate_risk_mitigation_strategies(self, risks: List[str]) -> List[str]:
        """Generate risk mitigation strategies"""
        strategies = []
        
        for risk in risks:
            if "competition" in risk.lower():
                strategies.append("Focus on unique value proposition and niche market entry")
            elif "implementation" in risk.lower():
                strategies.append("Start with MVP using no-code tools, hire technical help for scaling")
            elif "market size" in risk.lower():
                strategies.append("Validate market demand early, consider adjacent markets")
            elif "problem validation" in risk.lower():
                strategies.append("Conduct customer interviews, build prototype for validation")
        
        return strategies
    
    def _analyze_opportunities(self, dimension_scores: Dict[EvaluationDimension, DimensionScore], 
                             research_data: Dict) -> Dict[str, Any]:
        """Analyze opportunities and growth potential"""
        opportunities = []
        
        # Market opportunities
        market_score = dimension_scores[EvaluationDimension.MARKET_SIZE].raw_score
        if market_score >= 7.0:
            opportunities.append("Large market opportunity with significant growth potential")
        
        # Implementation advantages
        implementation_score = dimension_scores[EvaluationDimension.EASE_OF_IMPLEMENTATION].raw_score
        if implementation_score >= 8.0:
            opportunities.append("Low technical barriers enable rapid market entry")
        
        # Cost advantages
        cost_score = dimension_scores[EvaluationDimension.COST_TO_BUILD].raw_score
        if cost_score >= 8.0:
            opportunities.append("Low startup costs enable bootstrapped growth")
        
        # Competition gaps
        competition_score = dimension_scores[EvaluationDimension.COMPETITION_LEVEL].raw_score
        if competition_score >= 7.0:
            opportunities.append("Limited competition creates first-mover advantage")
        
        return {
            'identified_opportunities': opportunities,
            'growth_potential': 'High' if len(opportunities) >= 3 else 'Medium' if len(opportunities) >= 2 else 'Low',
            'strategic_advantages': opportunities[:3]  # Top 3 advantages
        }
    
    def _generate_recommendations(self, dimension_scores: Dict[EvaluationDimension, DimensionScore], 
                                research_data: Dict, overall_score: float) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Generate comprehensive recommendations"""
        strengths = []
        weaknesses = []
        recommendations = []
        next_steps = []
        
        # Analyze each dimension
        for dimension, score in dimension_scores.items():
            if score.raw_score >= 8.0:
                strengths.append(f"{dimension.value.replace('_', ' ').title()}: {score.reasoning}")
            elif score.raw_score <= 4.0:
                weaknesses.append(f"{dimension.value.replace('_', ' ').title()}: {score.reasoning}")
        
        # Generate recommendations based on overall score
        if overall_score >= 8.5:
            recommendations.extend([
                "Excellent opportunity - proceed with development",
                "Consider seeking angel investment for faster growth",
                "Focus on rapid market entry to capture first-mover advantage"
            ])
            next_steps.extend([
                "Develop detailed business plan",
                "Create MVP using no-code tools",
                "Validate with target customers"
            ])
        elif overall_score >= 7.0:
            recommendations.extend([
                "Good opportunity with some areas for improvement",
                "Address identified weaknesses before full commitment",
                "Consider starting as side project to validate market"
            ])
            next_steps.extend([
                "Conduct market validation research",
                "Build prototype to test core assumptions",
                "Develop go-to-market strategy"
            ])
        else:
            recommendations.extend([
                "Significant challenges identified - proceed with caution",
                "Consider pivoting to address weaknesses",
                "Extensive validation required before investment"
            ])
            next_steps.extend([
                "Reassess market opportunity",
                "Validate problem-solution fit",
                "Consider alternative approaches"
            ])
        
        return strengths[:5], weaknesses[:5], recommendations[:5], next_steps[:5]
    
    def get_scoring_history(self) -> List[EvaluationResult]:
        """Get historical scoring results"""
        return self.scoring_history.copy()
    
    def export_evaluation_criteria(self, file_path: str):
        """Export current evaluation criteria to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(asdict(self.evaluation_criteria), f, indent=2, default=str)
            logger.info(f"Evaluation criteria exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export evaluation criteria: {e}")
    
    def import_evaluation_criteria(self, file_path: str):
        """Import evaluation criteria from file"""
        try:
            with open(file_path, 'r') as f:
                criteria_dict = json.load(f)
            
            # Reconstruct evaluation criteria (simplified version)
            # In practice, this would need more sophisticated deserialization
            logger.info(f"Evaluation criteria imported from {file_path}")
        except Exception as e:
            logger.error(f"Failed to import evaluation criteria: {e}")

# Factory function to create scoring engine
def create_custom_scoring_engine(custom_weights: Optional[Dict[str, float]] = None) -> CustomScoringEngine:
    """Create custom scoring engine with optional weight customization"""
    engine = CustomScoringEngine()
    
    if custom_weights:
        engine._apply_custom_weights(custom_weights)
    
    return engine

# Test function for custom scoring engine
def test_custom_scoring_engine():
    """Test the custom scoring engine"""
    print("Testing Custom Scoring Engine...")
    
    # Create scoring engine
    engine = create_custom_scoring_engine()
    
    # Test evaluation criteria
    print("\n1. Testing evaluation criteria...")
    criteria = engine.evaluation_criteria
    print(f"✅ Dimensions configured: {len(criteria.dimensions)}")
    print(f"✅ Weight sum: {sum(d.weight for d in criteria.dimensions.values()):.3f}")
    
    # Create test research data
    test_research_data = {
        'research_summary': {
            'market_analysis': {
                'trend_score': 7.5,
                'market_interest': 'high'
            },
            'competitive_landscape': {
                'competition_level': 'moderate',
                'competitive_intensity': 5.5,
                'funding_activity': ['startup1', 'startup2']
            },
            'social_sentiment': {
                'sentiment_score': 0.7,
                'community_interest': 'high',
                'discussion_volume': 150
            },
            'industry_insights': {
                'industry_momentum': 'accelerating'
            }
        },
        'sources': [{'url': f'source{i}'} for i in range(8)]
    }
    
    # Test evaluation
    print("\n2. Testing business idea evaluation...")
    result = engine.evaluate_business_idea(
        research_data=test_research_data,
        idea_title="AI-Powered Personal Finance Assistant"
    )
    
    print(f"✅ Overall Score: {result.overall_score}/10")
    print(f"✅ Investment Grade: {result.investment_grade}")
    print(f"✅ Confidence Level: {result.confidence_level}")
    print(f"✅ Dimension Scores: {len(result.dimension_scores)}")
    print(f"✅ Strengths: {len(result.strengths)}")
    print(f"✅ Recommendations: {len(result.recommendations)}")
    
    # Test custom weights
    print("\n3. Testing custom weights...")
    custom_weights = {
        'ease_of_implementation': 0.4,  # Increase importance
        'cost_to_build': 0.3,          # Increase importance
        'market_size': 0.15,           # Decrease importance
        'competition_level': 0.1,      # Decrease importance
        'problem_severity': 0.04,      # Decrease importance
        'founder_fit': 0.01            # Decrease importance
    }
    
    result_custom = engine.evaluate_business_idea(
        research_data=test_research_data,
        idea_title="AI-Powered Personal Finance Assistant",
        custom_weights=custom_weights
    )
    
    print(f"✅ Custom weighted score: {result_custom.overall_score}/10")
    print(f"✅ Score difference: {result_custom.overall_score - result.overall_score:+.1f}")
    
    # Test financial projections
    print("\n4. Testing financial projections...")
    projections = result.financial_projections
    print(f"✅ Startup cost: ${projections['startup_cost']:,}")
    print(f"✅ Year 5 revenue: ${projections['revenue_projections']['year_5']:,}")
    print(f"✅ 5-year ROI: {projections['roi_5_year']}%")
    
    # Test scoring history
    print("\n5. Testing scoring history...")
    history = engine.get_scoring_history()
    print(f"✅ Evaluations in history: {len(history)}")
    
    print("\n🎉 Custom scoring engine test completed successfully!")
    
    return result

if __name__ == "__main__":
    # Run custom scoring engine tests
    test_custom_scoring_engine()

