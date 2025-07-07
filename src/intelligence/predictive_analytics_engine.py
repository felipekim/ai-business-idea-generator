"""
Predictive Analytics Engine
Forecasts market opportunities, trends, and business outcomes using predictive models
"""

import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict, deque
import math

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    """Types of predictions"""
    MARKET_OPPORTUNITY = "market_opportunity"
    TREND_FORECAST = "trend_forecast"
    SUCCESS_PROBABILITY = "success_probability"
    REVENUE_PROJECTION = "revenue_projection"
    COMPETITION_EVOLUTION = "competition_evolution"
    TIMING_OPTIMIZATION = "timing_optimization"

class TimeHorizon(Enum):
    """Prediction time horizons"""
    SHORT_TERM = "short_term"      # 1-3 months
    MEDIUM_TERM = "medium_term"    # 3-12 months
    LONG_TERM = "long_term"        # 1-3 years

class PredictionConfidence(Enum):
    """Prediction confidence levels"""
    VERY_HIGH = "very_high"    # 90-100%
    HIGH = "high"              # 80-89%
    MEDIUM = "medium"          # 60-79%
    LOW = "low"                # 40-59%
    VERY_LOW = "very_low"      # 0-39%

@dataclass
class Prediction:
    """Predictive analytics result"""
    prediction_id: str
    type: PredictionType
    title: str
    description: str
    time_horizon: TimeHorizon
    confidence: float  # 0.0 to 1.0
    confidence_level: PredictionConfidence
    predicted_value: Any
    probability_distribution: Dict[str, float] = field(default_factory=dict)
    supporting_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    prediction_date: datetime = field(default_factory=datetime.utcnow)
    target_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'prediction_id': self.prediction_id,
            'type': self.type.value,
            'title': self.title,
            'description': self.description,
            'time_horizon': self.time_horizon.value,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'predicted_value': self.predicted_value,
            'probability_distribution': self.probability_distribution,
            'supporting_factors': self.supporting_factors,
            'risk_factors': self.risk_factors,
            'prediction_date': self.prediction_date.isoformat(),
            'target_date': self.target_date.isoformat() if self.target_date else None
        }

@dataclass
class MarketForecast:
    """Market forecast data"""
    market_segment: str
    current_size: float
    projected_size: float
    growth_rate: float
    confidence: float
    time_horizon: TimeHorizon
    key_drivers: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

@dataclass
class PredictiveConfig:
    """Configuration for predictive analytics"""
    # Model parameters
    trend_window_days: int = 30
    seasonality_periods: int = 12
    confidence_threshold: float = 0.6
    
    # Prediction horizons (in days)
    short_term_days: int = 90
    medium_term_days: int = 365
    long_term_days: int = 1095
    
    # Market analysis
    market_volatility_threshold: float = 0.2
    growth_rate_threshold: float = 0.1
    competition_impact_factor: float = 0.3
    
    # Risk assessment
    enable_risk_analysis: bool = True
    risk_factors_weight: float = 0.2
    uncertainty_adjustment: float = 0.1

class PredictiveAnalyticsEngine:
    """Predictive analytics engine for business intelligence"""
    
    def __init__(self, config: Optional[PredictiveConfig] = None):
        self.config = config or PredictiveConfig()
        self.historical_data = deque(maxlen=1000)  # Store recent data points
        self.trend_models = {}
        self.market_forecasts = {}
        self.prediction_accuracy = {}
        
        logger.info("Predictive Analytics Engine initialized")
    
    def add_historical_data(self, data_point: Dict[str, Any]):
        """Add historical data point for analysis"""
        data_point['timestamp'] = data_point.get('timestamp', datetime.utcnow())
        self.historical_data.append(data_point)
        
        # Update trend models with new data
        self._update_trend_models(data_point)
    
    def predict_market_opportunities(self, market_context: Dict[str, Any]) -> List[Prediction]:
        """Predict future market opportunities"""
        logger.info("Predicting market opportunities")
        
        try:
            predictions = []
            
            # Analyze different market segments
            market_segments = market_context.get('segments', ['AI automation', 'SaaS', 'fintech'])
            
            for segment in market_segments:
                # Short-term prediction
                short_term_pred = self._predict_segment_opportunity(
                    segment, TimeHorizon.SHORT_TERM, market_context
                )
                if short_term_pred:
                    predictions.append(short_term_pred)
                
                # Medium-term prediction
                medium_term_pred = self._predict_segment_opportunity(
                    segment, TimeHorizon.MEDIUM_TERM, market_context
                )
                if medium_term_pred:
                    predictions.append(medium_term_pred)
            
            logger.info(f"Generated {len(predictions)} market opportunity predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting market opportunities: {e}")
            return []
    
    def forecast_trends(self, trend_data: List[Dict[str, Any]]) -> List[Prediction]:
        """Forecast future trends based on historical data"""
        logger.info("Forecasting trends")
        
        try:
            predictions = []
            
            # Analyze trend patterns
            trend_analysis = self._analyze_trend_patterns(trend_data)
            
            for trend_name, analysis in trend_analysis.items():
                # Create trend forecast
                forecast = self._create_trend_forecast(trend_name, analysis)
                if forecast:
                    predictions.append(forecast)
            
            logger.info(f"Generated {len(predictions)} trend forecasts")
            return predictions
            
        except Exception as e:
            logger.error(f"Error forecasting trends: {e}")
            return []
    
    def predict_success_probability(self, idea_data: Dict[str, Any]) -> Prediction:
        """Predict success probability for a business idea"""
        logger.info(f"Predicting success probability for: {idea_data.get('name', 'Unknown')}")
        
        try:
            # Extract features for prediction
            features = self._extract_prediction_features(idea_data)
            
            # Calculate base success probability
            base_probability = self._calculate_base_success_probability(features)
            
            # Apply market timing adjustments
            timing_adjustment = self._calculate_timing_adjustment(idea_data)
            
            # Apply competition adjustments
            competition_adjustment = self._calculate_competition_adjustment(idea_data)
            
            # Final probability calculation
            final_probability = base_probability * timing_adjustment * competition_adjustment
            final_probability = max(0.0, min(1.0, final_probability))  # Clamp to [0,1]
            
            # Determine confidence based on data quality
            confidence = self._calculate_prediction_confidence(idea_data, features)
            
            # Create prediction
            prediction = Prediction(
                prediction_id=f"success_prob_{idea_data.get('name', 'idea').replace(' ', '_')}_{int(datetime.utcnow().timestamp())}",
                type=PredictionType.SUCCESS_PROBABILITY,
                title=f"Success Probability: {idea_data.get('name', 'Business Idea')}",
                description=f"Predicted success probability of {final_probability*100:.1f}% based on "
                           f"market analysis, timing, and competitive factors.",
                time_horizon=TimeHorizon.MEDIUM_TERM,
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                predicted_value=final_probability,
                probability_distribution={
                    'very_high': max(0, final_probability - 0.2),
                    'high': final_probability,
                    'medium': min(1, final_probability + 0.1),
                    'low': min(1, final_probability + 0.2)
                },
                supporting_factors=self._identify_supporting_factors(features, idea_data),
                risk_factors=self._identify_risk_factors(features, idea_data),
                target_date=datetime.utcnow() + timedelta(days=self.config.medium_term_days)
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting success probability: {e}")
            return self._create_default_prediction(PredictionType.SUCCESS_PROBABILITY)
    
    def predict_revenue_projections(self, idea_data: Dict[str, Any]) -> List[Prediction]:
        """Predict revenue projections for different time horizons"""
        logger.info(f"Predicting revenue projections for: {idea_data.get('name', 'Unknown')}")
        
        try:
            predictions = []
            
            # Get base financial data
            financial_analysis = idea_data.get('financial_analysis', {})
            base_revenue = financial_analysis.get('revenue_projections', {}).get(
                'revenue_by_year', {}
            ).get('year_1', {}).get('annual_revenue', 100000)
            
            # Predict for different time horizons
            for horizon in TimeHorizon:
                revenue_prediction = self._predict_revenue_for_horizon(
                    base_revenue, horizon, idea_data
                )
                if revenue_prediction:
                    predictions.append(revenue_prediction)
            
            logger.info(f"Generated {len(predictions)} revenue predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting revenue projections: {e}")
            return []
    
    def predict_competition_evolution(self, market_data: Dict[str, Any]) -> Prediction:
        """Predict how competition will evolve in the market"""
        logger.info("Predicting competition evolution")
        
        try:
            # Analyze current competition
            current_competition = market_data.get('competitive_analysis', {}).get('competition_level', 5.0)
            market_growth = market_data.get('market_growth_rate', 0.1)
            
            # Predict competition evolution
            predicted_competition = self._predict_competition_change(
                current_competition, market_growth, market_data
            )
            
            # Calculate confidence
            confidence = 0.7 if len(self.historical_data) > 10 else 0.5
            
            prediction = Prediction(
                prediction_id=f"competition_evolution_{int(datetime.utcnow().timestamp())}",
                type=PredictionType.COMPETITION_EVOLUTION,
                title="Competition Evolution Forecast",
                description=f"Competition level predicted to change from {current_competition:.1f} "
                           f"to {predicted_competition:.1f} over the next 12 months.",
                time_horizon=TimeHorizon.MEDIUM_TERM,
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                predicted_value=predicted_competition,
                supporting_factors=[
                    f"Market growth rate: {market_growth*100:.1f}%",
                    "Historical competition patterns",
                    "Industry trend analysis"
                ],
                risk_factors=[
                    "New market entrants",
                    "Technology disruption",
                    "Economic changes"
                ],
                target_date=datetime.utcnow() + timedelta(days=365)
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting competition evolution: {e}")
            return self._create_default_prediction(PredictionType.COMPETITION_EVOLUTION)
    
    def optimize_timing(self, idea_data: Dict[str, Any]) -> Prediction:
        """Predict optimal timing for idea launch"""
        logger.info(f"Optimizing timing for: {idea_data.get('name', 'Unknown')}")
        
        try:
            # Analyze timing factors
            timing_factors = self._analyze_timing_factors(idea_data)
            
            # Find optimal launch window
            optimal_timing = self._calculate_optimal_timing(timing_factors)
            
            # Calculate confidence
            confidence = 0.75 if timing_factors['data_quality'] > 0.7 else 0.6
            
            prediction = Prediction(
                prediction_id=f"timing_optimization_{int(datetime.utcnow().timestamp())}",
                type=PredictionType.TIMING_OPTIMIZATION,
                title="Optimal Launch Timing",
                description=f"Optimal launch timing predicted for {optimal_timing['target_date'].strftime('%B %Y')} "
                           f"with {optimal_timing['success_multiplier']:.1f}x success probability.",
                time_horizon=TimeHorizon.SHORT_TERM,
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                predicted_value=optimal_timing,
                supporting_factors=optimal_timing['supporting_factors'],
                risk_factors=optimal_timing['risk_factors'],
                target_date=optimal_timing['target_date']
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error optimizing timing: {e}")
            return self._create_default_prediction(PredictionType.TIMING_OPTIMIZATION)
    
    def _update_trend_models(self, data_point: Dict[str, Any]):
        """Update trend models with new data"""
        # Extract trend indicators
        timestamp = data_point.get('timestamp', datetime.utcnow())
        
        # Update market size trends
        market_research = data_point.get('market_research', {})
        market_size = market_research.get('market_size', {}).get('total_addressable_market', 0)
        
        if market_size > 0:
            market_segment = data_point.get('niche_category', 'general')
            if market_segment not in self.trend_models:
                self.trend_models[market_segment] = {
                    'market_sizes': [],
                    'timestamps': [],
                    'quality_scores': []
                }
            
            self.trend_models[market_segment]['market_sizes'].append(market_size)
            self.trend_models[market_segment]['timestamps'].append(timestamp)
            
            # Add quality score if available
            custom_scores = data_point.get('custom_scores', {})
            overall_score = custom_scores.get('overall_score', 5.0)
            self.trend_models[market_segment]['quality_scores'].append(overall_score)
    
    def _predict_segment_opportunity(self, segment: str, horizon: TimeHorizon, 
                                   context: Dict[str, Any]) -> Optional[Prediction]:
        """Predict opportunity for a specific market segment"""
        
        # Get historical data for this segment
        segment_data = self.trend_models.get(segment, {})
        
        if not segment_data.get('market_sizes'):
            # No historical data, use market context
            base_opportunity = context.get('market_indicators', {}).get(segment, 0.7)
        else:
            # Calculate trend from historical data
            market_sizes = segment_data['market_sizes']
            if len(market_sizes) >= 2:
                growth_rate = (market_sizes[-1] - market_sizes[0]) / market_sizes[0]
                base_opportunity = min(1.0, 0.5 + growth_rate)
            else:
                base_opportunity = 0.6
        
        # Apply time horizon adjustments
        horizon_multipliers = {
            TimeHorizon.SHORT_TERM: 1.0,
            TimeHorizon.MEDIUM_TERM: 1.2,
            TimeHorizon.LONG_TERM: 1.5
        }
        
        adjusted_opportunity = base_opportunity * horizon_multipliers[horizon]
        adjusted_opportunity = min(1.0, adjusted_opportunity)
        
        # Calculate confidence
        data_quality = len(segment_data.get('market_sizes', [])) / 10.0  # Normalize
        confidence = 0.6 + min(0.3, data_quality)
        
        # Create prediction
        days_ahead = {
            TimeHorizon.SHORT_TERM: self.config.short_term_days,
            TimeHorizon.MEDIUM_TERM: self.config.medium_term_days,
            TimeHorizon.LONG_TERM: self.config.long_term_days
        }
        
        prediction = Prediction(
            prediction_id=f"market_opp_{segment}_{horizon.value}_{int(datetime.utcnow().timestamp())}",
            type=PredictionType.MARKET_OPPORTUNITY,
            title=f"{segment.title()} Market Opportunity - {horizon.value.title()}",
            description=f"Predicted market opportunity score of {adjusted_opportunity*100:.1f}% "
                       f"for {segment} segment over {horizon.value.replace('_', ' ')} horizon.",
            time_horizon=horizon,
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
            predicted_value=adjusted_opportunity,
            supporting_factors=[
                f"Historical growth trend",
                f"Market segment: {segment}",
                f"Time horizon: {horizon.value}"
            ],
            risk_factors=[
                "Market saturation",
                "Competitive pressure",
                "Economic uncertainty"
            ],
            target_date=datetime.utcnow() + timedelta(days=days_ahead[horizon])
        )
        
        return prediction
    
    def _analyze_trend_patterns(self, trend_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze patterns in trend data"""
        trend_analysis = {}
        
        # Group trends by category
        trend_categories = defaultdict(list)
        for trend in trend_data:
            category = trend.get('category', 'general')
            trend_categories[category].append(trend)
        
        # Analyze each category
        for category, trends in trend_categories.items():
            if len(trends) >= 2:
                # Calculate trend momentum
                recent_trends = sorted(trends, key=lambda x: x.get('timestamp', datetime.utcnow()))
                
                # Simple momentum calculation
                momentum = len(recent_trends) / 10.0  # Normalize
                
                # Calculate growth pattern
                if len(recent_trends) >= 3:
                    values = [trend.get('strength', 0.5) for trend in recent_trends]
                    growth_rate = (values[-1] - values[0]) / max(values[0], 0.1)
                else:
                    growth_rate = 0.1
                
                trend_analysis[category] = {
                    'momentum': momentum,
                    'growth_rate': growth_rate,
                    'trend_count': len(trends),
                    'latest_strength': recent_trends[-1].get('strength', 0.5),
                    'prediction_confidence': min(0.9, 0.5 + momentum)
                }
        
        return trend_analysis
    
    def _create_trend_forecast(self, trend_name: str, analysis: Dict[str, Any]) -> Optional[Prediction]:
        """Create trend forecast prediction"""
        
        momentum = analysis.get('momentum', 0.5)
        growth_rate = analysis.get('growth_rate', 0.1)
        confidence = analysis.get('prediction_confidence', 0.6)
        
        # Predict future trend strength
        current_strength = analysis.get('latest_strength', 0.5)
        predicted_strength = current_strength * (1 + growth_rate)
        predicted_strength = max(0.0, min(1.0, predicted_strength))
        
        prediction = Prediction(
            prediction_id=f"trend_forecast_{trend_name}_{int(datetime.utcnow().timestamp())}",
            type=PredictionType.TREND_FORECAST,
            title=f"{trend_name.title()} Trend Forecast",
            description=f"Trend strength predicted to reach {predicted_strength*100:.1f}% "
                       f"with {momentum*100:.1f}% momentum over next 6 months.",
            time_horizon=TimeHorizon.MEDIUM_TERM,
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
            predicted_value=predicted_strength,
            supporting_factors=[
                f"Current momentum: {momentum*100:.1f}%",
                f"Growth rate: {growth_rate*100:.1f}%",
                f"Historical trend data: {analysis.get('trend_count', 0)} points"
            ],
            risk_factors=[
                "Market saturation",
                "Technology shifts",
                "Consumer preference changes"
            ],
            target_date=datetime.utcnow() + timedelta(days=180)
        )
        
        return prediction
    
    def _extract_prediction_features(self, idea_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for prediction models"""
        features = {}
        
        # Market features
        market_research = idea_data.get('market_research', {})
        features['market_score'] = market_research.get('market_score', {}).get('total_score', 5.0) / 10.0
        
        market_size = market_research.get('market_size', {}).get('total_addressable_market', 1000000)
        features['market_size_log'] = math.log10(max(market_size, 1)) / 10.0  # Normalize
        
        # Competition features
        competitive_analysis = market_research.get('competitive_analysis', {})
        features['competition_level'] = competitive_analysis.get('competition_level', 5.0) / 10.0
        
        # Financial features
        financial_analysis = idea_data.get('financial_analysis', {})
        financial_metrics = financial_analysis.get('financial_metrics', {})
        features['roi'] = min(financial_metrics.get('roi_percentage', 100) / 500.0, 1.0)  # Normalize
        features['break_even'] = max(0, 1 - financial_metrics.get('break_even_point_months', 12) / 24.0)
        
        # Quality features
        custom_scores = idea_data.get('custom_scores', {})
        features['overall_quality'] = custom_scores.get('overall_score', 5.0) / 10.0
        
        # Validation features
        validation_evidence = idea_data.get('validation_evidence', {})
        sources_count = len(validation_evidence.get('sources', []))
        features['validation_strength'] = min(sources_count / 8.0, 1.0)  # Normalize to max 8 sources
        
        return features
    
    def _calculate_base_success_probability(self, features: Dict[str, float]) -> float:
        """Calculate base success probability from features"""
        # Weighted combination of features
        weights = {
            'market_score': 0.25,
            'market_size_log': 0.15,
            'competition_level': -0.15,  # Negative weight (higher competition = lower success)
            'roi': 0.20,
            'break_even': 0.10,
            'overall_quality': 0.20,
            'validation_strength': 0.15
        }
        
        probability = 0.5  # Base probability
        
        for feature, value in features.items():
            weight = weights.get(feature, 0)
            probability += weight * value
        
        return max(0.0, min(1.0, probability))
    
    def _calculate_timing_adjustment(self, idea_data: Dict[str, Any]) -> float:
        """Calculate timing adjustment factor"""
        # Simple timing analysis
        current_month = datetime.utcnow().month
        
        # Some months are better for launches (simplified)
        month_multipliers = {
            1: 0.9,   # January - slow start
            2: 0.95,  # February
            3: 1.1,   # March - good for B2B
            4: 1.05,  # April
            5: 1.0,   # May
            6: 0.95,  # June
            7: 0.9,   # July - summer slowdown
            8: 0.9,   # August
            9: 1.1,   # September - back to business
            10: 1.05, # October
            11: 0.95, # November
            12: 0.9   # December - holidays
        }
        
        return month_multipliers.get(current_month, 1.0)
    
    def _calculate_competition_adjustment(self, idea_data: Dict[str, Any]) -> float:
        """Calculate competition adjustment factor"""
        market_research = idea_data.get('market_research', {})
        competitive_analysis = market_research.get('competitive_analysis', {})
        competition_level = competitive_analysis.get('competition_level', 5.0)
        
        # Higher competition reduces success probability
        if competition_level <= 3.0:
            return 1.2  # Low competition boost
        elif competition_level <= 6.0:
            return 1.0  # Medium competition neutral
        elif competition_level <= 8.0:
            return 0.8  # High competition penalty
        else:
            return 0.6  # Very high competition significant penalty
    
    def _calculate_prediction_confidence(self, idea_data: Dict[str, Any], 
                                       features: Dict[str, float]) -> float:
        """Calculate confidence in prediction"""
        confidence = 0.6  # Base confidence
        
        # Adjust based on data completeness
        required_sections = ['market_research', 'financial_analysis', 'validation_evidence']
        present_sections = sum(1 for section in required_sections if idea_data.get(section))
        data_completeness = present_sections / len(required_sections)
        
        confidence += 0.2 * data_completeness
        
        # Adjust based on validation strength
        validation_strength = features.get('validation_strength', 0)
        confidence += 0.1 * validation_strength
        
        # Adjust based on historical data availability
        if len(self.historical_data) > 20:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _identify_supporting_factors(self, features: Dict[str, float], 
                                   idea_data: Dict[str, Any]) -> List[str]:
        """Identify factors supporting success"""
        factors = []
        
        if features.get('market_score', 0) > 0.7:
            factors.append("Strong market validation")
        
        if features.get('market_size_log', 0) > 0.6:
            factors.append("Large addressable market")
        
        if features.get('competition_level', 1) < 0.6:
            factors.append("Low competition environment")
        
        if features.get('roi', 0) > 0.6:
            factors.append("High ROI potential")
        
        if features.get('overall_quality', 0) > 0.7:
            factors.append("High quality score")
        
        if features.get('validation_strength', 0) > 0.6:
            factors.append("Strong validation evidence")
        
        return factors
    
    def _identify_risk_factors(self, features: Dict[str, float], 
                             idea_data: Dict[str, Any]) -> List[str]:
        """Identify risk factors"""
        factors = []
        
        if features.get('competition_level', 0) > 0.8:
            factors.append("High competition risk")
        
        if features.get('market_size_log', 1) < 0.4:
            factors.append("Limited market size")
        
        if features.get('break_even', 1) < 0.5:
            factors.append("Long break-even period")
        
        if features.get('validation_strength', 1) < 0.4:
            factors.append("Insufficient validation")
        
        # Add general risks
        factors.extend([
            "Market timing uncertainty",
            "Technology adoption risk",
            "Economic conditions"
        ])
        
        return factors
    
    def _predict_revenue_for_horizon(self, base_revenue: float, horizon: TimeHorizon, 
                                   idea_data: Dict[str, Any]) -> Optional[Prediction]:
        """Predict revenue for specific time horizon"""
        
        # Growth rate assumptions based on market and competition
        market_research = idea_data.get('market_research', {})
        market_score = market_research.get('market_score', {}).get('total_score', 5.0)
        competition_level = market_research.get('competitive_analysis', {}).get('competition_level', 5.0)
        
        # Base growth rate
        base_growth_rate = 0.2  # 20% annual growth
        
        # Adjust based on market score
        market_adjustment = (market_score - 5.0) / 10.0  # -0.5 to +0.5
        
        # Adjust based on competition
        competition_adjustment = (5.0 - competition_level) / 20.0  # -0.25 to +0.25
        
        adjusted_growth_rate = base_growth_rate + market_adjustment + competition_adjustment
        adjusted_growth_rate = max(0.0, min(1.0, adjusted_growth_rate))
        
        # Calculate predicted revenue based on horizon
        if horizon == TimeHorizon.SHORT_TERM:
            months = 3
            predicted_revenue = base_revenue * (1 + adjusted_growth_rate * months / 12)
        elif horizon == TimeHorizon.MEDIUM_TERM:
            months = 12
            predicted_revenue = base_revenue * (1 + adjusted_growth_rate)
        else:  # LONG_TERM
            years = 3
            predicted_revenue = base_revenue * ((1 + adjusted_growth_rate) ** years)
        
        # Calculate confidence
        confidence = 0.7 if horizon == TimeHorizon.SHORT_TERM else 0.6 if horizon == TimeHorizon.MEDIUM_TERM else 0.5
        
        prediction = Prediction(
            prediction_id=f"revenue_{horizon.value}_{int(datetime.utcnow().timestamp())}",
            type=PredictionType.REVENUE_PROJECTION,
            title=f"Revenue Projection - {horizon.value.title()}",
            description=f"Predicted revenue of ${predicted_revenue:,.0f} for {horizon.value.replace('_', ' ')} horizon "
                       f"with {adjusted_growth_rate*100:.1f}% growth rate.",
            time_horizon=horizon,
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
            predicted_value=predicted_revenue,
            supporting_factors=[
                f"Base revenue: ${base_revenue:,.0f}",
                f"Growth rate: {adjusted_growth_rate*100:.1f}%",
                f"Market score: {market_score:.1f}/10"
            ],
            risk_factors=[
                "Market volatility",
                "Competition impact",
                "Economic conditions"
            ]
        )
        
        return prediction
    
    def _predict_competition_change(self, current_competition: float, market_growth: float, 
                                  market_data: Dict[str, Any]) -> float:
        """Predict how competition will change"""
        
        # Higher market growth typically attracts more competition
        competition_increase = market_growth * self.config.competition_impact_factor
        
        # Market maturity factor
        market_size = market_data.get('market_size', {}).get('total_addressable_market', 1000000)
        if market_size > 1000000000:  # Large market
            competition_increase *= 1.5  # More attractive to competitors
        
        predicted_competition = current_competition + competition_increase
        return max(0.0, min(10.0, predicted_competition))
    
    def _analyze_timing_factors(self, idea_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze factors affecting optimal timing"""
        factors = {
            'market_readiness': 0.7,
            'competition_pressure': 0.5,
            'technology_maturity': 0.8,
            'economic_conditions': 0.6,
            'seasonal_factors': 0.7,
            'data_quality': 0.6
        }
        
        # Adjust based on available data
        market_research = idea_data.get('market_research', {})
        if market_research:
            factors['market_readiness'] = min(1.0, market_research.get('market_score', {}).get('total_score', 5.0) / 10.0 + 0.2)
            factors['data_quality'] += 0.2
        
        financial_analysis = idea_data.get('financial_analysis', {})
        if financial_analysis:
            factors['data_quality'] += 0.2
        
        return factors
    
    def _calculate_optimal_timing(self, timing_factors: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal timing based on factors"""
        
        # Simple timing optimization
        current_date = datetime.utcnow()
        
        # Find best month in next 12 months
        best_score = 0
        best_month = current_date
        
        for month_offset in range(1, 13):
            target_date = current_date + timedelta(days=30 * month_offset)
            
            # Calculate score for this month
            month_score = timing_factors['market_readiness']
            
            # Seasonal adjustments (simplified)
            if target_date.month in [3, 9, 10]:  # Good business months
                month_score += 0.1
            elif target_date.month in [7, 8, 12]:  # Slower months
                month_score -= 0.1
            
            if month_score > best_score:
                best_score = month_score
                best_month = target_date
        
        success_multiplier = 1.0 + (best_score - 0.7)  # Baseline 0.7
        
        return {
            'target_date': best_month,
            'success_multiplier': success_multiplier,
            'timing_score': best_score,
            'supporting_factors': [
                f"Market readiness: {timing_factors['market_readiness']*100:.0f}%",
                f"Optimal month: {best_month.strftime('%B %Y')}",
                f"Success multiplier: {success_multiplier:.1f}x"
            ],
            'risk_factors': [
                "Market conditions may change",
                "Competition may accelerate",
                "Economic uncertainty"
            ]
        }
    
    def _get_confidence_level(self, confidence: float) -> PredictionConfidence:
        """Convert confidence score to confidence level"""
        if confidence >= 0.9:
            return PredictionConfidence.VERY_HIGH
        elif confidence >= 0.8:
            return PredictionConfidence.HIGH
        elif confidence >= 0.6:
            return PredictionConfidence.MEDIUM
        elif confidence >= 0.4:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW
    
    def _create_default_prediction(self, prediction_type: PredictionType) -> Prediction:
        """Create default prediction when analysis fails"""
        return Prediction(
            prediction_id=f"default_{prediction_type.value}_{int(datetime.utcnow().timestamp())}",
            type=prediction_type,
            title=f"Default {prediction_type.value.title()} Prediction",
            description="Prediction generated with limited data. Results may be less accurate.",
            time_horizon=TimeHorizon.MEDIUM_TERM,
            confidence=0.4,
            confidence_level=PredictionConfidence.LOW,
            predicted_value=0.5,
            supporting_factors=["Limited historical data"],
            risk_factors=["High uncertainty due to insufficient data"]
        )
    
    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Get accuracy metrics for predictions"""
        return self.prediction_accuracy
    
    def validate_prediction(self, prediction_id: str, actual_outcome: Any) -> bool:
        """Validate a prediction against actual outcome"""
        # In a real implementation, this would update prediction accuracy metrics
        logger.info(f"Validating prediction {prediction_id} against actual outcome")
        return True
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of predictive analytics"""
        return {
            'historical_data_points': len(self.historical_data),
            'trend_models': len(self.trend_models),
            'market_forecasts': len(self.market_forecasts),
            'prediction_accuracy': self.prediction_accuracy,
            'config': {
                'trend_window_days': self.config.trend_window_days,
                'confidence_threshold': self.config.confidence_threshold,
                'time_horizons': {
                    'short_term': self.config.short_term_days,
                    'medium_term': self.config.medium_term_days,
                    'long_term': self.config.long_term_days
                }
            }
        }

