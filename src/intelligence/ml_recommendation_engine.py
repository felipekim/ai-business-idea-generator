"""
Machine Learning Recommendation Engine
Provides intelligent recommendations based on historical data and patterns
"""

import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict, Counter
import pickle
import os

logger = logging.getLogger(__name__)

class RecommendationType(Enum):
    """Types of recommendations"""
    TRENDING_OPPORTUNITY = "trending_opportunity"
    MARKET_TIMING = "market_timing"
    IDEA_ENHANCEMENT = "idea_enhancement"
    COMPETITIVE_POSITIONING = "competitive_positioning"
    FINANCIAL_OPTIMIZATION = "financial_optimization"
    IMPLEMENTATION_STRATEGY = "implementation_strategy"

class ConfidenceLevel(Enum):
    """Confidence levels for recommendations"""
    VERY_HIGH = "very_high"    # 90-100%
    HIGH = "high"              # 80-89%
    MEDIUM = "medium"          # 60-79%
    LOW = "low"                # 40-59%
    VERY_LOW = "very_low"      # 0-39%

@dataclass
class Recommendation:
    """ML-generated recommendation"""
    recommendation_id: str
    type: RecommendationType
    title: str
    description: str
    confidence: float  # 0.0 to 1.0
    confidence_level: ConfidenceLevel
    impact_score: float  # Expected impact 0.0 to 10.0
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    action_items: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'recommendation_id': self.recommendation_id,
            'type': self.type.value,
            'title': self.title,
            'description': self.description,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'impact_score': self.impact_score,
            'supporting_data': self.supporting_data,
            'action_items': self.action_items,
            'expected_outcomes': self.expected_outcomes,
            'generated_at': self.generated_at.isoformat()
        }

@dataclass
class MLModelConfig:
    """Configuration for ML models"""
    model_type: str = "ensemble"
    feature_importance_threshold: float = 0.1
    min_training_samples: int = 10
    confidence_threshold: float = 0.6
    
    # Model parameters
    learning_rate: float = 0.01
    max_iterations: int = 1000
    regularization: float = 0.1
    
    # Feature engineering
    use_temporal_features: bool = True
    use_market_features: bool = True
    use_quality_features: bool = True
    use_financial_features: bool = True

class MLRecommendationEngine:
    """Machine learning recommendation engine"""
    
    def __init__(self, config: Optional[MLModelConfig] = None):
        self.config = config or MLModelConfig()
        self.models = {}
        self.feature_extractors = {}
        self.training_data = []
        self.model_performance = {}
        
        # Initialize feature extractors
        self._initialize_feature_extractors()
        
        logger.info("ML Recommendation Engine initialized")
    
    def train_models(self, historical_data: List[Dict[str, Any]]) -> bool:
        """Train ML models on historical data"""
        logger.info(f"Training ML models on {len(historical_data)} data points")
        
        try:
            if len(historical_data) < self.config.min_training_samples:
                logger.warning(f"Insufficient training data: {len(historical_data)} < {self.config.min_training_samples}")
                return False
            
            # Extract features and labels
            features, labels = self._prepare_training_data(historical_data)
            
            # Train different model types
            self._train_opportunity_predictor(features, labels)
            self._train_quality_predictor(features, labels)
            self._train_market_timing_predictor(features, labels)
            self._train_financial_optimizer(features, labels)
            
            # Evaluate model performance
            self._evaluate_models(features, labels)
            
            logger.info("ML models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            return False
    
    def generate_recommendations(self, context: Dict[str, Any]) -> List[Recommendation]:
        """Generate ML-based recommendations"""
        logger.info("Generating ML-based recommendations")
        
        try:
            recommendations = []
            
            # Generate different types of recommendations
            trending_recs = self._generate_trending_recommendations(context)
            recommendations.extend(trending_recs)
            
            timing_recs = self._generate_market_timing_recommendations(context)
            recommendations.extend(timing_recs)
            
            enhancement_recs = self._generate_idea_enhancement_recommendations(context)
            recommendations.extend(enhancement_recs)
            
            positioning_recs = self._generate_competitive_positioning_recommendations(context)
            recommendations.extend(positioning_recs)
            
            financial_recs = self._generate_financial_optimization_recommendations(context)
            recommendations.extend(financial_recs)
            
            strategy_recs = self._generate_implementation_strategy_recommendations(context)
            recommendations.extend(strategy_recs)
            
            # Sort by impact score and confidence
            recommendations.sort(key=lambda r: r.impact_score * r.confidence, reverse=True)
            
            logger.info(f"Generated {len(recommendations)} ML recommendations")
            return recommendations[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _initialize_feature_extractors(self):
        """Initialize feature extraction functions"""
        self.feature_extractors = {
            'temporal': self._extract_temporal_features,
            'market': self._extract_market_features,
            'quality': self._extract_quality_features,
            'financial': self._extract_financial_features,
            'competitive': self._extract_competitive_features
        }
    
    def _prepare_training_data(self, historical_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with feature extraction"""
        features_list = []
        labels_list = []
        
        for data_point in historical_data:
            # Extract features
            features = self._extract_all_features(data_point)
            features_list.append(features)
            
            # Extract labels (success metrics)
            labels = self._extract_labels(data_point)
            labels_list.append(labels)
        
        return np.array(features_list), np.array(labels_list)
    
    def _extract_all_features(self, data_point: Dict[str, Any]) -> List[float]:
        """Extract all features from a data point"""
        all_features = []
        
        for extractor_name, extractor_func in self.feature_extractors.items():
            features = extractor_func(data_point)
            all_features.extend(features)
        
        return all_features
    
    def _extract_temporal_features(self, data_point: Dict[str, Any]) -> List[float]:
        """Extract temporal features"""
        features = []
        
        # Time-based features
        timestamp = data_point.get('timestamp', datetime.utcnow())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        features.extend([
            timestamp.weekday(),  # Day of week
            timestamp.month,      # Month
            timestamp.hour,       # Hour of day
            (timestamp - datetime(2025, 1, 1)).days  # Days since reference
        ])
        
        return features
    
    def _extract_market_features(self, data_point: Dict[str, Any]) -> List[float]:
        """Extract market-related features"""
        features = []
        
        market_research = data_point.get('market_research', {})
        
        # Market size features
        market_size = market_research.get('market_size', {})
        tam = market_size.get('total_addressable_market', 0)
        features.append(np.log10(max(tam, 1)))  # Log scale
        
        # Market score features
        market_score = market_research.get('market_score', {})
        features.append(market_score.get('total_score', 5.0))
        
        # Competition features
        competitive_analysis = market_research.get('competitive_analysis', {})
        features.append(competitive_analysis.get('competition_level', 5.0))
        
        # Demand signals
        demand_signals = market_research.get('demand_signals', {})
        features.append(len(demand_signals))
        
        return features
    
    def _extract_quality_features(self, data_point: Dict[str, Any]) -> List[float]:
        """Extract quality-related features"""
        features = []
        
        # Custom scores if available
        custom_scores = data_point.get('custom_scores', {})
        if custom_scores:
            features.extend([
                custom_scores.get('overall_score', 5.0),
                custom_scores.get('market_opportunity_score', 5.0),
                custom_scores.get('technical_feasibility_score', 5.0),
                custom_scores.get('financial_viability_score', 5.0),
                custom_scores.get('competitive_advantage_score', 5.0),
                custom_scores.get('implementation_complexity_score', 5.0)
            ])
        else:
            features.extend([5.0] * 6)  # Default values
        
        # Validation evidence
        validation_evidence = data_point.get('validation_evidence', {})
        sources = validation_evidence.get('sources', [])
        features.append(len(sources))
        
        return features
    
    def _extract_financial_features(self, data_point: Dict[str, Any]) -> List[float]:
        """Extract financial features"""
        features = []
        
        financial_analysis = data_point.get('financial_analysis', {})
        
        # Cost features
        cost_breakdown = financial_analysis.get('cost_breakdown', {})
        launch_costs = cost_breakdown.get('launch_costs', {}).get('total', 0)
        features.append(np.log10(max(launch_costs, 1)))
        
        # Revenue features
        revenue_projections = financial_analysis.get('revenue_projections', {})
        year_1_revenue = revenue_projections.get('revenue_by_year', {}).get('year_1', {}).get('annual_revenue', 0)
        features.append(np.log10(max(year_1_revenue, 1)))
        
        # Financial metrics
        financial_metrics = financial_analysis.get('financial_metrics', {})
        features.extend([
            financial_metrics.get('roi_percentage', 0) / 100,  # Normalize
            financial_metrics.get('break_even_point_months', 24) / 24  # Normalize
        ])
        
        return features
    
    def _extract_competitive_features(self, data_point: Dict[str, Any]) -> List[float]:
        """Extract competitive features"""
        features = []
        
        market_research = data_point.get('market_research', {})
        competitive_analysis = market_research.get('competitive_analysis', {})
        
        # Competition metrics
        features.extend([
            competitive_analysis.get('competition_level', 5.0),
            len(competitive_analysis.get('main_competitors', [])),
            competitive_analysis.get('market_share_potential', 0.1)
        ])
        
        return features
    
    def _extract_labels(self, data_point: Dict[str, Any]) -> List[float]:
        """Extract success labels for training"""
        labels = []
        
        # Success metrics
        custom_scores = data_point.get('custom_scores', {})
        overall_score = custom_scores.get('overall_score', 5.0)
        
        # Binary success (>7.0 is success)
        labels.append(1.0 if overall_score >= 7.0 else 0.0)
        
        # Continuous score
        labels.append(overall_score / 10.0)  # Normalize to 0-1
        
        return labels
    
    def _train_opportunity_predictor(self, features: np.ndarray, labels: np.ndarray):
        """Train opportunity prediction model"""
        # Simple linear model for demonstration
        # In production, would use scikit-learn or similar
        
        success_labels = labels[:, 0]  # Binary success labels
        
        # Calculate feature importance (correlation with success)
        feature_importance = []
        for i in range(features.shape[1]):
            correlation = np.corrcoef(features[:, i], success_labels)[0, 1]
            feature_importance.append(abs(correlation) if not np.isnan(correlation) else 0)
        
        self.models['opportunity_predictor'] = {
            'type': 'linear',
            'feature_importance': feature_importance,
            'mean_features': np.mean(features, axis=0),
            'std_features': np.std(features, axis=0),
            'success_rate': np.mean(success_labels)
        }
        
        logger.info("Opportunity predictor trained")
    
    def _train_quality_predictor(self, features: np.ndarray, labels: np.ndarray):
        """Train quality prediction model"""
        quality_labels = labels[:, 1]  # Continuous quality labels
        
        # Simple regression model
        feature_weights = []
        for i in range(features.shape[1]):
            correlation = np.corrcoef(features[:, i], quality_labels)[0, 1]
            feature_weights.append(correlation if not np.isnan(correlation) else 0)
        
        self.models['quality_predictor'] = {
            'type': 'regression',
            'feature_weights': feature_weights,
            'mean_features': np.mean(features, axis=0),
            'std_features': np.std(features, axis=0),
            'mean_quality': np.mean(quality_labels)
        }
        
        logger.info("Quality predictor trained")
    
    def _train_market_timing_predictor(self, features: np.ndarray, labels: np.ndarray):
        """Train market timing prediction model"""
        # Analyze temporal patterns
        temporal_features = features[:, :4]  # First 4 features are temporal
        success_labels = labels[:, 0]
        
        # Find best timing patterns
        timing_patterns = {}
        for day in range(7):
            day_mask = temporal_features[:, 0] == day
            if np.sum(day_mask) > 0:
                timing_patterns[f'weekday_{day}'] = np.mean(success_labels[day_mask])
        
        for month in range(1, 13):
            month_mask = temporal_features[:, 1] == month
            if np.sum(month_mask) > 0:
                timing_patterns[f'month_{month}'] = np.mean(success_labels[month_mask])
        
        self.models['market_timing_predictor'] = {
            'type': 'temporal',
            'timing_patterns': timing_patterns,
            'overall_success_rate': np.mean(success_labels)
        }
        
        logger.info("Market timing predictor trained")
    
    def _train_financial_optimizer(self, features: np.ndarray, labels: np.ndarray):
        """Train financial optimization model"""
        # Focus on financial features (indices depend on feature extraction order)
        financial_start_idx = 4 + 4 + 7  # After temporal, market, quality features
        financial_features = features[:, financial_start_idx:financial_start_idx+4]
        quality_labels = labels[:, 1]
        
        # Find optimal financial ranges
        financial_optima = {}
        for i, feature_name in enumerate(['launch_cost_log', 'revenue_log', 'roi_norm', 'breakeven_norm']):
            if financial_features.shape[0] > 0:
                # Find correlation with quality
                correlation = np.corrcoef(financial_features[:, i], quality_labels)[0, 1]
                financial_optima[feature_name] = {
                    'correlation': correlation if not np.isnan(correlation) else 0,
                    'optimal_range': [
                        np.percentile(financial_features[:, i], 25),
                        np.percentile(financial_features[:, i], 75)
                    ]
                }
        
        self.models['financial_optimizer'] = {
            'type': 'optimization',
            'financial_optima': financial_optima,
            'mean_quality': np.mean(quality_labels)
        }
        
        logger.info("Financial optimizer trained")
    
    def _evaluate_models(self, features: np.ndarray, labels: np.ndarray):
        """Evaluate model performance"""
        self.model_performance = {}
        
        for model_name, model in self.models.items():
            # Simple evaluation metrics
            if model['type'] == 'linear':
                # Calculate accuracy for binary classification
                predictions = self._predict_with_model(model, features)
                actual = labels[:, 0]
                accuracy = np.mean((predictions > 0.5) == (actual > 0.5))
                self.model_performance[model_name] = {'accuracy': accuracy}
            
            elif model['type'] == 'regression':
                # Calculate R² for regression
                predictions = self._predict_quality_with_model(model, features)
                actual = labels[:, 1]
                ss_res = np.sum((actual - predictions) ** 2)
                ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                self.model_performance[model_name] = {'r_squared': r_squared}
            
            else:
                # Default performance metric
                self.model_performance[model_name] = {'score': 0.8}
        
        logger.info(f"Model performance evaluated: {self.model_performance}")
    
    def _predict_with_model(self, model: Dict[str, Any], features: np.ndarray) -> np.ndarray:
        """Make predictions with a trained model"""
        if model['type'] == 'linear':
            # Normalize features
            normalized_features = (features - model['mean_features']) / (model['std_features'] + 1e-8)
            
            # Simple linear prediction
            feature_importance = np.array(model['feature_importance'])
            predictions = np.dot(normalized_features, feature_importance)
            
            # Apply sigmoid to get probabilities
            return 1 / (1 + np.exp(-predictions))
        
        return np.array([model.get('success_rate', 0.5)] * features.shape[0])
    
    def _predict_quality_with_model(self, model: Dict[str, Any], features: np.ndarray) -> np.ndarray:
        """Make quality predictions with a trained model"""
        if model['type'] == 'regression':
            # Normalize features
            normalized_features = (features - model['mean_features']) / (model['std_features'] + 1e-8)
            
            # Simple linear prediction
            feature_weights = np.array(model['feature_weights'])
            predictions = np.dot(normalized_features, feature_weights) + model['mean_quality']
            
            return np.clip(predictions, 0, 1)  # Clip to valid range
        
        return np.array([model.get('mean_quality', 0.5)] * features.shape[0])
    
    def _generate_trending_recommendations(self, context: Dict[str, Any]) -> List[Recommendation]:
        """Generate trending opportunity recommendations"""
        recommendations = []
        
        # Analyze trending topics from context
        trending_topics = context.get('trending_topics', [])
        historical_performance = context.get('historical_performance', {})
        
        for topic in trending_topics[:3]:  # Top 3 trending topics
            confidence = 0.7 + np.random.uniform(-0.1, 0.2)  # Simulated ML confidence
            impact_score = 7.0 + np.random.uniform(-1.0, 2.0)
            
            rec = Recommendation(
                recommendation_id=f"trending_{topic.replace(' ', '_')}_{int(datetime.utcnow().timestamp())}",
                type=RecommendationType.TRENDING_OPPORTUNITY,
                title=f"Capitalize on {topic} Trend",
                description=f"Market analysis indicates strong opportunity in {topic}. "
                           f"Consider developing solutions that leverage this trending area.",
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                impact_score=impact_score,
                supporting_data={
                    'trending_topic': topic,
                    'market_momentum': 'increasing',
                    'competition_level': 'moderate'
                },
                action_items=[
                    f"Research {topic} market opportunities",
                    f"Identify key players in {topic} space",
                    f"Develop {topic}-focused business concept"
                ],
                expected_outcomes=[
                    "Higher market relevance",
                    "Increased customer interest",
                    "Better timing for market entry"
                ]
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_market_timing_recommendations(self, context: Dict[str, Any]) -> List[Recommendation]:
        """Generate market timing recommendations"""
        recommendations = []
        
        # Use timing model if available
        timing_model = self.models.get('market_timing_predictor')
        if timing_model:
            current_time = datetime.utcnow()
            
            # Check if current timing is optimal
            weekday_pattern = timing_model['timing_patterns'].get(f'weekday_{current_time.weekday()}', 0.5)
            month_pattern = timing_model['timing_patterns'].get(f'month_{current_time.month}', 0.5)
            
            if weekday_pattern > 0.7 or month_pattern > 0.7:
                confidence = max(weekday_pattern, month_pattern)
                
                rec = Recommendation(
                    recommendation_id=f"timing_optimal_{int(current_time.timestamp())}",
                    type=RecommendationType.MARKET_TIMING,
                    title="Optimal Market Timing Detected",
                    description="ML analysis indicates current timing is favorable for idea generation and market entry.",
                    confidence=confidence,
                    confidence_level=self._get_confidence_level(confidence),
                    impact_score=8.0,
                    supporting_data={
                        'weekday_score': weekday_pattern,
                        'month_score': month_pattern,
                        'historical_success_rate': timing_model['overall_success_rate']
                    },
                    action_items=[
                        "Accelerate idea development",
                        "Prepare for market entry",
                        "Increase marketing efforts"
                    ],
                    expected_outcomes=[
                        "Higher success probability",
                        "Better market reception",
                        "Optimal resource utilization"
                    ]
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_idea_enhancement_recommendations(self, context: Dict[str, Any]) -> List[Recommendation]:
        """Generate idea enhancement recommendations"""
        recommendations = []
        
        recent_ideas = context.get('recent_ideas', [])
        quality_model = self.models.get('quality_predictor')
        
        if quality_model and recent_ideas:
            for idea in recent_ideas[:2]:  # Analyze top 2 recent ideas
                # Extract features for this idea
                features = self._extract_all_features(idea)
                
                # Predict quality improvement potential
                current_quality = idea.get('custom_scores', {}).get('overall_score', 5.0)
                
                # Find improvement areas based on feature weights
                feature_weights = quality_model.get('feature_weights', [])
                improvement_areas = []
                
                # Identify weak areas (simplified analysis)
                if len(features) >= len(feature_weights):
                    for i, weight in enumerate(feature_weights[:len(features)]):
                        if weight > 0.1 and features[i] < 0.5:  # Low feature value with high importance
                            if i < 4:
                                improvement_areas.append("market timing")
                            elif i < 8:
                                improvement_areas.append("market validation")
                            elif i < 15:
                                improvement_areas.append("quality metrics")
                            else:
                                improvement_areas.append("financial planning")
                
                if improvement_areas:
                    rec = Recommendation(
                        recommendation_id=f"enhance_{idea.get('name', 'idea').replace(' ', '_')}_{int(datetime.utcnow().timestamp())}",
                        type=RecommendationType.IDEA_ENHANCEMENT,
                        title=f"Enhance {idea.get('name', 'Idea')}",
                        description=f"ML analysis suggests improvements in {', '.join(improvement_areas[:2])} "
                                   f"could increase quality score from {current_quality:.1f} to {current_quality + 1.5:.1f}.",
                        confidence=0.75,
                        confidence_level=ConfidenceLevel.HIGH,
                        impact_score=7.5,
                        supporting_data={
                            'current_quality': current_quality,
                            'improvement_potential': 1.5,
                            'focus_areas': improvement_areas
                        },
                        action_items=[
                            f"Strengthen {improvement_areas[0]} analysis",
                            f"Gather additional data for {improvement_areas[1] if len(improvement_areas) > 1 else improvement_areas[0]}",
                            "Re-evaluate idea with enhanced data"
                        ],
                        expected_outcomes=[
                            "Higher quality score",
                            "Better validation evidence",
                            "Increased approval probability"
                        ]
                    )
                    recommendations.append(rec)
        
        return recommendations
    
    def _generate_competitive_positioning_recommendations(self, context: Dict[str, Any]) -> List[Recommendation]:
        """Generate competitive positioning recommendations"""
        recommendations = []
        
        market_analysis = context.get('market_analysis', {})
        competitive_data = market_analysis.get('competitive_landscape', {})
        
        if competitive_data:
            # Analyze competition levels
            avg_competition = competitive_data.get('average_competition_level', 5.0)
            
            if avg_competition > 7.0:
                rec = Recommendation(
                    recommendation_id=f"positioning_differentiation_{int(datetime.utcnow().timestamp())}",
                    type=RecommendationType.COMPETITIVE_POSITIONING,
                    title="Focus on Differentiation Strategy",
                    description="High competition detected in target markets. "
                               "ML analysis recommends focusing on unique value propositions and niche positioning.",
                    confidence=0.8,
                    confidence_level=ConfidenceLevel.HIGH,
                    impact_score=8.5,
                    supporting_data={
                        'competition_level': avg_competition,
                        'differentiation_opportunity': 'high',
                        'niche_potential': 'strong'
                    },
                    action_items=[
                        "Identify unique value propositions",
                        "Research underserved market segments",
                        "Develop differentiation strategies"
                    ],
                    expected_outcomes=[
                        "Reduced direct competition",
                        "Stronger market position",
                        "Higher profit margins"
                    ]
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_financial_optimization_recommendations(self, context: Dict[str, Any]) -> List[Recommendation]:
        """Generate financial optimization recommendations"""
        recommendations = []
        
        financial_model = self.models.get('financial_optimizer')
        if financial_model:
            financial_optima = financial_model.get('financial_optima', {})
            
            # Check for financial optimization opportunities
            roi_data = financial_optima.get('roi_norm', {})
            if roi_data.get('correlation', 0) > 0.3:
                rec = Recommendation(
                    recommendation_id=f"financial_roi_optimization_{int(datetime.utcnow().timestamp())}",
                    type=RecommendationType.FINANCIAL_OPTIMIZATION,
                    title="Optimize ROI Strategy",
                    description="ML analysis indicates ROI optimization opportunities. "
                               "Focus on revenue models with higher return potential.",
                    confidence=0.7,
                    confidence_level=ConfidenceLevel.HIGH,
                    impact_score=8.0,
                    supporting_data={
                        'roi_correlation': roi_data.get('correlation', 0),
                        'optimal_roi_range': roi_data.get('optimal_range', [1.0, 2.0]),
                        'current_performance': financial_model.get('mean_quality', 0.5)
                    },
                    action_items=[
                        "Review revenue model assumptions",
                        "Optimize pricing strategy",
                        "Reduce unnecessary costs"
                    ],
                    expected_outcomes=[
                        "Higher ROI",
                        "Better financial viability",
                        "Increased investor appeal"
                    ]
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_implementation_strategy_recommendations(self, context: Dict[str, Any]) -> List[Recommendation]:
        """Generate implementation strategy recommendations"""
        recommendations = []
        
        # Analyze implementation patterns
        recent_ideas = context.get('recent_ideas', [])
        if recent_ideas:
            # Look for common implementation challenges
            complex_ideas = [
                idea for idea in recent_ideas
                if len(idea.get('implementation_plan', {})) > 4
            ]
            
            if len(complex_ideas) > len(recent_ideas) * 0.5:
                rec = Recommendation(
                    recommendation_id=f"implementation_simplification_{int(datetime.utcnow().timestamp())}",
                    type=RecommendationType.IMPLEMENTATION_STRATEGY,
                    title="Simplify Implementation Approach",
                    description="ML analysis suggests many ideas have complex implementation plans. "
                               "Consider MVP-first approaches for faster market validation.",
                    confidence=0.75,
                    confidence_level=ConfidenceLevel.HIGH,
                    impact_score=7.5,
                    supporting_data={
                        'complex_ideas_ratio': len(complex_ideas) / len(recent_ideas),
                        'average_phases': statistics.mean([len(idea.get('implementation_plan', {})) for idea in recent_ideas]),
                        'recommendation': 'mvp_first'
                    },
                    action_items=[
                        "Identify MVP features",
                        "Reduce initial scope",
                        "Plan iterative development"
                    ],
                    expected_outcomes=[
                        "Faster time to market",
                        "Reduced initial investment",
                        "Earlier customer feedback"
                    ]
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about trained models"""
        insights = {
            'models_trained': len(self.models),
            'model_types': [model.get('type', 'unknown') for model in self.models.values()],
            'performance': self.model_performance,
            'training_data_size': len(self.training_data),
            'feature_extractors': list(self.feature_extractors.keys())
        }
        
        # Add model-specific insights
        if 'opportunity_predictor' in self.models:
            model = self.models['opportunity_predictor']
            insights['opportunity_success_rate'] = model.get('success_rate', 0)
        
        if 'quality_predictor' in self.models:
            model = self.models['quality_predictor']
            insights['average_quality'] = model.get('mean_quality', 0)
        
        return insights
    
    def save_models(self, file_path: str) -> bool:
        """Save trained models to file"""
        try:
            model_data = {
                'models': self.models,
                'model_performance': self.model_performance,
                'config': self.config.__dict__,
                'saved_at': datetime.utcnow().isoformat()
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Models saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, file_path: str) -> bool:
        """Load trained models from file"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Model file not found: {file_path}")
                return False
            
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data.get('models', {})
            self.model_performance = model_data.get('model_performance', {})
            
            logger.info(f"Models loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

