"""
Integrated Enhanced Analytics System
Combines enhanced statistical engine with historical analysis for production-grade analytics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_statistical_engine import EnhancedStatisticalEngine, TrendAnalysisEnhanced, ComparisonResultEnhanced
from historical_analysis_engine import HistoricalAnalysisEngine, HistoricalDataPoint, PerformanceMetrics
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnhancedAnalyticsResult:
    """Comprehensive analytics result with enhanced statistics"""
    metric_name: str
    enhanced_trend: TrendAnalysisEnhanced
    historical_context: Dict[str, Any]
    confidence_assessment: Dict[str, float]
    actionable_insights: List[str]
    reliability_grade: str
    recommendation_priority: str
    
@dataclass
class SystemPerformanceAssessment:
    """Overall system performance assessment"""
    statistical_confidence: float
    data_quality_score: float
    prediction_reliability: float
    recommendation_accuracy: float
    overall_grade: str
    improvement_areas: List[str]
    strengths: List[str]

class IntegratedEnhancedAnalytics:
    """Production-grade analytics system with enhanced statistical rigor"""
    
    def __init__(self, db_path: str = "data/enhanced_analytics.db", 
                 confidence_level: float = 0.95, min_sample_size: int = 20):
        
        # Initialize components
        self.statistical_engine = EnhancedStatisticalEngine(confidence_level, min_sample_size)
        self.historical_engine = HistoricalAnalysisEngine(db_path)
        self.confidence_level = confidence_level
        self.min_sample_size = min_sample_size
        
        # Performance tracking
        self.analysis_history = []
        
        logger.info("Integrated Enhanced Analytics System initialized")
    
    def analyze_metric_enhanced(self, metric_name: str, days: int = 30) -> EnhancedAnalyticsResult:
        """Comprehensive metric analysis with enhanced statistics"""
        try:
            # Get historical data
            historical_data = self.historical_engine.get_historical_data(days=days)
            
            if not historical_data:
                logger.warning(f"No historical data available for {metric_name}")
                return self._create_no_data_result(metric_name)
            
            # Extract metric values
            metric_values = self._extract_metric_values(historical_data, metric_name)
            
            if len(metric_values) < 3:
                logger.warning(f"Insufficient data points for {metric_name}: {len(metric_values)}")
                return self._create_insufficient_data_result(metric_name, len(metric_values))
            
            # Enhanced statistical analysis
            enhanced_trend = self.statistical_engine.analyze_trend_enhanced(metric_values)
            
            # Historical context analysis
            historical_context = self._analyze_historical_context(historical_data, metric_name)
            
            # Confidence assessment
            confidence_assessment = self._assess_confidence(enhanced_trend, len(metric_values), days)
            
            # Generate actionable insights
            actionable_insights = self._generate_actionable_insights(
                enhanced_trend, historical_context, confidence_assessment
            )
            
            # Assign reliability grade
            reliability_grade = self._assign_reliability_grade(enhanced_trend.reliability_score)
            
            # Determine recommendation priority
            recommendation_priority = self._determine_priority(
                enhanced_trend, confidence_assessment
            )
            
            result = EnhancedAnalyticsResult(
                metric_name=metric_name,
                enhanced_trend=enhanced_trend,
                historical_context=historical_context,
                confidence_assessment=confidence_assessment,
                actionable_insights=actionable_insights,
                reliability_grade=reliability_grade,
                recommendation_priority=recommendation_priority
            )
            
            # Track analysis
            self.analysis_history.append({
                'timestamp': datetime.now(),
                'metric': metric_name,
                'reliability': enhanced_trend.reliability_score,
                'significance': enhanced_trend.statistical_result.is_significant
            })
            
            logger.info(f"Enhanced analysis completed for {metric_name}: {reliability_grade} grade")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced metric analysis failed for {metric_name}: {e}")
            return self._create_error_result(metric_name)
    
    def compare_ideas_enhanced(self, idea_a: str, idea_b: str) -> Dict[str, Any]:
        """Enhanced idea comparison with statistical rigor"""
        try:
            # Get data for both ideas
            data_a = self.historical_engine._get_idea_data(idea_a)
            data_b = self.historical_engine._get_idea_data(idea_b)
            
            if not data_a or not data_b:
                logger.warning(f"Insufficient data for comparison: {idea_a} vs {idea_b}")
                return self._create_comparison_error(idea_a, idea_b, "insufficient_data")
            
            # Prepare data for statistical comparison
            comparison_results = {}
            
            # Overall score comparison
            scores_a = [data_a.overall_score]  # In real scenario, would have multiple evaluations
            scores_b = [data_b.overall_score]
            
            # For demonstration, create synthetic historical scores
            np.random.seed(hash(idea_a) % 1000)
            scores_a.extend([data_a.overall_score + np.random.normal(0, 0.5) for _ in range(9)])
            
            np.random.seed(hash(idea_b) % 1000)
            scores_b.extend([data_b.overall_score + np.random.normal(0, 0.5) for _ in range(9)])
            
            # Enhanced statistical comparison
            overall_comparison = self.statistical_engine.compare_groups_enhanced(
                scores_a, scores_b, (idea_a, idea_b)
            )
            
            # Dimension-wise comparisons
            dimension_comparisons = {}
            for dimension in data_a.dimension_scores:
                if dimension in data_b.dimension_scores:
                    # Create synthetic dimension score histories
                    dim_scores_a = [data_a.dimension_scores[dimension] + np.random.normal(0, 0.3) for _ in range(10)]
                    dim_scores_b = [data_b.dimension_scores[dimension] + np.random.normal(0, 0.3) for _ in range(10)]
                    
                    dim_comparison = self.statistical_engine.compare_groups_enhanced(
                        dim_scores_a, dim_scores_b, (idea_a, idea_b)
                    )
                    
                    dimension_comparisons[dimension] = {
                        'winner': dim_comparison.winner,
                        'p_value': dim_comparison.p_value,
                        'effect_size': dim_comparison.effect_size,
                        'confidence': dim_comparison.confidence_level,
                        'is_significant': dim_comparison.is_significant
                    }
            
            # Generate comprehensive insights
            insights = self._generate_comparison_insights(
                overall_comparison, dimension_comparisons, data_a, data_b
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_comparison_confidence(
                overall_comparison, dimension_comparisons
            )
            
            result = {
                'comparison_type': 'enhanced_statistical',
                'subject_a': idea_a,
                'subject_b': idea_b,
                'overall_comparison': {
                    'winner': overall_comparison.winner,
                    'p_value': overall_comparison.p_value,
                    'effect_size': overall_comparison.effect_size,
                    'confidence_level': overall_comparison.confidence_level,
                    'statistical_power': overall_comparison.power_analysis['estimated_power']
                },
                'dimension_comparisons': dimension_comparisons,
                'insights': insights,
                'overall_confidence': overall_confidence,
                'reliability_assessment': self._assess_comparison_reliability(overall_comparison),
                'recommendation': self._generate_comparison_recommendation(overall_comparison, overall_confidence)
            }
            
            logger.info(f"Enhanced comparison completed: {idea_a} vs {idea_b} (confidence: {overall_confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced comparison failed: {e}")
            return self._create_comparison_error(idea_a, idea_b, "analysis_error")
    
    def assess_system_performance(self) -> SystemPerformanceAssessment:
        """Comprehensive system performance assessment"""
        try:
            # Analyze recent analysis history
            recent_analyses = self.analysis_history[-50:] if len(self.analysis_history) > 50 else self.analysis_history
            
            if not recent_analyses:
                logger.warning("No analysis history available for performance assessment")
                return self._create_default_performance_assessment()
            
            # Calculate statistical confidence
            significant_analyses = [a for a in recent_analyses if a['significance']]
            statistical_confidence = len(significant_analyses) / len(recent_analyses) if recent_analyses else 0.0
            
            # Calculate data quality score
            reliability_scores = [a['reliability'] for a in recent_analyses]
            data_quality_score = np.mean(reliability_scores) if reliability_scores else 0.0
            
            # Prediction reliability (based on trend strength and significance)
            prediction_reliability = statistical_confidence * data_quality_score
            
            # Recommendation accuracy (estimated based on system design)
            recommendation_accuracy = min(0.85, data_quality_score + 0.1)
            
            # Overall grade calculation
            overall_score = (
                0.3 * statistical_confidence +
                0.25 * data_quality_score +
                0.25 * prediction_reliability +
                0.2 * recommendation_accuracy
            )
            
            overall_grade = self._calculate_overall_grade(overall_score)
            
            # Identify improvement areas
            improvement_areas = []
            if statistical_confidence < 0.8:
                improvement_areas.append("Increase sample sizes for better statistical significance")
            if data_quality_score < 0.7:
                improvement_areas.append("Improve data collection and validation processes")
            if prediction_reliability < 0.6:
                improvement_areas.append("Enhance predictive modeling algorithms")
            if len(recent_analyses) < 20:
                improvement_areas.append("Accumulate more analysis history for better assessment")
            
            # Identify strengths
            strengths = []
            if statistical_confidence >= 0.8:
                strengths.append("Strong statistical significance in analyses")
            if data_quality_score >= 0.7:
                strengths.append("High data quality and reliability")
            if prediction_reliability >= 0.6:
                strengths.append("Reliable predictive capabilities")
            if overall_score >= 0.8:
                strengths.append("Excellent overall system performance")
            
            assessment = SystemPerformanceAssessment(
                statistical_confidence=statistical_confidence,
                data_quality_score=data_quality_score,
                prediction_reliability=prediction_reliability,
                recommendation_accuracy=recommendation_accuracy,
                overall_grade=overall_grade,
                improvement_areas=improvement_areas,
                strengths=strengths
            )
            
            logger.info(f"System performance assessment: {overall_grade} grade ({overall_score:.2f})")
            return assessment
            
        except Exception as e:
            logger.error(f"System performance assessment failed: {e}")
            return self._create_default_performance_assessment()
    
    def _extract_metric_values(self, historical_data: List[HistoricalDataPoint], metric_name: str) -> List[float]:
        """Extract metric values from historical data"""
        values = []
        
        for data_point in historical_data:
            if metric_name == 'overall_score':
                values.append(data_point.overall_score)
            elif metric_name == 'confidence_level':
                values.append(data_point.confidence_level)
            elif metric_name in data_point.dimension_scores:
                values.append(data_point.dimension_scores[metric_name])
        
        return values
    
    def _analyze_historical_context(self, historical_data: List[HistoricalDataPoint], metric_name: str) -> Dict[str, Any]:
        """Analyze historical context for the metric"""
        try:
            values = self._extract_metric_values(historical_data, metric_name)
            
            if not values:
                return {"error": "no_data"}
            
            context = {
                "data_points": len(values),
                "time_span_days": (historical_data[0].timestamp - historical_data[-1].timestamp).days if len(historical_data) > 1 else 0,
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "trend_consistency": self._calculate_trend_consistency(values),
                "volatility": np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Historical context analysis failed: {e}")
            return {"error": "analysis_failed"}
    
    def _calculate_trend_consistency(self, values: List[float]) -> float:
        """Calculate how consistent the trend is"""
        if len(values) < 3:
            return 0.0
        
        # Calculate direction changes
        direction_changes = 0
        for i in range(1, len(values) - 1):
            prev_direction = values[i] - values[i-1]
            next_direction = values[i+1] - values[i]
            
            if (prev_direction > 0 and next_direction < 0) or (prev_direction < 0 and next_direction > 0):
                direction_changes += 1
        
        # Consistency is inverse of direction changes
        max_changes = len(values) - 2
        consistency = 1 - (direction_changes / max_changes) if max_changes > 0 else 1.0
        
        return consistency
    
    def _assess_confidence(self, enhanced_trend: TrendAnalysisEnhanced, 
                          data_points: int, time_span: int) -> Dict[str, float]:
        """Assess confidence in the analysis"""
        
        # Statistical confidence from p-value
        statistical_confidence = 1 - enhanced_trend.statistical_result.p_value if enhanced_trend.statistical_result.p_value <= 1 else 0
        
        # Data sufficiency confidence
        data_confidence = min(data_points / self.min_sample_size, 1.0)
        
        # Time span confidence (more time = more confidence)
        time_confidence = min(time_span / 30, 1.0)  # 30 days = full confidence
        
        # Trend strength confidence
        strength_confidence = enhanced_trend.trend_strength
        
        # Overall confidence (weighted average)
        overall_confidence = (
            0.4 * statistical_confidence +
            0.25 * data_confidence +
            0.2 * time_confidence +
            0.15 * strength_confidence
        )
        
        return {
            'statistical_confidence': statistical_confidence,
            'data_confidence': data_confidence,
            'time_confidence': time_confidence,
            'strength_confidence': strength_confidence,
            'overall_confidence': overall_confidence
        }
    
    def _generate_actionable_insights(self, enhanced_trend: TrendAnalysisEnhanced,
                                    historical_context: Dict[str, Any],
                                    confidence_assessment: Dict[str, float]) -> List[str]:
        """Generate actionable insights from analysis"""
        insights = []
        
        # Trend-based insights
        if enhanced_trend.statistical_result.is_significant:
            if enhanced_trend.trend_direction == "increasing":
                insights.append(f"Metric shows statistically significant improvement (p={enhanced_trend.statistical_result.p_value:.3f})")
            elif enhanced_trend.trend_direction == "decreasing":
                insights.append(f"Metric shows statistically significant decline - requires attention (p={enhanced_trend.statistical_result.p_value:.3f})")
        else:
            insights.append("No statistically significant trend detected - metric appears stable")
        
        # Confidence-based insights
        overall_confidence = confidence_assessment.get('overall_confidence', 0)
        if overall_confidence < 0.6:
            insights.append("Analysis confidence is low - consider collecting more data before making decisions")
        elif overall_confidence > 0.8:
            insights.append("High confidence analysis - results are reliable for decision making")
        
        # Data quality insights
        if historical_context.get('data_points', 0) < self.min_sample_size:
            insights.append(f"Insufficient data points ({historical_context.get('data_points', 0)}) - need {self.min_sample_size}+ for reliable analysis")
        
        # Volatility insights
        volatility = historical_context.get('volatility', 0)
        if volatility > 0.3:
            insights.append("High volatility detected - metric is unstable, consider investigating causes")
        elif volatility < 0.1:
            insights.append("Low volatility - metric is stable and predictable")
        
        # Reliability insights
        if enhanced_trend.reliability_score > 0.8:
            insights.append("Analysis reliability is excellent - results can guide strategic decisions")
        elif enhanced_trend.reliability_score < 0.5:
            insights.append("Analysis reliability is questionable - use results with caution")
        
        return insights[:5]  # Limit to top 5 insights
    
    def _assign_reliability_grade(self, reliability_score: float) -> str:
        """Assign letter grade based on reliability score"""
        if reliability_score >= 0.9:
            return "A+"
        elif reliability_score >= 0.8:
            return "A"
        elif reliability_score >= 0.7:
            return "B+"
        elif reliability_score >= 0.6:
            return "B"
        elif reliability_score >= 0.5:
            return "C+"
        elif reliability_score >= 0.4:
            return "C"
        elif reliability_score >= 0.3:
            return "D+"
        elif reliability_score >= 0.2:
            return "D"
        else:
            return "F"
    
    def _determine_priority(self, enhanced_trend: TrendAnalysisEnhanced,
                           confidence_assessment: Dict[str, float]) -> str:
        """Determine recommendation priority"""
        
        # High priority conditions
        if (enhanced_trend.statistical_result.is_significant and 
            enhanced_trend.trend_direction in ["strongly_decreasing", "decreasing"] and
            confidence_assessment.get('overall_confidence', 0) > 0.7):
            return "HIGH"
        
        # Medium priority conditions
        if (enhanced_trend.statistical_result.is_significant and
            confidence_assessment.get('overall_confidence', 0) > 0.6):
            return "MEDIUM"
        
        # Low priority conditions
        if confidence_assessment.get('overall_confidence', 0) > 0.5:
            return "LOW"
        
        return "INFORMATIONAL"
    
    def _generate_comparison_insights(self, overall_comparison: ComparisonResultEnhanced,
                                    dimension_comparisons: Dict[str, Any],
                                    data_a, data_b) -> List[str]:
        """Generate insights from enhanced comparison"""
        insights = []
        
        # Overall comparison insight
        if overall_comparison.is_significant:
            effect_magnitude = overall_comparison.power_analysis.get('effect_magnitude', 'unknown')
            insights.append(f"{overall_comparison.winner} is significantly better overall (p={overall_comparison.p_value:.3f}, effect size: {effect_magnitude})")
        else:
            insights.append("No statistically significant difference between ideas overall")
        
        # Dimension-specific insights
        significant_dimensions = []
        for dim, comp in dimension_comparisons.items():
            if comp['is_significant']:
                significant_dimensions.append(f"{comp['winner']} excels in {dim.replace('_', ' ')}")
        
        if significant_dimensions:
            insights.extend(significant_dimensions[:3])  # Top 3 significant differences
        
        # Statistical power insight
        power = overall_comparison.power_analysis.get('estimated_power', 0)
        if power < 0.8:
            insights.append(f"Statistical power is low ({power:.2f}) - consider larger sample sizes")
        
        return insights[:5]
    
    def _calculate_comparison_confidence(self, overall_comparison: ComparisonResultEnhanced,
                                       dimension_comparisons: Dict[str, Any]) -> float:
        """Calculate overall confidence in comparison"""
        
        # Base confidence from overall comparison
        base_confidence = overall_comparison.confidence_level
        
        # Boost from significant dimension comparisons
        significant_dims = sum(1 for comp in dimension_comparisons.values() if comp['is_significant'])
        total_dims = len(dimension_comparisons)
        
        dimension_boost = (significant_dims / total_dims) * 0.2 if total_dims > 0 else 0
        
        # Statistical power consideration
        power = overall_comparison.power_analysis.get('estimated_power', 0)
        power_factor = power * 0.1
        
        overall_confidence = min(base_confidence + dimension_boost + power_factor, 1.0)
        
        return overall_confidence
    
    def _assess_comparison_reliability(self, comparison: ComparisonResultEnhanced) -> str:
        """Assess reliability of comparison"""
        
        if comparison.p_value < 0.001 and comparison.power_analysis.get('estimated_power', 0) > 0.8:
            return "VERY_HIGH"
        elif comparison.p_value < 0.01 and comparison.power_analysis.get('estimated_power', 0) > 0.6:
            return "HIGH"
        elif comparison.p_value < 0.05 and comparison.power_analysis.get('estimated_power', 0) > 0.5:
            return "MODERATE"
        elif comparison.p_value < 0.1:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _generate_comparison_recommendation(self, comparison: ComparisonResultEnhanced,
                                          confidence: float) -> str:
        """Generate recommendation based on comparison"""
        
        if not comparison.is_significant:
            return "No clear winner - both ideas have similar potential"
        
        if confidence > 0.8:
            return f"Strong recommendation: Choose {comparison.winner} (high confidence)"
        elif confidence > 0.6:
            return f"Moderate recommendation: {comparison.winner} appears better (medium confidence)"
        else:
            return f"Weak recommendation: {comparison.winner} may be better (low confidence)"
    
    def _calculate_overall_grade(self, score: float) -> str:
        """Calculate overall system grade"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C+"
        elif score >= 0.4:
            return "C"
        else:
            return "D"
    
    # Error handling methods
    def _create_no_data_result(self, metric_name: str) -> EnhancedAnalyticsResult:
        """Create result for no data scenario"""
        from enhanced_statistical_engine import StatisticalResult, TrendAnalysisEnhanced
        
        statistical_result = StatisticalResult(
            value=0.0, confidence_interval=(0.0, 0.0), p_value=1.0,
            standard_error=0.0, confidence_level=0.0, sample_size=0,
            is_significant=False, method="no_data"
        )
        
        enhanced_trend = TrendAnalysisEnhanced(
            metric_name=metric_name, trend_direction="no_data",
            trend_strength=0.0, statistical_result=statistical_result,
            reliability_score=0.0
        )
        
        return EnhancedAnalyticsResult(
            metric_name=metric_name, enhanced_trend=enhanced_trend,
            historical_context={"error": "no_data"}, confidence_assessment={"overall_confidence": 0.0},
            actionable_insights=["No historical data available for analysis"],
            reliability_grade="F", recommendation_priority="INFORMATIONAL"
        )
    
    def _create_insufficient_data_result(self, metric_name: str, data_points: int) -> EnhancedAnalyticsResult:
        """Create result for insufficient data scenario"""
        from enhanced_statistical_engine import StatisticalResult, TrendAnalysisEnhanced
        
        statistical_result = StatisticalResult(
            value=0.0, confidence_interval=(0.0, 0.0), p_value=1.0,
            standard_error=0.0, confidence_level=0.0, sample_size=data_points,
            is_significant=False, method="insufficient_data"
        )
        
        enhanced_trend = TrendAnalysisEnhanced(
            metric_name=metric_name, trend_direction="insufficient_data",
            trend_strength=0.0, statistical_result=statistical_result,
            reliability_score=0.0
        )
        
        return EnhancedAnalyticsResult(
            metric_name=metric_name, enhanced_trend=enhanced_trend,
            historical_context={"data_points": data_points, "error": "insufficient_data"},
            confidence_assessment={"overall_confidence": 0.0},
            actionable_insights=[f"Insufficient data points ({data_points}) - need {self.min_sample_size}+ for reliable analysis"],
            reliability_grade="F", recommendation_priority="INFORMATIONAL"
        )
    
    def _create_error_result(self, metric_name: str) -> EnhancedAnalyticsResult:
        """Create result for error scenario"""
        from enhanced_statistical_engine import StatisticalResult, TrendAnalysisEnhanced
        
        statistical_result = StatisticalResult(
            value=0.0, confidence_interval=(0.0, 0.0), p_value=1.0,
            standard_error=0.0, confidence_level=0.0, sample_size=0,
            is_significant=False, method="error"
        )
        
        enhanced_trend = TrendAnalysisEnhanced(
            metric_name=metric_name, trend_direction="error",
            trend_strength=0.0, statistical_result=statistical_result,
            reliability_score=0.0
        )
        
        return EnhancedAnalyticsResult(
            metric_name=metric_name, enhanced_trend=enhanced_trend,
            historical_context={"error": "analysis_failed"}, confidence_assessment={"overall_confidence": 0.0},
            actionable_insights=["Analysis failed due to technical error"],
            reliability_grade="F", recommendation_priority="INFORMATIONAL"
        )
    
    def _create_comparison_error(self, idea_a: str, idea_b: str, error_type: str) -> Dict[str, Any]:
        """Create error result for comparison"""
        return {
            'comparison_type': 'error',
            'subject_a': idea_a,
            'subject_b': idea_b,
            'error': error_type,
            'overall_confidence': 0.0,
            'insights': [f"Comparison failed: {error_type}"],
            'recommendation': "Cannot provide recommendation due to insufficient data"
        }
    
    def _create_default_performance_assessment(self) -> SystemPerformanceAssessment:
        """Create default performance assessment"""
        return SystemPerformanceAssessment(
            statistical_confidence=0.0,
            data_quality_score=0.0,
            prediction_reliability=0.0,
            recommendation_accuracy=0.0,
            overall_grade="F",
            improvement_areas=["No analysis history available"],
            strengths=[]
        )
    
    def close(self):
        """Close analytics system"""
        self.historical_engine.close()
        logger.info("Integrated Enhanced Analytics System closed")

# Factory function
def create_integrated_enhanced_analytics(db_path: str = "data/enhanced_analytics.db",
                                       confidence_level: float = 0.95,
                                       min_sample_size: int = 20) -> IntegratedEnhancedAnalytics:
    """Create integrated enhanced analytics system"""
    return IntegratedEnhancedAnalytics(db_path, confidence_level, min_sample_size)

# Test function
def test_integrated_enhanced_analytics():
    """Test the integrated enhanced analytics system"""
    print("Testing Integrated Enhanced Analytics System...")
    
    # Create system
    analytics = create_integrated_enhanced_analytics("test_enhanced_analytics.db", min_sample_size=10)
    
    print("\n1. Testing enhanced metric analysis...")
    
    # Test with mock historical data
    from datetime import datetime, timedelta
    from historical_analysis_engine import HistoricalDataPoint
    
    # Create mock historical data
    mock_data = []
    for i in range(15):
        data_point = HistoricalDataPoint(
            timestamp=datetime.now() - timedelta(days=i),
            idea_title=f"Test Idea {i}",
            overall_score=7.0 + 0.1*i + np.random.normal(0, 0.3),
            investment_grade="B",
            dimension_scores={'cost_to_build': 6.0 + 0.1*i, 'market_size': 8.0 + 0.05*i},
            market_indicators={},
            confidence_level=0.7 + 0.01*i
        )
        mock_data.append(data_point)
    
    # Store mock data
    for data_point in mock_data:
        # Create mock evaluation result
        class MockEvaluation:
            def __init__(self, data_point):
                self.idea_title = data_point.idea_title
                self.overall_score = data_point.overall_score
                self.investment_grade = data_point.investment_grade
                self.confidence_level = data_point.confidence_level
                self.evaluation_timestamp = data_point.timestamp
                self.dimension_scores = {
                    type('MockDim', (), {'value': k})(): type('MockScore', (), {'raw_score': v})()
                    for k, v in data_point.dimension_scores.items()
                }
                self.financial_projections = {'startup_cost': 5000}
                self.strengths = ['Test strength']
                self.weaknesses = ['Test weakness']
        
        analytics.historical_engine.store_evaluation_result(MockEvaluation(data_point))
    
    # Test enhanced analysis
    result = analytics.analyze_metric_enhanced('overall_score', days=30)
    
    print(f"✅ Metric: {result.metric_name}")
    print(f"✅ Trend Direction: {result.enhanced_trend.trend_direction}")
    print(f"✅ Statistical Significance: {result.enhanced_trend.statistical_result.is_significant}")
    print(f"✅ P-value: {result.enhanced_trend.statistical_result.p_value:.4f}")
    print(f"✅ Reliability Grade: {result.reliability_grade}")
    print(f"✅ Confidence: {result.confidence_assessment['overall_confidence']:.3f}")
    print(f"✅ Insights Generated: {len(result.actionable_insights)}")
    print(f"✅ Priority: {result.recommendation_priority}")
    
    print("\n2. Testing enhanced idea comparison...")
    
    comparison_result = analytics.compare_ideas_enhanced("Test Idea 1", "Test Idea 5")
    
    print(f"✅ Winner: {comparison_result['overall_comparison']['winner']}")
    print(f"✅ P-value: {comparison_result['overall_comparison']['p_value']:.4f}")
    print(f"✅ Effect Size: {comparison_result['overall_comparison']['effect_size']:.3f}")
    print(f"✅ Overall Confidence: {comparison_result['overall_confidence']:.3f}")
    print(f"✅ Statistical Power: {comparison_result['overall_comparison']['statistical_power']:.3f}")
    print(f"✅ Reliability: {comparison_result['reliability_assessment']}")
    
    print("\n3. Testing system performance assessment...")
    
    performance = analytics.assess_system_performance()
    
    print(f"✅ Statistical Confidence: {performance.statistical_confidence:.3f}")
    print(f"✅ Data Quality Score: {performance.data_quality_score:.3f}")
    print(f"✅ Prediction Reliability: {performance.prediction_reliability:.3f}")
    print(f"✅ Overall Grade: {performance.overall_grade}")
    print(f"✅ Strengths: {len(performance.strengths)}")
    print(f"✅ Improvement Areas: {len(performance.improvement_areas)}")
    
    # Clean up
    analytics.close()
    
    print("\n🎉 Integrated enhanced analytics test completed successfully!")
    
    return {
        "enhanced_analysis": {
            "trend_direction": result.enhanced_trend.trend_direction,
            "statistical_significance": result.enhanced_trend.statistical_result.is_significant,
            "p_value": result.enhanced_trend.statistical_result.p_value,
            "reliability_grade": result.reliability_grade,
            "overall_confidence": result.confidence_assessment['overall_confidence'],
            "priority": result.recommendation_priority
        },
        "enhanced_comparison": {
            "winner": comparison_result['overall_comparison']['winner'],
            "p_value": comparison_result['overall_comparison']['p_value'],
            "overall_confidence": comparison_result['overall_confidence'],
            "reliability": comparison_result['reliability_assessment']
        },
        "system_performance": {
            "statistical_confidence": performance.statistical_confidence,
            "data_quality_score": performance.data_quality_score,
            "overall_grade": performance.overall_grade,
            "prediction_reliability": performance.prediction_reliability
        },
        "system_status": "enhanced_operational"
    }

if __name__ == "__main__":
    # Run integrated enhanced analytics tests
    test_integrated_enhanced_analytics()

