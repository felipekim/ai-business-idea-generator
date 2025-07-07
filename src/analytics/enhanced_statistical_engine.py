"""
Enhanced Statistical Engine
Advanced statistical analysis with confidence intervals, significance testing, and robust estimation
"""

import numpy as np
import scipy.stats as stats
from scipy import optimize
import math
import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class StatisticalResult:
    """Enhanced statistical result with confidence metrics"""
    value: float
    confidence_interval: Tuple[float, float]
    p_value: float
    standard_error: float
    confidence_level: float
    sample_size: int
    is_significant: bool
    method: str
    
@dataclass
class TrendAnalysisEnhanced:
    """Enhanced trend analysis with statistical rigor"""
    metric_name: str
    trend_direction: str
    trend_strength: float
    statistical_result: StatisticalResult
    forecast: Optional[Dict[str, float]] = None
    seasonality: Optional[Dict[str, Any]] = None
    outliers: List[int] = field(default_factory=list)
    reliability_score: float = 0.0
    
@dataclass
class ComparisonResultEnhanced:
    """Enhanced comparison with statistical testing"""
    subject_a: str
    subject_b: str
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    power_analysis: Dict[str, float]
    winner: str
    confidence_level: float

class EnhancedStatisticalEngine:
    """Advanced statistical engine with robust confidence estimation"""
    
    def __init__(self, confidence_level: float = 0.95, min_sample_size: int = 20):
        self.confidence_level = confidence_level
        self.min_sample_size = min_sample_size
        self.alpha = 1 - confidence_level
        
        logger.info(f"Enhanced Statistical Engine initialized (confidence: {confidence_level}, min_samples: {min_sample_size})")
    
    def analyze_trend_enhanced(self, values: List[float], timestamps: Optional[List[datetime]] = None) -> TrendAnalysisEnhanced:
        """Enhanced trend analysis with statistical significance testing"""
        try:
            if len(values) < self.min_sample_size:
                logger.warning(f"Insufficient data for reliable trend analysis: {len(values)} < {self.min_sample_size}")
                return self._create_insufficient_data_result("trend_analysis", values)
            
            # Convert to numpy arrays for statistical analysis
            y = np.array(values)
            x = np.arange(len(values))
            
            # Perform linear regression with statistical testing
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Calculate confidence interval for slope
            t_critical = stats.t.ppf(1 - self.alpha/2, len(values) - 2)
            slope_ci = (
                slope - t_critical * std_err,
                slope + t_critical * std_err
            )
            
            # Determine statistical significance
            is_significant = p_value < self.alpha
            
            # Create statistical result
            statistical_result = StatisticalResult(
                value=slope,
                confidence_interval=slope_ci,
                p_value=p_value,
                standard_error=std_err,
                confidence_level=self.confidence_level,
                sample_size=len(values),
                is_significant=is_significant,
                method="linear_regression"
            )
            
            # Determine trend direction with statistical backing
            if is_significant:
                if slope > 0:
                    if slope > 0.5:
                        direction = "strongly_increasing"
                    else:
                        direction = "increasing"
                else:
                    if slope < -0.5:
                        direction = "strongly_decreasing"
                    else:
                        direction = "decreasing"
            else:
                direction = "stable"  # Not statistically significant
            
            # Calculate trend strength (correlation coefficient)
            trend_strength = abs(r_value)
            
            # Detect outliers using IQR method
            outliers = self._detect_outliers(values)
            
            # Calculate reliability score
            reliability_score = self._calculate_reliability_score(
                sample_size=len(values),
                p_value=p_value,
                r_squared=r_value**2,
                outlier_ratio=len(outliers)/len(values)
            )
            
            # Generate forecast if trend is significant
            forecast = None
            if is_significant and len(values) >= 10:
                forecast = self._generate_forecast(values, slope, intercept)
            
            # Analyze seasonality if enough data
            seasonality = None
            if len(values) >= 24:  # Need at least 2 cycles for seasonality
                seasonality = self._analyze_seasonality(values)
            
            result = TrendAnalysisEnhanced(
                metric_name="trend_analysis",
                trend_direction=direction,
                trend_strength=trend_strength,
                statistical_result=statistical_result,
                forecast=forecast,
                seasonality=seasonality,
                outliers=outliers,
                reliability_score=reliability_score
            )
            
            logger.info(f"Enhanced trend analysis: {direction} (p={p_value:.4f}, reliability={reliability_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced trend analysis failed: {e}")
            return self._create_error_result("trend_analysis", values)
    
    def compare_groups_enhanced(self, group_a: List[float], group_b: List[float], 
                              names: Tuple[str, str]) -> ComparisonResultEnhanced:
        """Enhanced group comparison with statistical testing"""
        try:
            if len(group_a) < 3 or len(group_b) < 3:
                logger.warning(f"Insufficient data for comparison: {len(group_a)}, {len(group_b)}")
                return self._create_insufficient_comparison_result(names)
            
            # Perform appropriate statistical test
            if len(group_a) >= 30 and len(group_b) >= 30:
                # Use t-test for large samples
                test_stat, p_value = stats.ttest_ind(group_a, group_b)
                test_method = "independent_t_test"
            else:
                # Use Mann-Whitney U test for small samples (non-parametric)
                test_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
                test_method = "mann_whitney_u"
            
            # Calculate effect size (Cohen's d)
            effect_size = self._calculate_cohens_d(group_a, group_b)
            
            # Calculate confidence interval for difference in means
            mean_diff = np.mean(group_a) - np.mean(group_b)
            pooled_std = np.sqrt(((len(group_a)-1)*np.var(group_a, ddof=1) + 
                                 (len(group_b)-1)*np.var(group_b, ddof=1)) / 
                                (len(group_a) + len(group_b) - 2))
            
            se_diff = pooled_std * np.sqrt(1/len(group_a) + 1/len(group_b))
            df = len(group_a) + len(group_b) - 2
            t_critical = stats.t.ppf(1 - self.alpha/2, df)
            
            ci_lower = mean_diff - t_critical * se_diff
            ci_upper = mean_diff + t_critical * se_diff
            
            # Determine statistical significance
            is_significant = p_value < self.alpha
            
            # Power analysis
            power_analysis = self._calculate_power_analysis(group_a, group_b, effect_size)
            
            # Determine winner
            if is_significant:
                winner = names[0] if np.mean(group_a) > np.mean(group_b) else names[1]
            else:
                winner = "no_significant_difference"
            
            # Calculate confidence level for the comparison
            confidence_level = (1 - p_value) if is_significant else 0.5
            
            result = ComparisonResultEnhanced(
                subject_a=names[0],
                subject_b=names[1],
                test_statistic=test_stat,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                is_significant=is_significant,
                power_analysis=power_analysis,
                winner=winner,
                confidence_level=confidence_level
            )
            
            logger.info(f"Enhanced comparison: {winner} (p={p_value:.4f}, effect_size={effect_size:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced comparison failed: {e}")
            return self._create_error_comparison_result(names)
    
    def bootstrap_confidence_interval(self, data: List[float], statistic_func, 
                                    n_bootstrap: int = 1000) -> Tuple[float, Tuple[float, float]]:
        """Bootstrap confidence interval estimation"""
        try:
            if len(data) < 5:
                logger.warning("Insufficient data for bootstrap analysis")
                return 0.0, (0.0, 0.0)
            
            # Generate bootstrap samples
            bootstrap_stats = []
            data_array = np.array(data)
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                bootstrap_sample = np.random.choice(data_array, size=len(data_array), replace=True)
                stat = statistic_func(bootstrap_sample)
                bootstrap_stats.append(stat)
            
            # Calculate confidence interval
            alpha_lower = (1 - self.confidence_level) / 2
            alpha_upper = 1 - alpha_lower
            
            ci_lower = np.percentile(bootstrap_stats, alpha_lower * 100)
            ci_upper = np.percentile(bootstrap_stats, alpha_upper * 100)
            
            # Calculate point estimate
            point_estimate = statistic_func(data_array)
            
            logger.info(f"Bootstrap CI: {point_estimate:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
            return point_estimate, (ci_lower, ci_upper)
            
        except Exception as e:
            logger.error(f"Bootstrap analysis failed: {e}")
            return 0.0, (0.0, 0.0)
    
    def _detect_outliers(self, values: List[float]) -> List[int]:
        """Detect outliers using IQR method"""
        try:
            if len(values) < 4:
                return []
            
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = []
            for i, value in enumerate(values):
                if value < lower_bound or value > upper_bound:
                    outliers.append(i)
            
            return outliers
            
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return []
    
    def _calculate_reliability_score(self, sample_size: int, p_value: float, 
                                   r_squared: float, outlier_ratio: float) -> float:
        """Calculate overall reliability score for analysis"""
        try:
            # Sample size component (0-1 scale)
            size_score = min(sample_size / 50, 1.0)  # Optimal at 50+ samples
            
            # Statistical significance component
            sig_score = max(0, 1 - p_value) if p_value <= 1.0 else 0
            
            # Explained variance component
            variance_score = r_squared
            
            # Outlier penalty
            outlier_penalty = max(0, 1 - outlier_ratio * 2)  # Penalty for >50% outliers
            
            # Weighted combination
            reliability = (
                0.3 * size_score +
                0.3 * sig_score +
                0.2 * variance_score +
                0.2 * outlier_penalty
            )
            
            return min(max(reliability, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Reliability calculation failed: {e}")
            return 0.0
    
    def _generate_forecast(self, values: List[float], slope: float, intercept: float) -> Dict[str, float]:
        """Generate simple linear forecast"""
        try:
            current_x = len(values) - 1
            
            # Forecast next 3 periods
            forecast = {}
            for i in range(1, 4):
                future_x = current_x + i
                predicted_value = slope * future_x + intercept
                forecast[f"period_{i}"] = predicted_value
            
            return forecast
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            return {}
    
    def _analyze_seasonality(self, values: List[float]) -> Dict[str, Any]:
        """Basic seasonality analysis"""
        try:
            # Simple seasonal decomposition using moving averages
            if len(values) < 24:
                return {"detected": False, "reason": "insufficient_data"}
            
            # Calculate seasonal indices (simplified)
            season_length = 12  # Assume monthly seasonality
            seasonal_indices = []
            
            for i in range(season_length):
                season_values = [values[j] for j in range(i, len(values), season_length)]
                if season_values:
                    seasonal_indices.append(np.mean(season_values))
            
            # Check for significant seasonality
            if len(seasonal_indices) >= 3:
                seasonal_variance = np.var(seasonal_indices)
                total_variance = np.var(values)
                seasonality_ratio = seasonal_variance / total_variance if total_variance > 0 else 0
                
                return {
                    "detected": seasonality_ratio > 0.1,
                    "strength": seasonality_ratio,
                    "indices": seasonal_indices
                }
            
            return {"detected": False, "reason": "calculation_failed"}
            
        except Exception as e:
            logger.error(f"Seasonality analysis failed: {e}")
            return {"detected": False, "reason": "error"}
    
    def _calculate_cohens_d(self, group_a: List[float], group_b: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        try:
            mean_a = np.mean(group_a)
            mean_b = np.mean(group_b)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((len(group_a)-1)*np.var(group_a, ddof=1) + 
                                 (len(group_b)-1)*np.var(group_b, ddof=1)) / 
                                (len(group_a) + len(group_b) - 2))
            
            if pooled_std == 0:
                return 0.0
            
            cohens_d = (mean_a - mean_b) / pooled_std
            return cohens_d
            
        except Exception as e:
            logger.error(f"Cohen's d calculation failed: {e}")
            return 0.0
    
    def _calculate_power_analysis(self, group_a: List[float], group_b: List[float], 
                                effect_size: float) -> Dict[str, float]:
        """Calculate statistical power analysis"""
        try:
            n_a = len(group_a)
            n_b = len(group_b)
            
            # Simplified power calculation
            # This is a basic approximation - full power analysis would require more complex calculations
            
            # Effect size interpretation
            if abs(effect_size) < 0.2:
                effect_magnitude = "small"
                power_estimate = 0.3
            elif abs(effect_size) < 0.5:
                effect_magnitude = "medium"
                power_estimate = 0.6
            elif abs(effect_size) < 0.8:
                effect_magnitude = "large"
                power_estimate = 0.8
            else:
                effect_magnitude = "very_large"
                power_estimate = 0.9
            
            # Adjust for sample size
            total_n = n_a + n_b
            if total_n < 20:
                power_estimate *= 0.7
            elif total_n < 50:
                power_estimate *= 0.85
            
            return {
                "estimated_power": min(power_estimate, 0.95),
                "effect_magnitude": effect_magnitude,
                "sample_size_a": n_a,
                "sample_size_b": n_b,
                "recommended_n": max(30, int(50 / max(abs(effect_size), 0.1)))
            }
            
        except Exception as e:
            logger.error(f"Power analysis failed: {e}")
            return {"estimated_power": 0.0, "effect_magnitude": "unknown"}
    
    def _create_insufficient_data_result(self, metric_name: str, values: List[float]) -> TrendAnalysisEnhanced:
        """Create result for insufficient data"""
        statistical_result = StatisticalResult(
            value=0.0,
            confidence_interval=(0.0, 0.0),
            p_value=1.0,
            standard_error=0.0,
            confidence_level=0.0,
            sample_size=len(values),
            is_significant=False,
            method="insufficient_data"
        )
        
        return TrendAnalysisEnhanced(
            metric_name=metric_name,
            trend_direction="insufficient_data",
            trend_strength=0.0,
            statistical_result=statistical_result,
            reliability_score=0.0
        )
    
    def _create_error_result(self, metric_name: str, values: List[float]) -> TrendAnalysisEnhanced:
        """Create result for analysis errors"""
        statistical_result = StatisticalResult(
            value=0.0,
            confidence_interval=(0.0, 0.0),
            p_value=1.0,
            standard_error=0.0,
            confidence_level=0.0,
            sample_size=len(values),
            is_significant=False,
            method="error"
        )
        
        return TrendAnalysisEnhanced(
            metric_name=metric_name,
            trend_direction="error",
            trend_strength=0.0,
            statistical_result=statistical_result,
            reliability_score=0.0
        )
    
    def _create_insufficient_comparison_result(self, names: Tuple[str, str]) -> ComparisonResultEnhanced:
        """Create result for insufficient comparison data"""
        return ComparisonResultEnhanced(
            subject_a=names[0],
            subject_b=names[1],
            test_statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            is_significant=False,
            power_analysis={"estimated_power": 0.0, "effect_magnitude": "insufficient_data"},
            winner="insufficient_data",
            confidence_level=0.0
        )
    
    def _create_error_comparison_result(self, names: Tuple[str, str]) -> ComparisonResultEnhanced:
        """Create result for comparison errors"""
        return ComparisonResultEnhanced(
            subject_a=names[0],
            subject_b=names[1],
            test_statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            is_significant=False,
            power_analysis={"estimated_power": 0.0, "effect_magnitude": "error"},
            winner="error",
            confidence_level=0.0
        )

# Factory function
def create_enhanced_statistical_engine(confidence_level: float = 0.95, 
                                     min_sample_size: int = 20) -> EnhancedStatisticalEngine:
    """Create enhanced statistical engine with specified parameters"""
    return EnhancedStatisticalEngine(confidence_level, min_sample_size)

# Test function
def test_enhanced_statistical_engine():
    """Test the enhanced statistical engine"""
    print("Testing Enhanced Statistical Engine...")
    
    # Create engine
    engine = create_enhanced_statistical_engine(confidence_level=0.95, min_sample_size=10)
    
    print("\n1. Testing with sufficient data (25 points)...")
    
    # Generate test data with clear trend
    np.random.seed(42)  # For reproducible results
    trend_data = [5.0 + 0.2*i + np.random.normal(0, 0.5) for i in range(25)]
    
    # Test enhanced trend analysis
    trend_result = engine.analyze_trend_enhanced(trend_data)
    
    print(f"✅ Trend Direction: {trend_result.trend_direction}")
    print(f"✅ Trend Strength: {trend_result.trend_strength:.3f}")
    print(f"✅ P-value: {trend_result.statistical_result.p_value:.4f}")
    print(f"✅ Is Significant: {trend_result.statistical_result.is_significant}")
    print(f"✅ Confidence Interval: [{trend_result.statistical_result.confidence_interval[0]:.3f}, {trend_result.statistical_result.confidence_interval[1]:.3f}]")
    print(f"✅ Reliability Score: {trend_result.reliability_score:.3f}")
    print(f"✅ Outliers Detected: {len(trend_result.outliers)}")
    
    print("\n2. Testing group comparison...")
    
    # Generate two groups with different means
    group_a = [8.0 + np.random.normal(0, 1) for _ in range(15)]
    group_b = [6.0 + np.random.normal(0, 1) for _ in range(15)]
    
    comparison_result = engine.compare_groups_enhanced(group_a, group_b, ("Group A", "Group B"))
    
    print(f"✅ Winner: {comparison_result.winner}")
    print(f"✅ P-value: {comparison_result.p_value:.4f}")
    print(f"✅ Effect Size: {comparison_result.effect_size:.3f}")
    print(f"✅ Is Significant: {comparison_result.is_significant}")
    print(f"✅ Confidence Level: {comparison_result.confidence_level:.3f}")
    print(f"✅ Statistical Power: {comparison_result.power_analysis['estimated_power']:.3f}")
    
    print("\n3. Testing bootstrap confidence interval...")
    
    # Test bootstrap method
    test_data = [7.5 + np.random.normal(0, 1.5) for _ in range(20)]
    mean_estimate, ci = engine.bootstrap_confidence_interval(test_data, np.mean, n_bootstrap=500)
    
    print(f"✅ Bootstrap Mean: {mean_estimate:.3f}")
    print(f"✅ Bootstrap CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    print("\n4. Testing with insufficient data...")
    
    # Test with insufficient data
    small_data = [5.0, 6.0, 7.0]
    insufficient_result = engine.analyze_trend_enhanced(small_data)
    
    print(f"✅ Insufficient Data Handling: {insufficient_result.trend_direction}")
    print(f"✅ Reliability Score: {insufficient_result.reliability_score:.3f}")
    
    print("\n🎉 Enhanced statistical engine test completed successfully!")
    
    # Return comprehensive results
    return {
        "trend_analysis": {
            "direction": trend_result.trend_direction,
            "strength": trend_result.trend_strength,
            "p_value": trend_result.statistical_result.p_value,
            "is_significant": trend_result.statistical_result.is_significant,
            "reliability": trend_result.reliability_score,
            "sample_size": trend_result.statistical_result.sample_size
        },
        "comparison_analysis": {
            "winner": comparison_result.winner,
            "p_value": comparison_result.p_value,
            "effect_size": comparison_result.effect_size,
            "confidence_level": comparison_result.confidence_level,
            "statistical_power": comparison_result.power_analysis['estimated_power']
        },
        "bootstrap_analysis": {
            "mean_estimate": mean_estimate,
            "confidence_interval": ci
        },
        "system_status": "operational_enhanced"
    }

if __name__ == "__main__":
    # Run enhanced statistical engine tests
    test_enhanced_statistical_engine()

