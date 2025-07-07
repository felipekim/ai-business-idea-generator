"""
Smart Quality Assurance Engine
Automated validation pipeline with intelligent quality gates and filtering
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"      # 9.0-10.0
    VERY_GOOD = "very_good"     # 8.0-8.9
    GOOD = "good"               # 7.0-7.9
    ACCEPTABLE = "acceptable"   # 6.0-6.9
    POOR = "poor"               # 4.0-5.9
    UNACCEPTABLE = "unacceptable"  # 0.0-3.9

class ValidationResult(Enum):
    """Validation results"""
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"
    CONDITIONAL_APPROVAL = "conditional_approval"

class QualityGate(Enum):
    """Quality gate checkpoints"""
    IDEA_COHERENCE = "idea_coherence"
    MARKET_VALIDATION = "market_validation"
    FINANCIAL_VIABILITY = "financial_viability"
    SOURCE_QUALITY = "source_quality"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    IMPLEMENTATION_FEASIBILITY = "implementation_feasibility"

@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    overall_score: float
    quality_level: QualityLevel
    validation_result: ValidationResult
    gate_scores: Dict[QualityGate, float] = field(default_factory=dict)
    quality_issues: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    confidence_level: float = 0.0
    assessment_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'overall_score': self.overall_score,
            'quality_level': self.quality_level.value,
            'validation_result': self.validation_result.value,
            'gate_scores': {gate.value: score for gate, score in self.gate_scores.items()},
            'quality_issues': self.quality_issues,
            'improvement_suggestions': self.improvement_suggestions,
            'confidence_level': self.confidence_level,
            'assessment_timestamp': self.assessment_timestamp.isoformat()
        }

@dataclass
class QualityAssuranceConfig:
    """Configuration for quality assurance"""
    # Score thresholds
    minimum_overall_score: float = 7.0
    minimum_gate_score: float = 6.0
    confidence_threshold: float = 0.8
    
    # Quality gates weights
    gate_weights: Dict[QualityGate, float] = field(default_factory=lambda: {
        QualityGate.IDEA_COHERENCE: 0.20,
        QualityGate.MARKET_VALIDATION: 0.25,
        QualityGate.FINANCIAL_VIABILITY: 0.20,
        QualityGate.SOURCE_QUALITY: 0.15,
        QualityGate.COMPETITIVE_ANALYSIS: 0.10,
        QualityGate.IMPLEMENTATION_FEASIBILITY: 0.10
    })
    
    # Validation criteria
    min_sources_required: int = 5
    max_sources_allowed: int = 8
    min_market_size: float = 1000000  # $1M minimum market size
    max_competition_level: float = 8.0  # Max competition score
    min_implementation_score: float = 6.0
    
    # Content quality criteria
    min_description_length: int = 100
    required_keywords: List[str] = field(default_factory=lambda: [
        "market", "customer", "solution", "revenue", "business"
    ])
    
    # Automated rejection criteria
    auto_reject_keywords: List[str] = field(default_factory=lambda: [
        "illegal", "unethical", "scam", "pyramid", "ponzi"
    ])

class SmartQualityAssuranceEngine:
    """Smart quality assurance engine for automated validation"""
    
    def __init__(self, config: Optional[QualityAssuranceConfig] = None):
        self.config = config or QualityAssuranceConfig()
        self.quality_history: List[QualityMetrics] = []
        self.performance_stats = {
            'total_assessments': 0,
            'approved_count': 0,
            'rejected_count': 0,
            'average_score': 0.0,
            'quality_trends': []
        }
        
        logger.info("Smart Quality Assurance Engine initialized")
    
    def assess_idea_quality(self, idea_data: Dict[str, Any]) -> QualityMetrics:
        """Comprehensive quality assessment of a business idea"""
        logger.info(f"Assessing quality for idea: {idea_data.get('name', 'Unknown')}")
        
        try:
            # Run all quality gates
            gate_scores = {}
            quality_issues = []
            improvement_suggestions = []
            
            # Gate 1: Idea Coherence
            coherence_score, coherence_issues, coherence_suggestions = self._assess_idea_coherence(idea_data)
            gate_scores[QualityGate.IDEA_COHERENCE] = coherence_score
            quality_issues.extend(coherence_issues)
            improvement_suggestions.extend(coherence_suggestions)
            
            # Gate 2: Market Validation
            market_score, market_issues, market_suggestions = self._assess_market_validation(idea_data)
            gate_scores[QualityGate.MARKET_VALIDATION] = market_score
            quality_issues.extend(market_issues)
            improvement_suggestions.extend(market_suggestions)
            
            # Gate 3: Financial Viability
            financial_score, financial_issues, financial_suggestions = self._assess_financial_viability(idea_data)
            gate_scores[QualityGate.FINANCIAL_VIABILITY] = financial_score
            quality_issues.extend(financial_issues)
            improvement_suggestions.extend(financial_suggestions)
            
            # Gate 4: Source Quality
            source_score, source_issues, source_suggestions = self._assess_source_quality(idea_data)
            gate_scores[QualityGate.SOURCE_QUALITY] = source_score
            quality_issues.extend(source_issues)
            improvement_suggestions.extend(source_suggestions)
            
            # Gate 5: Competitive Analysis
            competitive_score, competitive_issues, competitive_suggestions = self._assess_competitive_analysis(idea_data)
            gate_scores[QualityGate.COMPETITIVE_ANALYSIS] = competitive_score
            quality_issues.extend(competitive_issues)
            improvement_suggestions.extend(competitive_suggestions)
            
            # Gate 6: Implementation Feasibility
            implementation_score, impl_issues, impl_suggestions = self._assess_implementation_feasibility(idea_data)
            gate_scores[QualityGate.IMPLEMENTATION_FEASIBILITY] = implementation_score
            quality_issues.extend(impl_issues)
            improvement_suggestions.extend(impl_suggestions)
            
            # Calculate overall score
            overall_score = self._calculate_weighted_score(gate_scores)
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Determine validation result
            validation_result = self._determine_validation_result(
                overall_score, gate_scores, quality_issues
            )
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(gate_scores, idea_data)
            
            # Create quality metrics
            quality_metrics = QualityMetrics(
                overall_score=overall_score,
                quality_level=quality_level,
                validation_result=validation_result,
                gate_scores=gate_scores,
                quality_issues=quality_issues,
                improvement_suggestions=improvement_suggestions,
                confidence_level=confidence_level
            )
            
            # Store in history
            self.quality_history.append(quality_metrics)
            self._update_performance_stats(quality_metrics)
            
            logger.info(f"Quality assessment completed: {overall_score:.2f} ({quality_level.value})")
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            # Return default poor quality metrics
            return QualityMetrics(
                overall_score=3.0,
                quality_level=QualityLevel.UNACCEPTABLE,
                validation_result=ValidationResult.REJECTED,
                quality_issues=[f"Assessment error: {str(e)}"],
                improvement_suggestions=["Manual review required due to assessment error"]
            )
    
    def _assess_idea_coherence(self, idea_data: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Assess idea coherence and clarity"""
        score = 8.0  # Start with good score
        issues = []
        suggestions = []
        
        # Check required fields
        required_fields = ['name', 'tagline', 'problem_statement', 'ai_solution']
        missing_fields = [field for field in required_fields if not idea_data.get(field)]
        
        if missing_fields:
            score -= len(missing_fields) * 1.5
            issues.append(f"Missing required fields: {', '.join(missing_fields)}")
            suggestions.append("Ensure all core idea components are defined")
        
        # Check description quality
        description = idea_data.get('problem_statement', '') + ' ' + idea_data.get('ai_solution', '')
        if len(description) < self.config.min_description_length:
            score -= 2.0
            issues.append("Insufficient detail in problem statement and solution")
            suggestions.append("Provide more detailed problem and solution descriptions")
        
        # Check for required keywords
        description_lower = description.lower()
        missing_keywords = [
            kw for kw in self.config.required_keywords 
            if kw not in description_lower
        ]
        
        if len(missing_keywords) > 2:
            score -= 1.0
            issues.append("Missing key business concepts in description")
            suggestions.append("Include more business-focused terminology")
        
        # Check for auto-reject keywords
        for reject_keyword in self.config.auto_reject_keywords:
            if reject_keyword in description_lower:
                score = 0.0
                issues.append(f"Contains prohibited content: {reject_keyword}")
                suggestions.append("Remove inappropriate or unethical content")
                break
        
        return max(0.0, min(10.0, score)), issues, suggestions
    
    def _assess_market_validation(self, idea_data: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Assess market validation quality"""
        score = 7.0
        issues = []
        suggestions = []
        
        market_research = idea_data.get('market_research', {})
        
        if not market_research:
            score = 4.0
            issues.append("No market research data available")
            suggestions.append("Conduct comprehensive market research")
            return score, issues, suggestions
        
        # Check market score
        market_score_data = market_research.get('market_score', {})
        total_market_score = market_score_data.get('total_score', 5.0)
        
        if total_market_score < 6.0:
            score -= 2.0
            issues.append("Low market validation score")
            suggestions.append("Strengthen market opportunity validation")
        
        # Check market size
        market_size = market_research.get('market_size', {}).get('total_addressable_market', 0)
        if market_size < self.config.min_market_size:
            score -= 1.5
            issues.append("Market size below minimum threshold")
            suggestions.append("Target larger market opportunities")
        
        # Check demand signals
        demand_signals = market_research.get('demand_signals', {})
        if not demand_signals or len(demand_signals) < 3:
            score -= 1.0
            issues.append("Insufficient demand validation")
            suggestions.append("Gather more demand indicators")
        
        return max(0.0, min(10.0, score)), issues, suggestions
    
    def _assess_financial_viability(self, idea_data: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Assess financial viability"""
        score = 7.0
        issues = []
        suggestions = []
        
        financial_analysis = idea_data.get('financial_analysis', {})
        
        if not financial_analysis:
            score = 4.0
            issues.append("No financial analysis available")
            suggestions.append("Conduct detailed financial analysis")
            return score, issues, suggestions
        
        # Check cost breakdown
        cost_breakdown = financial_analysis.get('cost_breakdown', {})
        launch_costs = cost_breakdown.get('launch_costs', {}).get('total', 0)
        
        if launch_costs > 100000:  # $100K threshold
            score -= 1.0
            issues.append("High launch costs may limit accessibility")
            suggestions.append("Consider ways to reduce initial investment")
        
        # Check revenue projections
        revenue_projections = financial_analysis.get('revenue_projections', {})
        year_1_revenue = revenue_projections.get('revenue_by_year', {}).get('year_1', {}).get('annual_revenue', 0)
        
        if year_1_revenue < launch_costs:
            score -= 1.5
            issues.append("Revenue projections don't cover launch costs in year 1")
            suggestions.append("Improve revenue model or reduce costs")
        
        # Check ROI
        financial_metrics = financial_analysis.get('financial_metrics', {})
        roi_percentage = financial_metrics.get('roi_percentage', 0)
        
        if roi_percentage < 100:  # 100% ROI minimum
            score -= 1.0
            issues.append("Low return on investment")
            suggestions.append("Optimize revenue model for better ROI")
        
        # Check break-even point
        break_even_months = financial_metrics.get('break_even_point_months', 24)
        if break_even_months > 18:  # 18 months maximum
            score -= 1.0
            issues.append("Long break-even period")
            suggestions.append("Accelerate path to profitability")
        
        return max(0.0, min(10.0, score)), issues, suggestions
    
    def _assess_source_quality(self, idea_data: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Assess quality of research sources"""
        score = 8.0
        issues = []
        suggestions = []
        
        # Check if validation evidence exists
        validation_evidence = idea_data.get('validation_evidence', {})
        if not validation_evidence:
            score = 5.0
            issues.append("No validation evidence provided")
            suggestions.append("Include research sources and validation data")
            return score, issues, suggestions
        
        # Check number of sources
        sources = validation_evidence.get('sources', [])
        source_count = len(sources)
        
        if source_count < self.config.min_sources_required:
            score -= 2.0
            issues.append(f"Insufficient sources ({source_count} < {self.config.min_sources_required})")
            suggestions.append("Include more research sources")
        elif source_count > self.config.max_sources_allowed:
            score -= 0.5
            issues.append("Too many sources may indicate unfocused research")
            suggestions.append("Focus on highest quality sources")
        
        # Check source diversity
        source_types = set()
        for source in sources:
            source_type = source.get('type', 'unknown')
            source_types.add(source_type)
        
        if len(source_types) < 3:
            score -= 1.0
            issues.append("Limited source diversity")
            suggestions.append("Include diverse source types (news, research, industry reports)")
        
        return max(0.0, min(10.0, score)), issues, suggestions
    
    def _assess_competitive_analysis(self, idea_data: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Assess competitive analysis quality"""
        score = 7.0
        issues = []
        suggestions = []
        
        # Check if competitive analysis exists
        market_research = idea_data.get('market_research', {})
        competitive_analysis = market_research.get('competitive_analysis', {})
        
        if not competitive_analysis:
            score = 5.0
            issues.append("No competitive analysis provided")
            suggestions.append("Conduct thorough competitive analysis")
            return score, issues, suggestions
        
        # Check competition level
        competition_level = competitive_analysis.get('competition_level', 5.0)
        if competition_level > self.config.max_competition_level:
            score -= 1.5
            issues.append("High competition level")
            suggestions.append("Consider differentiation strategies or niche markets")
        
        # Check competitor count
        competitors = competitive_analysis.get('main_competitors', [])
        if len(competitors) < 2:
            score -= 1.0
            issues.append("Insufficient competitor analysis")
            suggestions.append("Identify and analyze more competitors")
        
        return max(0.0, min(10.0, score)), issues, suggestions
    
    def _assess_implementation_feasibility(self, idea_data: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Assess implementation feasibility"""
        score = 7.0
        issues = []
        suggestions = []
        
        # Check implementation plan
        implementation_plan = idea_data.get('implementation_plan', {})
        if not implementation_plan:
            score = 5.0
            issues.append("No implementation plan provided")
            suggestions.append("Develop detailed implementation roadmap")
            return score, issues, suggestions
        
        # Check if plan has phases
        phases = [key for key in implementation_plan.keys() if 'phase' in key.lower()]
        if len(phases) < 2:
            score -= 1.0
            issues.append("Implementation plan lacks structured phases")
            suggestions.append("Break implementation into clear phases")
        
        # Check for technical requirements
        technical_requirements = implementation_plan.get('technical_requirements', '')
        if not technical_requirements:
            score -= 0.5
            issues.append("Missing technical requirements")
            suggestions.append("Define technical implementation requirements")
        
        return max(0.0, min(10.0, score)), issues, suggestions
    
    def _calculate_weighted_score(self, gate_scores: Dict[QualityGate, float]) -> float:
        """Calculate weighted overall score"""
        total_score = 0.0
        total_weight = 0.0
        
        for gate, score in gate_scores.items():
            weight = self.config.gate_weights.get(gate, 0.1)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine quality level based on score"""
        if overall_score >= 9.0:
            return QualityLevel.EXCELLENT
        elif overall_score >= 8.0:
            return QualityLevel.VERY_GOOD
        elif overall_score >= 7.0:
            return QualityLevel.GOOD
        elif overall_score >= 6.0:
            return QualityLevel.ACCEPTABLE
        elif overall_score >= 4.0:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE
    
    def _determine_validation_result(self, overall_score: float, gate_scores: Dict[QualityGate, float], 
                                   quality_issues: List[str]) -> ValidationResult:
        """Determine validation result"""
        # Auto-reject for critical issues
        critical_keywords = ["prohibited", "illegal", "unethical"]
        for issue in quality_issues:
            if any(keyword in issue.lower() for keyword in critical_keywords):
                return ValidationResult.REJECTED
        
        # Check minimum thresholds
        if overall_score < self.config.minimum_overall_score:
            return ValidationResult.REJECTED
        
        # Check individual gate scores
        failed_gates = [
            gate for gate, score in gate_scores.items()
            if score < self.config.minimum_gate_score
        ]
        
        if len(failed_gates) > 2:
            return ValidationResult.REJECTED
        elif len(failed_gates) > 0:
            return ValidationResult.CONDITIONAL_APPROVAL
        elif overall_score >= 8.5:
            return ValidationResult.APPROVED
        else:
            return ValidationResult.NEEDS_REVIEW
    
    def _calculate_confidence_level(self, gate_scores: Dict[QualityGate, float], 
                                  idea_data: Dict[str, Any]) -> float:
        """Calculate confidence level in the assessment"""
        confidence = 0.8  # Base confidence
        
        # Adjust based on data completeness
        required_sections = ['market_research', 'financial_analysis', 'validation_evidence']
        missing_sections = [section for section in required_sections if not idea_data.get(section)]
        
        confidence -= len(missing_sections) * 0.1
        
        # Adjust based on score consistency
        scores = list(gate_scores.values())
        if scores:
            score_std = statistics.stdev(scores) if len(scores) > 1 else 0
            if score_std > 2.0:  # High variance indicates inconsistency
                confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _update_performance_stats(self, quality_metrics: QualityMetrics):
        """Update performance statistics"""
        self.performance_stats['total_assessments'] += 1
        
        if quality_metrics.validation_result == ValidationResult.APPROVED:
            self.performance_stats['approved_count'] += 1
        elif quality_metrics.validation_result == ValidationResult.REJECTED:
            self.performance_stats['rejected_count'] += 1
        
        # Update average score
        total_score = sum(qm.overall_score for qm in self.quality_history)
        self.performance_stats['average_score'] = total_score / len(self.quality_history)
        
        # Track quality trends (last 10 assessments)
        recent_scores = [qm.overall_score for qm in self.quality_history[-10:]]
        self.performance_stats['quality_trends'] = recent_scores
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get quality assurance performance report"""
        total = self.performance_stats['total_assessments']
        approved = self.performance_stats['approved_count']
        rejected = self.performance_stats['rejected_count']
        
        approval_rate = (approved / total * 100) if total > 0 else 0
        rejection_rate = (rejected / total * 100) if total > 0 else 0
        
        return {
            'total_assessments': total,
            'approved_count': approved,
            'rejected_count': rejected,
            'approval_rate': approval_rate,
            'rejection_rate': rejection_rate,
            'average_score': self.performance_stats['average_score'],
            'quality_trends': self.performance_stats['quality_trends'],
            'recent_quality_levels': [
                qm.quality_level.value for qm in self.quality_history[-10:]
            ],
            'common_issues': self._get_common_issues(),
            'improvement_areas': self._get_improvement_areas()
        }
    
    def _get_common_issues(self) -> List[str]:
        """Get most common quality issues"""
        issue_counts = defaultdict(int)
        
        for qm in self.quality_history[-20:]:  # Last 20 assessments
            for issue in qm.quality_issues:
                issue_counts[issue] += 1
        
        # Return top 5 most common issues
        return [issue for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    def _get_improvement_areas(self) -> List[str]:
        """Get areas needing improvement based on gate scores"""
        gate_averages = defaultdict(list)
        
        for qm in self.quality_history[-10:]:  # Last 10 assessments
            for gate, score in qm.gate_scores.items():
                gate_averages[gate].append(score)
        
        # Calculate average scores for each gate
        weak_gates = []
        for gate, scores in gate_averages.items():
            avg_score = statistics.mean(scores) if scores else 0
            if avg_score < 7.0:
                weak_gates.append(gate.value)
        
        return weak_gates
    
    def batch_assess_ideas(self, ideas: List[Dict[str, Any]]) -> List[QualityMetrics]:
        """Assess quality for a batch of ideas"""
        logger.info(f"Starting batch quality assessment for {len(ideas)} ideas")
        
        results = []
        for i, idea in enumerate(ideas, 1):
            logger.info(f"Assessing idea {i}/{len(ideas)}")
            quality_metrics = self.assess_idea_quality(idea)
            results.append(quality_metrics)
        
        # Generate batch summary
        approved_count = sum(1 for qm in results if qm.validation_result == ValidationResult.APPROVED)
        avg_score = statistics.mean([qm.overall_score for qm in results]) if results else 0
        
        logger.info(f"Batch assessment complete: {approved_count}/{len(ideas)} approved, avg score: {avg_score:.2f}")
        
        return results

