"""
Enhanced Data Validation Framework
Implements comprehensive validation, fact-checking, and quality assurance
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from urllib.parse import urlparse
import asyncio
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of validation process"""
    is_valid: bool
    confidence_score: float
    validation_issues: List[str] = field(default_factory=list)
    validation_details: Dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class SourceDiversityReport:
    """Report on source diversity"""
    total_sources: int
    unique_domains: int
    source_types: Dict[str, int]
    domain_distribution: Dict[str, int]
    diversity_score: float
    meets_requirements: bool
    recommendations: List[str] = field(default_factory=list)

@dataclass
class FactCheckResult:
    """Result of fact-checking process"""
    claim: str
    is_supported: bool
    confidence_level: str
    supporting_sources: List[str]
    contradicting_sources: List[str]
    verification_notes: str

class SourceDiversityValidator:
    """Validates source diversity requirements"""
    
    def __init__(self):
        self.min_sources = 5
        self.max_sources = 8
        self.min_source_types = 3
        self.max_sources_per_domain = 2
        self.min_unique_domains = 4
        
        # Source type categories
        self.source_type_weights = {
            'academic': 1.5,    # Higher weight for academic sources
            'government': 1.4,  # Government sources
            'news': 1.2,        # News sources
            'industry': 1.1,    # Industry reports
            'social': 0.8,      # Social media
            'trends': 0.9,      # Trend data
            'financial': 1.3,   # Financial data
            'research': 1.4     # Research reports
        }
    
    def validate_source_diversity(self, sources: List[Dict]) -> SourceDiversityReport:
        """Validate source diversity requirements"""
        total_sources = len(sources)
        
        # Extract domains and source types
        domains = []
        source_types = []
        
        for source in sources:
            if 'url' in source:
                domain = urlparse(source['url']).netloc.lower()
                domains.append(domain)
            
            if 'source_type' in source:
                source_types.append(source['source_type'])
        
        # Count unique domains and types
        unique_domains = len(set(domains))
        domain_distribution = dict(Counter(domains))
        source_type_distribution = dict(Counter(source_types))
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(
            total_sources, unique_domains, source_type_distribution, domain_distribution
        )
        
        # Check requirements
        meets_requirements = self._check_diversity_requirements(
            total_sources, unique_domains, source_type_distribution, domain_distribution
        )
        
        # Generate recommendations
        recommendations = self._generate_diversity_recommendations(
            total_sources, unique_domains, source_type_distribution, domain_distribution
        )
        
        return SourceDiversityReport(
            total_sources=total_sources,
            unique_domains=unique_domains,
            source_types=source_type_distribution,
            domain_distribution=domain_distribution,
            diversity_score=diversity_score,
            meets_requirements=meets_requirements,
            recommendations=recommendations
        )
    
    def _calculate_diversity_score(self, total_sources: int, unique_domains: int, 
                                 source_types: Dict[str, int], domains: Dict[str, int]) -> float:
        """Calculate overall diversity score (0-10)"""
        
        # Source count score (optimal range: 5-8)
        if self.min_sources <= total_sources <= self.max_sources:
            count_score = 10.0
        elif total_sources < self.min_sources:
            count_score = (total_sources / self.min_sources) * 10
        else:
            count_score = max(10 - (total_sources - self.max_sources), 5)
        
        # Domain diversity score
        domain_ratio = unique_domains / max(total_sources, 1)
        domain_score = min(domain_ratio * 10, 10)
        
        # Source type diversity score
        type_count = len(source_types)
        if type_count >= self.min_source_types:
            type_score = min(type_count * 2.5, 10)
        else:
            type_score = (type_count / self.min_source_types) * 10
        
        # Domain concentration penalty
        max_domain_count = max(domains.values()) if domains else 0
        if max_domain_count > self.max_sources_per_domain:
            concentration_penalty = (max_domain_count - self.max_sources_per_domain) * 2
        else:
            concentration_penalty = 0
        
        # Weighted average
        diversity_score = (count_score * 0.3 + domain_score * 0.3 + type_score * 0.4) - concentration_penalty
        
        return round(max(min(diversity_score, 10.0), 0.0), 1)
    
    def _check_diversity_requirements(self, total_sources: int, unique_domains: int,
                                    source_types: Dict[str, int], domains: Dict[str, int]) -> bool:
        """Check if diversity requirements are met"""
        
        # Check source count
        if not (self.min_sources <= total_sources <= self.max_sources):
            return False
        
        # Check unique domains
        if unique_domains < self.min_unique_domains:
            return False
        
        # Check source type diversity
        if len(source_types) < self.min_source_types:
            return False
        
        # Check domain concentration
        max_domain_count = max(domains.values()) if domains else 0
        if max_domain_count > self.max_sources_per_domain:
            return False
        
        return True
    
    def _generate_diversity_recommendations(self, total_sources: int, unique_domains: int,
                                          source_types: Dict[str, int], domains: Dict[str, int]) -> List[str]:
        """Generate recommendations for improving diversity"""
        recommendations = []
        
        # Source count recommendations
        if total_sources < self.min_sources:
            recommendations.append(f"Add {self.min_sources - total_sources} more sources to meet minimum requirement")
        elif total_sources > self.max_sources:
            recommendations.append(f"Remove {total_sources - self.max_sources} sources to stay within optimal range")
        
        # Domain diversity recommendations
        if unique_domains < self.min_unique_domains:
            recommendations.append(f"Add sources from {self.min_unique_domains - unique_domains} more unique domains")
        
        # Source type recommendations
        if len(source_types) < self.min_source_types:
            missing_types = self.min_source_types - len(source_types)
            recommendations.append(f"Add {missing_types} more source types for better diversity")
        
        # Domain concentration recommendations
        over_concentrated = [domain for domain, count in domains.items() if count > self.max_sources_per_domain]
        if over_concentrated:
            recommendations.append(f"Reduce sources from over-represented domains: {', '.join(over_concentrated)}")
        
        # Source type balance recommendations
        if source_types:
            max_type_count = max(source_types.values())
            if max_type_count > total_sources * 0.6:  # More than 60% from one type
                dominant_type = max(source_types, key=source_types.get)
                recommendations.append(f"Reduce reliance on '{dominant_type}' sources for better balance")
        
        return recommendations

class CrossReferenceValidator:
    """Validates information across multiple sources"""
    
    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
    
    def cross_validate_claims(self, sources: List[Dict], claims: List[str]) -> List[FactCheckResult]:
        """Cross-validate claims across multiple sources"""
        fact_check_results = []
        
        for claim in claims:
            result = self._validate_single_claim(claim, sources)
            fact_check_results.append(result)
        
        return fact_check_results
    
    def _validate_single_claim(self, claim: str, sources: List[Dict]) -> FactCheckResult:
        """Validate a single claim against sources"""
        supporting_sources = []
        contradicting_sources = []
        
        # Extract keywords from claim
        claim_keywords = self._extract_keywords(claim)
        
        # Check each source for support/contradiction
        for source in sources:
            source_content = source.get('content', '') + ' ' + source.get('title', '')
            source_url = source.get('url', 'Unknown')
            
            support_score = self._calculate_support_score(claim_keywords, source_content)
            
            if support_score > 0.6:
                supporting_sources.append(source_url)
            elif support_score < -0.3:
                contradicting_sources.append(source_url)
        
        # Determine overall support
        total_sources = len(supporting_sources) + len(contradicting_sources)
        if total_sources == 0:
            is_supported = False
            confidence_level = 'unknown'
        else:
            support_ratio = len(supporting_sources) / total_sources
            is_supported = support_ratio > 0.5
            
            if support_ratio >= self.confidence_thresholds['high']:
                confidence_level = 'high'
            elif support_ratio >= self.confidence_thresholds['medium']:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
        
        verification_notes = self._generate_verification_notes(
            claim, supporting_sources, contradicting_sources, confidence_level
        )
        
        return FactCheckResult(
            claim=claim,
            is_supported=is_supported,
            confidence_level=confidence_level,
            supporting_sources=supporting_sources,
            contradicting_sources=contradicting_sources,
            verification_notes=verification_notes
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return keywords[:10]  # Return top 10 keywords
    
    def _calculate_support_score(self, claim_keywords: List[str], source_content: str) -> float:
        """Calculate how much a source supports a claim"""
        source_content_lower = source_content.lower()
        
        # Count keyword matches
        matches = sum(1 for keyword in claim_keywords if keyword in source_content_lower)
        
        if not claim_keywords:
            return 0.0
        
        # Calculate support score
        support_score = matches / len(claim_keywords)
        
        # Look for explicit contradiction indicators
        contradiction_indicators = [
            'not', 'no', 'false', 'incorrect', 'wrong', 'dispute', 'deny', 'refute'
        ]
        
        contradiction_count = sum(1 for indicator in contradiction_indicators 
                                if indicator in source_content_lower)
        
        # Adjust score for contradictions
        if contradiction_count > 0:
            support_score -= contradiction_count * 0.2
        
        return max(min(support_score, 1.0), -1.0)
    
    def _generate_verification_notes(self, claim: str, supporting: List[str], 
                                   contradicting: List[str], confidence: str) -> str:
        """Generate verification notes"""
        notes = f"Claim: '{claim}'\n"
        notes += f"Confidence Level: {confidence.title()}\n"
        notes += f"Supporting Sources: {len(supporting)}\n"
        notes += f"Contradicting Sources: {len(contradicting)}\n"
        
        if supporting:
            notes += f"Support found in: {', '.join(supporting[:3])}{'...' if len(supporting) > 3 else ''}\n"
        
        if contradicting:
            notes += f"Contradictions found in: {', '.join(contradicting[:3])}{'...' if len(contradicting) > 3 else ''}\n"
        
        return notes

class QualityAssuranceEngine:
    """Comprehensive quality assurance for research data"""
    
    def __init__(self):
        self.diversity_validator = SourceDiversityValidator()
        self.cross_reference_validator = CrossReferenceValidator()
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 9.0,
            'good': 7.0,
            'acceptable': 5.0,
            'poor': 3.0
        }
    
    def validate_research_quality(self, research_data: Dict) -> ValidationResult:
        """Comprehensive quality validation of research data"""
        validation_issues = []
        validation_details = {}
        recommendations = []
        
        # Extract sources and claims
        sources = research_data.get('sources', [])
        claims = self._extract_claims_from_research(research_data)
        
        # 1. Source diversity validation
        diversity_report = self.diversity_validator.validate_source_diversity(sources)
        validation_details['source_diversity'] = diversity_report
        
        if not diversity_report.meets_requirements:
            validation_issues.append("Source diversity requirements not met")
            recommendations.extend(diversity_report.recommendations)
        
        # 2. Cross-reference validation
        if claims:
            fact_check_results = self.cross_reference_validator.cross_validate_claims(sources, claims)
            validation_details['fact_checking'] = fact_check_results
            
            # Check for unsupported claims
            unsupported_claims = [result for result in fact_check_results if not result.is_supported]
            if unsupported_claims:
                validation_issues.append(f"{len(unsupported_claims)} claims lack sufficient support")
                recommendations.append("Verify unsupported claims with additional sources")
        
        # 3. Source credibility validation
        credibility_issues = self._validate_source_credibility(sources)
        if credibility_issues:
            validation_issues.extend(credibility_issues)
            recommendations.append("Improve source credibility by adding more authoritative sources")
        
        # 4. Data freshness validation
        freshness_issues = self._validate_data_freshness(sources)
        if freshness_issues:
            validation_issues.extend(freshness_issues)
            recommendations.append("Update research with more recent sources")
        
        # 5. Content completeness validation
        completeness_issues = self._validate_content_completeness(research_data)
        if completeness_issues:
            validation_issues.extend(completeness_issues)
            recommendations.append("Complete missing research sections")
        
        # Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(
            diversity_report, validation_issues, sources
        )
        
        # Determine if validation passes
        is_valid = confidence_score >= self.quality_thresholds['acceptable'] and len(validation_issues) <= 3
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            validation_issues=validation_issues,
            validation_details=validation_details,
            recommendations=recommendations
        )
    
    def _extract_claims_from_research(self, research_data: Dict) -> List[str]:
        """Extract verifiable claims from research data"""
        claims = []
        
        # Extract from market analysis
        market_analysis = research_data.get('market_analysis', {})
        if 'key_insights' in market_analysis:
            claims.extend(market_analysis['key_insights'])
        
        # Extract from financial insights
        financial_insights = research_data.get('financial_insights', {})
        if 'market_size_estimate' in financial_insights:
            claims.append(f"Market size: {financial_insights['market_size_estimate']}")
        
        # Extract from competitive analysis
        competitive_analysis = research_data.get('competitive_analysis', {})
        if 'key_competitors' in competitive_analysis:
            for competitor in competitive_analysis['key_competitors']:
                claims.append(f"Key competitor: {competitor}")
        
        return claims[:10]  # Limit to top 10 claims
    
    def _validate_source_credibility(self, sources: List[Dict]) -> List[str]:
        """Validate source credibility"""
        issues = []
        
        if not sources:
            issues.append("No sources provided")
            return issues
        
        # Check average credibility score
        credibility_scores = [source.get('credibility_score', 0) for source in sources]
        avg_credibility = sum(credibility_scores) / len(credibility_scores)
        
        if avg_credibility < 6.0:
            issues.append(f"Low average source credibility: {avg_credibility:.1f}/10")
        
        # Check for sources with very low credibility
        low_credibility_count = sum(1 for score in credibility_scores if score < 5.0)
        if low_credibility_count > len(sources) * 0.3:  # More than 30%
            issues.append(f"{low_credibility_count} sources have low credibility scores")
        
        return issues
    
    def _validate_data_freshness(self, sources: List[Dict]) -> List[str]:
        """Validate data freshness"""
        issues = []
        
        current_time = datetime.now()
        old_sources = 0
        
        for source in sources:
            pub_date = source.get('publication_date')
            if pub_date:
                if isinstance(pub_date, str):
                    try:
                        pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    except:
                        continue
                
                age_days = (current_time - pub_date).days
                if age_days > 365:  # Older than 1 year
                    old_sources += 1
        
        if old_sources > len(sources) * 0.5:  # More than 50% old
            issues.append(f"{old_sources} sources are older than 1 year")
        
        return issues
    
    def _validate_content_completeness(self, research_data: Dict) -> List[str]:
        """Validate content completeness"""
        issues = []
        
        required_sections = [
            'market_analysis', 'competitive_analysis', 
            'financial_insights', 'risk_assessment'
        ]
        
        for section in required_sections:
            if section not in research_data or not research_data[section]:
                issues.append(f"Missing or empty section: {section}")
        
        return issues
    
    def _calculate_confidence_score(self, diversity_report: SourceDiversityReport, 
                                  validation_issues: List[str], sources: List[Dict]) -> float:
        """Calculate overall confidence score"""
        
        # Base score from source diversity
        diversity_score = diversity_report.diversity_score
        
        # Penalty for validation issues
        issue_penalty = len(validation_issues) * 0.5
        
        # Bonus for source quality
        if sources:
            credibility_scores = [source.get('credibility_score', 0) for source in sources]
            avg_credibility = sum(credibility_scores) / len(credibility_scores)
            credibility_bonus = (avg_credibility - 5.0) * 0.5  # Bonus above baseline of 5.0
        else:
            credibility_bonus = 0
        
        # Calculate final score
        confidence_score = diversity_score - issue_penalty + credibility_bonus
        
        return round(max(min(confidence_score, 10.0), 0.0), 1)

# Factory function to create validation services
def create_validation_services() -> QualityAssuranceEngine:
    """Create validation services"""
    return QualityAssuranceEngine()

# Test function for validation system
async def test_validation_system():
    """Test the validation system"""
    print("Testing Validation System...")
    
    # Create validation engine
    qa_engine = create_validation_services()
    
    # Create test research data
    test_sources = [
        {
            'url': 'https://techcrunch.com/article1',
            'title': 'AI Market Growth Analysis',
            'content': 'The AI market is experiencing significant growth with increasing investment.',
            'source_type': 'news',
            'credibility_score': 8.0,
            'publication_date': datetime.now() - timedelta(days=30)
        },
        {
            'url': 'https://academic.edu/research',
            'title': 'Academic Study on AI Adoption',
            'content': 'Research shows widespread AI adoption across industries.',
            'source_type': 'academic',
            'credibility_score': 9.0,
            'publication_date': datetime.now() - timedelta(days=60)
        },
        {
            'url': 'https://reddit.com/discussion',
            'title': 'Community Discussion on AI',
            'content': 'Users discuss AI implementation challenges and opportunities.',
            'source_type': 'social',
            'credibility_score': 6.0,
            'publication_date': datetime.now() - timedelta(days=10)
        }
    ]
    
    test_research_data = {
        'sources': test_sources,
        'market_analysis': {
            'key_insights': ['Growing AI market', 'Increasing investment']
        },
        'competitive_analysis': {
            'key_competitors': ['OpenAI', 'Google AI']
        },
        'financial_insights': {
            'market_size_estimate': '$100B by 2025'
        },
        'risk_assessment': {
            'overall_risk_level': 'Medium'
        }
    }
    
    # Test source diversity validation
    print("\n1. Testing source diversity validation...")
    diversity_report = qa_engine.diversity_validator.validate_source_diversity(test_sources)
    print(f"✅ Diversity score: {diversity_report.diversity_score}/10")
    print(f"✅ Meets requirements: {diversity_report.meets_requirements}")
    
    # Test cross-reference validation
    print("\n2. Testing cross-reference validation...")
    claims = ['AI market is growing', 'Investment is increasing']
    fact_check_results = qa_engine.cross_reference_validator.cross_validate_claims(test_sources, claims)
    print(f"✅ Fact-checked {len(fact_check_results)} claims")
    
    # Test comprehensive quality validation
    print("\n3. Testing comprehensive quality validation...")
    validation_result = qa_engine.validate_research_quality(test_research_data)
    print(f"✅ Overall validation: {validation_result.is_valid}")
    print(f"✅ Confidence score: {validation_result.confidence_score}/10")
    print(f"✅ Issues found: {len(validation_result.validation_issues)}")
    
    print("\n🎉 Validation system test completed successfully!")
    
    return {
        'diversity_report': diversity_report,
        'fact_check_results': fact_check_results,
        'validation_result': validation_result
    }

if __name__ == "__main__":
    # Run validation tests
    asyncio.run(test_validation_system())

