"""
Enhanced Data Validation System
Sophisticated source credibility scoring, data freshness validation, and cross-reference fact-checking
"""

import re
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from urllib.parse import urlparse
import logging
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)

class SourceType(Enum):
    """Source type classification"""
    NEWS_MEDIA = "news_media"
    ACADEMIC = "academic"
    GOVERNMENT = "government"
    INDUSTRY_REPORT = "industry_report"
    SOCIAL_MEDIA = "social_media"
    BLOG = "blog"
    FORUM = "forum"
    UNKNOWN = "unknown"

class CredibilityLevel(Enum):
    """Source credibility levels"""
    VERY_HIGH = "very_high"  # 9-10
    HIGH = "high"            # 7-8
    MEDIUM = "medium"        # 5-6
    LOW = "low"              # 3-4
    VERY_LOW = "very_low"    # 1-2

@dataclass
class SourceCredibilityScore:
    """Comprehensive source credibility assessment"""
    url: str
    domain: str
    source_type: SourceType
    credibility_level: CredibilityLevel
    raw_score: float  # 0-10
    factors: Dict[str, float]
    confidence: float
    last_updated: datetime
    
@dataclass
class DataFreshnessCheck:
    """Data freshness validation result"""
    content: str
    publish_date: Optional[datetime]
    age_days: Optional[int]
    is_fresh: bool
    freshness_score: float  # 0-1
    staleness_indicators: List[str]
    
@dataclass
class CrossReferenceResult:
    """Cross-reference fact-checking result"""
    claim: str
    supporting_sources: List[str]
    contradicting_sources: List[str]
    confidence_score: float  # 0-1
    consensus_level: str  # "strong", "moderate", "weak", "conflicted"
    verification_status: str  # "verified", "disputed", "unverified"

@dataclass
class ValidationResult:
    """Comprehensive validation result"""
    source_url: str
    content_hash: str
    credibility_score: SourceCredibilityScore
    freshness_check: DataFreshnessCheck
    cross_reference_results: List[CrossReferenceResult]
    overall_quality_score: float  # 0-1
    validation_grade: str  # A+ to F
    critical_issues: List[str]
    recommendations: List[str]
    validation_timestamp: datetime

class EnhancedDataValidator:
    """Production-grade data validation with sophisticated quality assurance"""
    
    def __init__(self):
        # Trusted domain lists
        self.high_credibility_domains = {
            # News Media
            'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'economist.com',
            'bbc.com', 'cnn.com', 'npr.org', 'pbs.org', 'apnews.com',
            
            # Academic & Research
            'nature.com', 'science.org', 'ieee.org', 'acm.org', 'arxiv.org',
            'pubmed.ncbi.nlm.nih.gov', 'scholar.google.com',
            
            # Government & Official
            'gov', 'edu', 'census.gov', 'bls.gov', 'sec.gov', 'ftc.gov',
            
            # Industry Reports
            'mckinsey.com', 'bcg.com', 'deloitte.com', 'pwc.com', 'kpmg.com',
            'gartner.com', 'forrester.com', 'idc.com', 'statista.com'
        }
        
        self.medium_credibility_domains = {
            'techcrunch.com', 'venturebeat.com', 'wired.com', 'arstechnica.com',
            'theverge.com', 'engadget.com', 'mashable.com', 'forbes.com',
            'businessinsider.com', 'inc.com', 'entrepreneur.com'
        }
        
        self.low_credibility_indicators = {
            'blog', 'wordpress', 'medium.com', 'substack.com', 'reddit.com',
            'twitter.com', 'facebook.com', 'linkedin.com', 'quora.com'
        }
        
        # Content quality indicators
        self.quality_indicators = {
            'positive': [
                'study shows', 'research indicates', 'data reveals', 'according to',
                'peer-reviewed', 'published in', 'survey of', 'analysis of',
                'statistics show', 'report finds', 'evidence suggests'
            ],
            'negative': [
                'i think', 'in my opinion', 'probably', 'maybe', 'could be',
                'rumor has it', 'some say', 'allegedly', 'unconfirmed',
                'breaking:', 'exclusive:', 'shocking'
            ]
        }
        
        # Staleness indicators
        self.staleness_keywords = [
            'last year', 'in 2020', 'in 2021', 'in 2022', 'previously',
            'former', 'outdated', 'legacy', 'deprecated', 'discontinued'
        ]
        
        logger.info("Enhanced Data Validator initialized with comprehensive quality checks")
    
    def validate_source_comprehensive(self, url: str, content: str, 
                                    metadata: Optional[Dict] = None) -> ValidationResult:
        """Comprehensive source validation with all quality checks"""
        try:
            # Generate content hash for tracking
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            # 1. Source credibility assessment
            credibility_score = self._assess_source_credibility(url, content, metadata)
            
            # 2. Data freshness validation
            freshness_check = self._validate_data_freshness(content, metadata)
            
            # 3. Cross-reference fact-checking (simplified for demo)
            cross_reference_results = self._perform_cross_reference_check(content)
            
            # 4. Calculate overall quality score
            overall_quality_score = self._calculate_overall_quality(
                credibility_score, freshness_check, cross_reference_results
            )
            
            # 5. Assign validation grade
            validation_grade = self._assign_validation_grade(overall_quality_score)
            
            # 6. Identify critical issues
            critical_issues = self._identify_critical_issues(
                credibility_score, freshness_check, cross_reference_results
            )
            
            # 7. Generate recommendations
            recommendations = self._generate_recommendations(
                credibility_score, freshness_check, critical_issues
            )
            
            result = ValidationResult(
                source_url=url,
                content_hash=content_hash,
                credibility_score=credibility_score,
                freshness_check=freshness_check,
                cross_reference_results=cross_reference_results,
                overall_quality_score=overall_quality_score,
                validation_grade=validation_grade,
                critical_issues=critical_issues,
                recommendations=recommendations,
                validation_timestamp=datetime.now()
            )
            
            logger.info(f"Comprehensive validation completed: {url} -> Grade {validation_grade} ({overall_quality_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed for {url}: {e}")
            return self._create_error_validation_result(url, str(e))
    
    def _assess_source_credibility(self, url: str, content: str, 
                                 metadata: Optional[Dict] = None) -> SourceCredibilityScore:
        """Sophisticated source credibility assessment"""
        try:
            domain = urlparse(url).netloc.lower()
            
            # Initialize scoring factors
            factors = {
                'domain_reputation': 0.0,
                'content_quality': 0.0,
                'author_credibility': 0.0,
                'publication_standards': 0.0,
                'citation_quality': 0.0,
                'bias_indicators': 0.0
            }
            
            # 1. Domain reputation scoring
            factors['domain_reputation'] = self._score_domain_reputation(domain)
            
            # 2. Content quality analysis
            factors['content_quality'] = self._analyze_content_quality(content)
            
            # 3. Author credibility (if available in metadata)
            factors['author_credibility'] = self._assess_author_credibility(metadata)
            
            # 4. Publication standards
            factors['publication_standards'] = self._evaluate_publication_standards(content, domain)
            
            # 5. Citation quality
            factors['citation_quality'] = self._analyze_citation_quality(content)
            
            # 6. Bias indicators
            factors['bias_indicators'] = self._detect_bias_indicators(content)
            
            # Calculate weighted raw score
            weights = {
                'domain_reputation': 0.25,
                'content_quality': 0.20,
                'author_credibility': 0.15,
                'publication_standards': 0.15,
                'citation_quality': 0.15,
                'bias_indicators': 0.10
            }
            
            raw_score = sum(factors[factor] * weights[factor] for factor in factors)
            raw_score = max(0.0, min(10.0, raw_score))  # Clamp to 0-10
            
            # Determine source type
            source_type = self._classify_source_type(domain, content)
            
            # Determine credibility level
            credibility_level = self._determine_credibility_level(raw_score)
            
            # Calculate confidence in assessment
            confidence = self._calculate_credibility_confidence(factors, metadata)
            
            return SourceCredibilityScore(
                url=url,
                domain=domain,
                source_type=source_type,
                credibility_level=credibility_level,
                raw_score=raw_score,
                factors=factors,
                confidence=confidence,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Source credibility assessment failed: {e}")
            return self._create_default_credibility_score(url)
    
    def _score_domain_reputation(self, domain: str) -> float:
        """Score domain reputation based on known credible sources"""
        
        # Check for high credibility domains
        for trusted_domain in self.high_credibility_domains:
            if trusted_domain in domain:
                return 9.0  # Very high credibility
        
        # Check for medium credibility domains
        for medium_domain in self.medium_credibility_domains:
            if medium_domain in domain:
                return 6.5  # Medium-high credibility
        
        # Check for low credibility indicators
        for low_indicator in self.low_credibility_indicators:
            if low_indicator in domain:
                return 3.0  # Low credibility
        
        # Check domain characteristics
        score = 5.0  # Default neutral score
        
        # Government and educational domains
        if domain.endswith('.gov') or domain.endswith('.edu'):
            score = 8.5
        
        # Academic institutions
        elif 'university' in domain or 'college' in domain or 'institute' in domain:
            score = 7.5
        
        # Organization domains
        elif domain.endswith('.org'):
            score = 6.0
        
        # Commercial domains (default)
        elif domain.endswith('.com'):
            score = 5.0
        
        return score
    
    def _analyze_content_quality(self, content: str) -> float:
        """Analyze content quality based on linguistic and structural indicators"""
        try:
            if not content or len(content) < 100:
                return 2.0  # Too short for quality analysis
            
            score = 5.0  # Base score
            content_lower = content.lower()
            
            # Positive quality indicators
            positive_count = sum(1 for indicator in self.quality_indicators['positive'] 
                               if indicator in content_lower)
            score += min(positive_count * 0.5, 3.0)  # Max +3.0
            
            # Negative quality indicators
            negative_count = sum(1 for indicator in self.quality_indicators['negative'] 
                               if indicator in content_lower)
            score -= min(negative_count * 0.3, 2.0)  # Max -2.0
            
            # Length and structure analysis
            word_count = len(content.split())
            if word_count > 500:
                score += 1.0  # Substantial content
            elif word_count < 100:
                score -= 1.0  # Too brief
            
            # Citation and reference indicators
            citation_indicators = ['http', 'www', 'doi:', 'isbn:', 'source:', 'reference:']
            citation_count = sum(1 for indicator in citation_indicators 
                               if indicator in content_lower)
            score += min(citation_count * 0.2, 1.0)  # Max +1.0
            
            # Professional language indicators
            professional_terms = ['analysis', 'research', 'study', 'data', 'methodology', 
                                 'findings', 'conclusion', 'evidence', 'statistics']
            professional_count = sum(1 for term in professional_terms 
                                   if term in content_lower)
            score += min(professional_count * 0.1, 1.0)  # Max +1.0
            
            return max(0.0, min(10.0, score))
            
        except Exception as e:
            logger.error(f"Content quality analysis failed: {e}")
            return 5.0  # Default neutral score
    
    def _assess_author_credibility(self, metadata: Optional[Dict]) -> float:
        """Assess author credibility from metadata"""
        if not metadata:
            return 5.0  # Neutral score when no metadata
        
        score = 5.0
        
        # Check for author information
        author = metadata.get('author', '').lower()
        if author:
            # Academic credentials
            if any(credential in author for credential in ['phd', 'dr.', 'professor', 'researcher']):
                score += 2.0
            
            # Professional titles
            elif any(title in author for title in ['analyst', 'director', 'manager', 'expert']):
                score += 1.0
        
        # Check for institutional affiliation
        affiliation = metadata.get('affiliation', '').lower()
        if affiliation:
            if any(inst in affiliation for inst in ['university', 'institute', 'research', 'government']):
                score += 1.5
        
        return max(0.0, min(10.0, score))
    
    def _evaluate_publication_standards(self, content: str, domain: str) -> float:
        """Evaluate publication standards and editorial quality"""
        score = 5.0
        content_lower = content.lower()
        
        # Check for editorial standards indicators
        editorial_indicators = [
            'editor', 'editorial', 'peer review', 'fact check', 'verified',
            'published', 'updated', 'correction', 'retraction'
        ]
        
        editorial_count = sum(1 for indicator in editorial_indicators 
                            if indicator in content_lower)
        score += min(editorial_count * 0.3, 2.0)
        
        # Check for transparency indicators
        transparency_indicators = [
            'methodology', 'sources', 'disclaimer', 'conflict of interest',
            'funding', 'sponsor', 'about the author'
        ]
        
        transparency_count = sum(1 for indicator in transparency_indicators 
                               if indicator in content_lower)
        score += min(transparency_count * 0.2, 1.5)
        
        # Domain-specific adjustments
        if any(domain_type in domain for domain_type in ['.gov', '.edu', 'reuters', 'bloomberg']):
            score += 1.0  # Higher standards expected
        
        return max(0.0, min(10.0, score))
    
    def _analyze_citation_quality(self, content: str) -> float:
        """Analyze quality and quantity of citations"""
        score = 5.0
        content_lower = content.lower()
        
        # Count different types of citations
        url_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content))
        doi_count = len(re.findall(r'doi:\s*10\.\d+', content_lower))
        reference_count = len(re.findall(r'reference[s]?:', content_lower))
        
        # Score based on citation density
        total_citations = url_count + doi_count + reference_count
        word_count = len(content.split())
        
        if word_count > 0:
            citation_density = total_citations / (word_count / 100)  # Citations per 100 words
            
            if citation_density > 2.0:
                score += 2.0  # Well-cited
            elif citation_density > 1.0:
                score += 1.0  # Adequately cited
            elif citation_density < 0.1:
                score -= 1.0  # Poorly cited
        
        # Bonus for academic citations (DOI)
        if doi_count > 0:
            score += 1.0
        
        return max(0.0, min(10.0, score))
    
    def _detect_bias_indicators(self, content: str) -> float:
        """Detect potential bias indicators (higher score = less bias)"""
        score = 7.0  # Start with assumption of moderate bias
        content_lower = content.lower()
        
        # Emotional language indicators (negative)
        emotional_words = [
            'shocking', 'amazing', 'incredible', 'unbelievable', 'devastating',
            'outrageous', 'scandalous', 'revolutionary', 'groundbreaking'
        ]
        emotional_count = sum(1 for word in emotional_words if word in content_lower)
        score -= min(emotional_count * 0.3, 2.0)
        
        # Balanced language indicators (positive)
        balanced_words = [
            'however', 'although', 'nevertheless', 'on the other hand',
            'alternatively', 'in contrast', 'conversely', 'meanwhile'
        ]
        balanced_count = sum(1 for phrase in balanced_words if phrase in content_lower)
        score += min(balanced_count * 0.2, 1.5)
        
        # Objective language indicators (positive)
        objective_words = [
            'according to', 'data shows', 'research indicates', 'study finds',
            'analysis reveals', 'statistics demonstrate', 'evidence suggests'
        ]
        objective_count = sum(1 for phrase in objective_words if phrase in content_lower)
        score += min(objective_count * 0.3, 2.0)
        
        return max(0.0, min(10.0, score))
    
    def _classify_source_type(self, domain: str, content: str) -> SourceType:
        """Classify the type of source"""
        domain_lower = domain.lower()
        content_lower = content.lower()
        
        # Government sources
        if '.gov' in domain_lower:
            return SourceType.GOVERNMENT
        
        # Academic sources
        if ('.edu' in domain_lower or 'university' in domain_lower or 
            'college' in domain_lower or 'arxiv' in domain_lower or
            'pubmed' in domain_lower):
            return SourceType.ACADEMIC
        
        # News media
        news_indicators = ['news', 'times', 'post', 'herald', 'tribune', 'reuters', 
                          'bloomberg', 'cnn', 'bbc', 'npr']
        if any(indicator in domain_lower for indicator in news_indicators):
            return SourceType.NEWS_MEDIA
        
        # Industry reports
        industry_indicators = ['mckinsey', 'bcg', 'deloitte', 'pwc', 'gartner', 
                              'forrester', 'idc', 'statista']
        if any(indicator in domain_lower for indicator in industry_indicators):
            return SourceType.INDUSTRY_REPORT
        
        # Social media
        social_indicators = ['twitter', 'facebook', 'linkedin', 'reddit', 'instagram']
        if any(indicator in domain_lower for indicator in social_indicators):
            return SourceType.SOCIAL_MEDIA
        
        # Blogs
        blog_indicators = ['blog', 'wordpress', 'medium', 'substack']
        if any(indicator in domain_lower for indicator in blog_indicators):
            return SourceType.BLOG
        
        # Forums
        forum_indicators = ['forum', 'discussion', 'community', 'quora']
        if any(indicator in domain_lower for indicator in forum_indicators):
            return SourceType.FORUM
        
        return SourceType.UNKNOWN
    
    def _determine_credibility_level(self, raw_score: float) -> CredibilityLevel:
        """Determine credibility level from raw score"""
        if raw_score >= 9.0:
            return CredibilityLevel.VERY_HIGH
        elif raw_score >= 7.0:
            return CredibilityLevel.HIGH
        elif raw_score >= 5.0:
            return CredibilityLevel.MEDIUM
        elif raw_score >= 3.0:
            return CredibilityLevel.LOW
        else:
            return CredibilityLevel.VERY_LOW
    
    def _calculate_credibility_confidence(self, factors: Dict[str, float], 
                                        metadata: Optional[Dict]) -> float:
        """Calculate confidence in credibility assessment"""
        confidence = 0.7  # Base confidence
        
        # Boost confidence if we have metadata
        if metadata:
            confidence += 0.1
        
        # Boost confidence if multiple factors agree
        high_factors = sum(1 for score in factors.values() if score > 7.0)
        low_factors = sum(1 for score in factors.values() if score < 3.0)
        
        if high_factors >= 3:
            confidence += 0.15
        elif low_factors >= 3:
            confidence += 0.1  # Consistent low scores also increase confidence
        
        return min(confidence, 1.0)
    
    def _validate_data_freshness(self, content: str, metadata: Optional[Dict] = None) -> DataFreshnessCheck:
        """Validate data freshness and detect staleness indicators"""
        try:
            # Try to extract publish date from metadata
            publish_date = None
            if metadata and 'publish_date' in metadata:
                try:
                    if isinstance(metadata['publish_date'], str):
                        publish_date = datetime.fromisoformat(metadata['publish_date'].replace('Z', '+00:00'))
                    elif isinstance(metadata['publish_date'], datetime):
                        publish_date = metadata['publish_date']
                except:
                    pass
            
            # Calculate age if we have publish date
            age_days = None
            if publish_date:
                age_days = (datetime.now() - publish_date.replace(tzinfo=None)).days
            
            # Detect staleness indicators in content
            staleness_indicators = []
            content_lower = content.lower()
            
            for indicator in self.staleness_keywords:
                if indicator in content_lower:
                    staleness_indicators.append(indicator)
            
            # Calculate freshness score
            freshness_score = self._calculate_freshness_score(age_days, staleness_indicators, content)
            
            # Determine if data is fresh
            is_fresh = freshness_score > 0.6
            
            return DataFreshnessCheck(
                content=content[:200] + "..." if len(content) > 200 else content,
                publish_date=publish_date,
                age_days=age_days,
                is_fresh=is_fresh,
                freshness_score=freshness_score,
                staleness_indicators=staleness_indicators
            )
            
        except Exception as e:
            logger.error(f"Data freshness validation failed: {e}")
            return DataFreshnessCheck(
                content="Error in analysis",
                publish_date=None,
                age_days=None,
                is_fresh=False,
                freshness_score=0.5,
                staleness_indicators=["validation_error"]
            )
    
    def _calculate_freshness_score(self, age_days: Optional[int], 
                                 staleness_indicators: List[str], content: str) -> float:
        """Calculate freshness score based on multiple factors"""
        score = 0.8  # Base score assuming reasonably fresh
        
        # Age-based scoring
        if age_days is not None:
            if age_days <= 30:
                score = 1.0  # Very fresh
            elif age_days <= 90:
                score = 0.8  # Fresh
            elif age_days <= 180:
                score = 0.6  # Moderately fresh
            elif age_days <= 365:
                score = 0.4  # Getting stale
            else:
                score = 0.2  # Stale
        
        # Staleness indicators penalty
        staleness_penalty = min(len(staleness_indicators) * 0.1, 0.4)
        score -= staleness_penalty
        
        # Current year references (positive indicator)
        current_year = datetime.now().year
        if str(current_year) in content:
            score += 0.1
        
        # Recent month references
        recent_months = ['january', 'february', 'march', 'april', 'may', 'june',
                        'july', 'august', 'september', 'october', 'november', 'december']
        current_month = datetime.now().strftime('%B').lower()
        
        if current_month in content.lower():
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _perform_cross_reference_check(self, content: str) -> List[CrossReferenceResult]:
        """Perform simplified cross-reference fact-checking"""
        # This is a simplified implementation for demonstration
        # In production, this would integrate with fact-checking APIs
        
        results = []
        
        # Extract key claims (simplified)
        claims = self._extract_key_claims(content)
        
        for claim in claims[:3]:  # Limit to top 3 claims
            # Simulate cross-reference checking
            result = CrossReferenceResult(
                claim=claim,
                supporting_sources=[],  # Would be populated by real fact-checking
                contradicting_sources=[],
                confidence_score=0.7,  # Default moderate confidence
                consensus_level="moderate",
                verification_status="unverified"  # Default status
            )
            results.append(result)
        
        return results
    
    def _extract_key_claims(self, content: str) -> List[str]:
        """Extract key factual claims from content"""
        # Simplified claim extraction
        sentences = content.split('.')
        claims = []
        
        # Look for sentences with factual indicators
        factual_indicators = ['percent', '%', 'million', 'billion', 'study shows', 
                             'research indicates', 'data reveals', 'according to']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                any(indicator in sentence.lower() for indicator in factual_indicators)):
                claims.append(sentence[:100] + "..." if len(sentence) > 100 else sentence)
        
        return claims[:5]  # Return top 5 claims
    
    def _calculate_overall_quality(self, credibility_score: SourceCredibilityScore,
                                 freshness_check: DataFreshnessCheck,
                                 cross_reference_results: List[CrossReferenceResult]) -> float:
        """Calculate overall quality score from all validation components"""
        
        # Weighted combination of quality factors
        weights = {
            'credibility': 0.5,
            'freshness': 0.3,
            'cross_reference': 0.2
        }
        
        # Normalize credibility score to 0-1
        credibility_normalized = credibility_score.raw_score / 10.0
        
        # Freshness score is already 0-1
        freshness_normalized = freshness_check.freshness_score
        
        # Cross-reference score (average of all checks)
        if cross_reference_results:
            cross_ref_normalized = sum(result.confidence_score for result in cross_reference_results) / len(cross_reference_results)
        else:
            cross_ref_normalized = 0.5  # Neutral if no cross-reference data
        
        overall_score = (
            weights['credibility'] * credibility_normalized +
            weights['freshness'] * freshness_normalized +
            weights['cross_reference'] * cross_ref_normalized
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _assign_validation_grade(self, overall_quality_score: float) -> str:
        """Assign letter grade based on overall quality score"""
        if overall_quality_score >= 0.95:
            return "A+"
        elif overall_quality_score >= 0.9:
            return "A"
        elif overall_quality_score >= 0.85:
            return "A-"
        elif overall_quality_score >= 0.8:
            return "B+"
        elif overall_quality_score >= 0.75:
            return "B"
        elif overall_quality_score >= 0.7:
            return "B-"
        elif overall_quality_score >= 0.65:
            return "C+"
        elif overall_quality_score >= 0.6:
            return "C"
        elif overall_quality_score >= 0.55:
            return "C-"
        elif overall_quality_score >= 0.5:
            return "D+"
        elif overall_quality_score >= 0.45:
            return "D"
        elif overall_quality_score >= 0.4:
            return "D-"
        else:
            return "F"
    
    def _identify_critical_issues(self, credibility_score: SourceCredibilityScore,
                                freshness_check: DataFreshnessCheck,
                                cross_reference_results: List[CrossReferenceResult]) -> List[str]:
        """Identify critical issues that affect data reliability"""
        issues = []
        
        # Credibility issues
        if credibility_score.credibility_level == CredibilityLevel.VERY_LOW:
            issues.append("Very low source credibility - unreliable for business decisions")
        elif credibility_score.credibility_level == CredibilityLevel.LOW:
            issues.append("Low source credibility - use with caution")
        
        # Freshness issues
        if not freshness_check.is_fresh:
            if freshness_check.age_days and freshness_check.age_days > 365:
                issues.append("Data is over 1 year old - may be outdated")
            elif freshness_check.staleness_indicators:
                issues.append("Contains staleness indicators - verify currency")
        
        # Cross-reference issues
        conflicted_results = [r for r in cross_reference_results if r.consensus_level == "conflicted"]
        if conflicted_results:
            issues.append("Conflicting information found in cross-references")
        
        # Confidence issues
        if credibility_score.confidence < 0.5:
            issues.append("Low confidence in credibility assessment")
        
        return issues
    
    def _generate_recommendations(self, credibility_score: SourceCredibilityScore,
                                freshness_check: DataFreshnessCheck,
                                critical_issues: List[str]) -> List[str]:
        """Generate actionable recommendations for data usage"""
        recommendations = []
        
        # Credibility-based recommendations
        if credibility_score.credibility_level in [CredibilityLevel.VERY_LOW, CredibilityLevel.LOW]:
            recommendations.append("Seek additional sources from higher credibility domains")
            recommendations.append("Verify claims through independent fact-checking")
        
        # Freshness-based recommendations
        if not freshness_check.is_fresh:
            recommendations.append("Look for more recent data on this topic")
            recommendations.append("Check if information has been updated or superseded")
        
        # General quality recommendations
        if len(critical_issues) > 2:
            recommendations.append("Consider excluding this source from analysis")
        elif len(critical_issues) > 0:
            recommendations.append("Use this source as supporting evidence only")
        else:
            recommendations.append("Source meets quality standards for business analysis")
        
        # Source type specific recommendations
        if credibility_score.source_type == SourceType.SOCIAL_MEDIA:
            recommendations.append("Social media source - verify through traditional media")
        elif credibility_score.source_type == SourceType.BLOG:
            recommendations.append("Blog source - check author credentials and citations")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _create_error_validation_result(self, url: str, error_message: str) -> ValidationResult:
        """Create error validation result"""
        error_credibility = SourceCredibilityScore(
            url=url, domain="error", source_type=SourceType.UNKNOWN,
            credibility_level=CredibilityLevel.VERY_LOW, raw_score=0.0,
            factors={}, confidence=0.0, last_updated=datetime.now()
        )
        
        error_freshness = DataFreshnessCheck(
            content="Error in analysis", publish_date=None, age_days=None,
            is_fresh=False, freshness_score=0.0, staleness_indicators=["error"]
        )
        
        return ValidationResult(
            source_url=url, content_hash="error",
            credibility_score=error_credibility, freshness_check=error_freshness,
            cross_reference_results=[], overall_quality_score=0.0,
            validation_grade="F", critical_issues=[f"Validation error: {error_message}"],
            recommendations=["Unable to validate - exclude from analysis"],
            validation_timestamp=datetime.now()
        )
    
    def _create_default_credibility_score(self, url: str) -> SourceCredibilityScore:
        """Create default credibility score for error cases"""
        return SourceCredibilityScore(
            url=url, domain="unknown", source_type=SourceType.UNKNOWN,
            credibility_level=CredibilityLevel.MEDIUM, raw_score=5.0,
            factors={'default': 5.0}, confidence=0.5, last_updated=datetime.now()
        )

# Factory function
def create_enhanced_data_validator() -> EnhancedDataValidator:
    """Create enhanced data validator instance"""
    return EnhancedDataValidator()

# Test function
def test_enhanced_data_validator():
    """Test the enhanced data validation system"""
    print("Testing Enhanced Data Validation System...")
    
    validator = create_enhanced_data_validator()
    
    print("\n1. Testing high-quality source validation...")
    
    # Test with high-quality content
    high_quality_url = "https://www.reuters.com/business/tech/ai-market-analysis-2024"
    high_quality_content = """
    According to a comprehensive study published by McKinsey & Company, the artificial intelligence market 
    is projected to reach $190 billion by 2025. The research, conducted over 18 months and surveying 
    2,000 executives across 15 industries, reveals significant growth in AI adoption.
    
    Data from the survey shows that 67% of companies have increased their AI investments in 2024, 
    compared to 45% in 2023. The study methodology involved structured interviews and quantitative 
    analysis of financial reports from Fortune 500 companies.
    
    Dr. Sarah Johnson, lead researcher at MIT's AI Lab and co-author of the study, noted that 
    "the evidence suggests a fundamental shift in how businesses approach artificial intelligence."
    
    The report cites multiple peer-reviewed sources and includes references to academic publications 
    in Nature and Science journals. Statistical analysis was performed using standard methodologies 
    with 95% confidence intervals.
    """
    
    high_quality_metadata = {
        'author': 'Dr. Sarah Johnson, PhD',
        'publish_date': '2024-06-15T10:30:00Z',
        'affiliation': 'MIT AI Laboratory'
    }
    
    result_high = validator.validate_source_comprehensive(
        high_quality_url, high_quality_content, high_quality_metadata
    )
    
    print(f"✅ High Quality Source:")
    print(f"   URL: {result_high.source_url}")
    print(f"   Validation Grade: {result_high.validation_grade}")
    print(f"   Overall Quality: {result_high.overall_quality_score:.3f}")
    print(f"   Credibility Level: {result_high.credibility_score.credibility_level.value}")
    print(f"   Credibility Score: {result_high.credibility_score.raw_score:.1f}/10")
    print(f"   Is Fresh: {result_high.freshness_check.is_fresh}")
    print(f"   Critical Issues: {len(result_high.critical_issues)}")
    print(f"   Recommendations: {len(result_high.recommendations)}")
    
    print("\n2. Testing low-quality source validation...")
    
    # Test with low-quality content
    low_quality_url = "https://myblog.wordpress.com/ai-predictions"
    low_quality_content = """
    I think AI is going to be huge! Maybe it will make billions of dollars, probably more than anyone expects.
    
    Some guy on Twitter said that AI companies are making crazy money. This is shocking news that will 
    revolutionize everything! I heard from a friend that Google is secretly working on something amazing.
    
    In my opinion, everyone should invest in AI right now. It's unbelievable how much money you could make!
    The rumors suggest that this could be the next big thing.
    """
    
    result_low = validator.validate_source_comprehensive(low_quality_url, low_quality_content)
    
    print(f"✅ Low Quality Source:")
    print(f"   URL: {result_low.source_url}")
    print(f"   Validation Grade: {result_low.validation_grade}")
    print(f"   Overall Quality: {result_low.overall_quality_score:.3f}")
    print(f"   Credibility Level: {result_low.credibility_score.credibility_level.value}")
    print(f"   Credibility Score: {result_low.credibility_score.raw_score:.1f}/10")
    print(f"   Is Fresh: {result_low.freshness_check.is_fresh}")
    print(f"   Critical Issues: {len(result_low.critical_issues)}")
    print(f"   Recommendations: {len(result_low.recommendations)}")
    
    print("\n3. Testing stale data validation...")
    
    # Test with stale content
    stale_url = "https://techreport.com/old-analysis"
    stale_content = """
    Last year's analysis showed that the market was growing. In 2020, we saw significant changes 
    in the industry. The former CEO mentioned that legacy systems were being deprecated.
    
    Previously published reports indicated strong growth, but these findings may now be outdated.
    """
    
    stale_metadata = {
        'publish_date': '2021-03-15T14:20:00Z'
    }
    
    result_stale = validator.validate_source_comprehensive(stale_url, stale_content, stale_metadata)
    
    print(f"✅ Stale Data Source:")
    print(f"   URL: {result_stale.source_url}")
    print(f"   Validation Grade: {result_stale.validation_grade}")
    print(f"   Age (days): {result_stale.freshness_check.age_days}")
    print(f"   Is Fresh: {result_stale.freshness_check.is_fresh}")
    print(f"   Freshness Score: {result_stale.freshness_check.freshness_score:.3f}")
    print(f"   Staleness Indicators: {len(result_stale.freshness_check.staleness_indicators)}")
    
    print("\n🎉 Enhanced data validation test completed successfully!")
    
    return {
        "high_quality_validation": {
            "grade": result_high.validation_grade,
            "quality_score": result_high.overall_quality_score,
            "credibility_level": result_high.credibility_score.credibility_level.value,
            "credibility_score": result_high.credibility_score.raw_score,
            "is_fresh": result_high.freshness_check.is_fresh,
            "critical_issues": len(result_high.critical_issues)
        },
        "low_quality_validation": {
            "grade": result_low.validation_grade,
            "quality_score": result_low.overall_quality_score,
            "credibility_level": result_low.credibility_score.credibility_level.value,
            "credibility_score": result_low.credibility_score.raw_score,
            "critical_issues": len(result_low.critical_issues)
        },
        "stale_data_validation": {
            "grade": result_stale.validation_grade,
            "age_days": result_stale.freshness_check.age_days,
            "is_fresh": result_stale.freshness_check.is_fresh,
            "freshness_score": result_stale.freshness_check.freshness_score,
            "staleness_indicators": len(result_stale.freshness_check.staleness_indicators)
        },
        "system_status": "enhanced_validation_operational"
    }

if __name__ == "__main__":
    # Run enhanced data validation tests
    test_enhanced_data_validator()

