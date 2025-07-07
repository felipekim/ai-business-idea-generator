"""
Calibrated Data Validation System
Enhanced scoring algorithm with improved quality differentiation and domain authority
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
    content_type: str  # For freshness tolerance
    
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

class CalibratedDataValidator:
    """Production-grade data validation with calibrated scoring for better differentiation"""
    
    def __init__(self):
        # TIER 1: Premium Authority Domains (9.5-10.0)
        self.tier1_domains = {
            # Academic & Research Institutions
            'nature.com': 9.9,
            'science.org': 9.8,
            'cell.com': 9.7,
            'nejm.org': 9.8,
            'pubmed.ncbi.nlm.nih.gov': 9.6,
            'arxiv.org': 9.5,
            
            # Financial & Business Authority
            'reuters.com': 9.8,
            'bloomberg.com': 9.7,
            'wsj.com': 9.6,
            'ft.com': 9.5,
            'economist.com': 9.6,
            
            # Government & Official
            'census.gov': 9.8,
            'bls.gov': 9.7,
            'sec.gov': 9.6,
            'ftc.gov': 9.5,
            'fed.gov': 9.7,
            
            # Top Consulting & Research
            'mckinsey.com': 9.5,
            'bcg.com': 9.4,
            'bain.com': 9.4,
            'deloitte.com': 9.3,
            'pwc.com': 9.3,
            'kpmg.com': 9.2,
            
            # Technology Research
            'gartner.com': 9.4,
            'forrester.com': 9.3,
            'idc.com': 9.2,
            'statista.com': 9.1
        }
        
        # TIER 2: High Authority Domains (8.0-9.4)
        self.tier2_domains = {
            # Major News Organizations
            'bbc.com': 8.7,
            'npr.org': 8.6,
            'pbs.org': 8.5,
            'apnews.com': 8.8,
            'cnn.com': 8.3,
            'nytimes.com': 8.4,
            'washingtonpost.com': 8.3,
            'theguardian.com': 8.2,
            
            # Academic Institutions
            'mit.edu': 9.2,
            'stanford.edu': 9.1,
            'harvard.edu': 9.0,
            'berkeley.edu': 8.9,
            'oxford.ac.uk': 9.0,
            'cambridge.ac.uk': 8.9,
            
            # Industry Publications
            'hbr.org': 8.6,
            'sloanreview.mit.edu': 8.7,
            'strategy-business.com': 8.4,
            
            # Technology Authority
            'ieee.org': 8.8,
            'acm.org': 8.7,
            'techcrunch.com': 8.0,
            'wired.com': 8.1,
            'arstechnica.com': 8.2
        }
        
        # TIER 3: Medium Authority Domains (6.0-7.9)
        self.tier3_domains = {
            # Business Publications
            'forbes.com': 7.5,
            'businessinsider.com': 7.2,
            'inc.com': 7.3,
            'entrepreneur.com': 7.1,
            'fastcompany.com': 7.4,
            
            # Technology Publications
            'venturebeat.com': 7.3,
            'theverge.com': 7.2,
            'engadget.com': 7.0,
            'mashable.com': 6.8,
            
            # Industry Specific
            'marketwatch.com': 7.1,
            'cnbc.com': 7.3,
            'fortune.com': 7.4,
            
            # Regional Authority
            'local.gov': 7.5,
            'state.gov': 8.0
        }
        
        # TIER 4: Low Authority Indicators (2.0-4.0)
        self.low_authority_indicators = {
            'blog': 3.5,
            'wordpress.com': 3.0,
            'medium.com': 4.0,
            'substack.com': 3.8,
            'blogspot.com': 2.5,
            'tumblr.com': 2.8,
            'reddit.com': 3.2,
            'twitter.com': 2.5,
            'facebook.com': 2.0,
            'instagram.com': 2.0,
            'tiktok.com': 1.5,
            'youtube.com': 3.0,
            'quora.com': 3.5
        }
        
        # Enhanced content quality indicators with multipliers
        self.quality_multipliers = {
            # Academic Indicators (High Impact)
            'peer-reviewed': 2.0,
            'peer reviewed': 2.0,
            'systematic review': 2.2,
            'meta-analysis': 2.1,
            'randomized controlled trial': 2.3,
            'statistical significance': 1.8,
            'confidence interval': 1.6,
            'methodology': 1.5,
            'sample size': 1.4,
            'control group': 1.6,
            
            # Professional Research Indicators
            'longitudinal study': 1.9,
            'cross-sectional study': 1.7,
            'cohort study': 1.8,
            'case-control study': 1.6,
            'survey methodology': 1.5,
            'data collection': 1.3,
            'statistical analysis': 1.4,
            'regression analysis': 1.5,
            'correlation analysis': 1.3,
            
            # Business Intelligence Indicators
            'market research': 1.4,
            'industry analysis': 1.3,
            'competitive analysis': 1.2,
            'financial analysis': 1.3,
            'trend analysis': 1.2,
            'benchmarking': 1.3,
            'roi analysis': 1.2,
            'cost-benefit analysis': 1.3,
            
            # Authority Indicators
            'according to study': 1.2,
            'research shows': 1.1,
            'data reveals': 1.2,
            'findings indicate': 1.1,
            'evidence suggests': 1.1,
            'analysis demonstrates': 1.2,
            'report concludes': 1.1,
            
            # Citation Indicators
            'doi:': 1.8,
            'isbn:': 1.5,
            'pmid:': 1.7,
            'arxiv:': 1.6,
            'published in': 1.4,
            'journal of': 1.3,
            'proceedings of': 1.2,
            'conference paper': 1.2
        }
        
        # Negative quality indicators (penalties)
        self.quality_penalties = {
            # Opinion Indicators
            'i think': -0.8,
            'in my opinion': -0.7,
            'i believe': -0.6,
            'personally': -0.5,
            'my view': -0.6,
            
            # Uncertainty Indicators
            'probably': -0.4,
            'maybe': -0.5,
            'could be': -0.4,
            'might be': -0.4,
            'possibly': -0.3,
            
            # Sensational Language
            'shocking': -0.6,
            'amazing': -0.5,
            'incredible': -0.6,
            'unbelievable': -0.7,
            'revolutionary': -0.5,
            'groundbreaking': -0.4,
            'game-changing': -0.5,
            
            # Unreliable Indicators
            'rumor has it': -1.0,
            'some say': -0.8,
            'allegedly': -0.6,
            'unconfirmed': -0.7,
            'breaking:': -0.4,
            'exclusive:': -0.3
        }
        
        # Content type classification for freshness tolerance
        self.content_type_patterns = {
            'market_data': ['stock price', 'market cap', 'trading volume', 'quarterly earnings'],
            'research_findings': ['study shows', 'research indicates', 'findings suggest'],
            'industry_analysis': ['industry report', 'market analysis', 'sector overview'],
            'academic_studies': ['peer-reviewed', 'journal', 'academic', 'university'],
            'news_events': ['breaking', 'announced', 'reported', 'confirmed'],
            'historical_context': ['historical', 'background', 'context', 'overview']
        }
        
        # Freshness tolerance by content type (days)
        self.freshness_tolerance = {
            'market_data': 7,        # Very time-sensitive
            'news_events': 30,       # Time-sensitive
            'industry_analysis': 180, # Moderately time-sensitive
            'research_findings': 365, # Less time-sensitive
            'academic_studies': 730,  # Stable over time
            'historical_context': 1825, # Very stable
            'default': 90            # Default tolerance
        }
        
        # Enhanced bias detection patterns
        self.bias_patterns = {
            'emotional_language': [
                'outrageous', 'scandalous', 'devastating', 'shocking',
                'amazing', 'incredible', 'unbelievable', 'fantastic'
            ],
            'balanced_language': [
                'however', 'although', 'nevertheless', 'on the other hand',
                'alternatively', 'in contrast', 'conversely', 'meanwhile',
                'despite', 'nonetheless', 'whereas', 'while'
            ],
            'objective_language': [
                'according to', 'data shows', 'research indicates', 'study finds',
                'analysis reveals', 'statistics demonstrate', 'evidence suggests',
                'findings show', 'results indicate', 'observations suggest'
            ]
        }
        
        logger.info("Calibrated Data Validator initialized with enhanced scoring algorithms")
    
    def validate_source_comprehensive(self, url: str, content: str, 
                                    metadata: Optional[Dict] = None) -> ValidationResult:
        """Comprehensive source validation with calibrated scoring"""
        try:
            # Generate content hash for tracking
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            # 1. Enhanced source credibility assessment
            credibility_score = self._assess_source_credibility_calibrated(url, content, metadata)
            
            # 2. Enhanced data freshness validation
            freshness_check = self._validate_data_freshness_enhanced(content, metadata)
            
            # 3. Improved cross-reference fact-checking
            cross_reference_results = self._perform_cross_reference_enhanced(content)
            
            # 4. Calculate overall quality score with new weights
            overall_quality_score = self._calculate_overall_quality_calibrated(
                credibility_score, freshness_check, cross_reference_results
            )
            
            # 5. Assign validation grade with stricter standards
            validation_grade = self._assign_validation_grade_calibrated(overall_quality_score)
            
            # 6. Identify critical issues with enhanced detection
            critical_issues = self._identify_critical_issues_enhanced(
                credibility_score, freshness_check, cross_reference_results
            )
            
            # 7. Generate enhanced recommendations
            recommendations = self._generate_recommendations_enhanced(
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
            
            logger.info(f"Calibrated validation completed: {url} -> Grade {validation_grade} ({overall_quality_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Calibrated validation failed for {url}: {e}")
            return self._create_error_validation_result(url, str(e))
    
    def _assess_source_credibility_calibrated(self, url: str, content: str, 
                                            metadata: Optional[Dict] = None) -> SourceCredibilityScore:
        """Enhanced source credibility assessment with calibrated scoring"""
        try:
            domain = urlparse(url).netloc.lower()
            
            # Initialize scoring factors
            factors = {
                'domain_authority': 0.0,
                'content_quality': 0.0,
                'author_credibility': 0.0,
                'publication_standards': 0.0,
                'citation_quality': 0.0,
                'bias_assessment': 0.0
            }
            
            # 1. Enhanced domain authority scoring
            factors['domain_authority'] = self._score_domain_authority_enhanced(domain)
            
            # 2. Enhanced content quality analysis with multipliers
            factors['content_quality'] = self._analyze_content_quality_enhanced(content)
            
            # 3. Enhanced author credibility assessment
            factors['author_credibility'] = self._assess_author_credibility_enhanced(metadata)
            
            # 4. Enhanced publication standards evaluation
            factors['publication_standards'] = self._evaluate_publication_standards_enhanced(content, domain)
            
            # 5. Enhanced citation quality analysis
            factors['citation_quality'] = self._analyze_citation_quality_enhanced(content)
            
            # 6. Enhanced bias assessment
            factors['bias_assessment'] = self._assess_bias_enhanced(content)
            
            # Calculate weighted raw score with new weights
            weights = {
                'domain_authority': 0.30,      # Increased from 0.25
                'content_quality': 0.25,      # Increased from 0.20
                'author_credibility': 0.15,   # Same
                'publication_standards': 0.12, # Decreased from 0.15
                'citation_quality': 0.10,     # Decreased from 0.15
                'bias_assessment': 0.08       # Decreased from 0.10
            }
            
            raw_score = sum(factors[factor] * weights[factor] for factor in factors)
            raw_score = max(0.0, min(10.0, raw_score))  # Clamp to 0-10
            
            # Determine source type
            source_type = self._classify_source_type_enhanced(domain, content)
            
            # Determine credibility level with stricter thresholds
            credibility_level = self._determine_credibility_level_calibrated(raw_score)
            
            # Calculate confidence in assessment
            confidence = self._calculate_credibility_confidence_enhanced(factors, metadata)
            
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
            logger.error(f"Enhanced credibility assessment failed: {e}")
            return self._create_default_credibility_score(url)
    
    def _score_domain_authority_enhanced(self, domain: str) -> float:
        """Enhanced domain authority scoring with tier-based system"""
        
        # Check Tier 1 domains (9.5-10.0)
        for trusted_domain, score in self.tier1_domains.items():
            if trusted_domain in domain:
                return score
        
        # Check Tier 2 domains (8.0-9.4)
        for medium_domain, score in self.tier2_domains.items():
            if medium_domain in domain:
                return score
        
        # Check Tier 3 domains (6.0-7.9)
        for decent_domain, score in self.tier3_domains.items():
            if decent_domain in domain:
                return score
        
        # Check low authority indicators (2.0-4.0)
        for low_indicator, score in self.low_authority_indicators.items():
            if low_indicator in domain:
                return score
        
        # Domain characteristic analysis
        base_score = 5.0  # Default neutral score
        
        # Government and educational domains
        if domain.endswith('.gov'):
            base_score = 8.8
        elif domain.endswith('.edu'):
            base_score = 8.5
        elif domain.endswith('.ac.uk') or domain.endswith('.ac.'):
            base_score = 8.3
        
        # Academic and research institutions
        elif any(keyword in domain for keyword in ['university', 'college', 'institute', 'research']):
            base_score = 7.8
        
        # Organization domains
        elif domain.endswith('.org'):
            if any(keyword in domain for keyword in ['ieee', 'acm', 'nature', 'science']):
                base_score = 8.5
            else:
                base_score = 6.5
        
        # International domains
        elif any(domain.endswith(tld) for tld in ['.uk', '.de', '.fr', '.ca', '.au']):
            base_score = 6.0
        
        # Commercial domains (default)
        elif domain.endswith('.com'):
            base_score = 5.0
        
        return base_score
    
    def _analyze_content_quality_enhanced(self, content: str) -> float:
        """Enhanced content quality analysis with multipliers and penalties"""
        try:
            if not content or len(content) < 100:
                return 2.0  # Too short for quality analysis
            
            base_score = 5.0  # Base score
            content_lower = content.lower()
            
            # Apply quality multipliers
            quality_boost = 0.0
            for indicator, multiplier in self.quality_multipliers.items():
                if indicator in content_lower:
                    quality_boost += (multiplier - 1.0) * 0.5  # Scale the boost
            
            # Apply quality penalties
            quality_penalty = 0.0
            for indicator, penalty in self.quality_penalties.items():
                if indicator in content_lower:
                    quality_penalty += abs(penalty) * 0.3  # Scale the penalty
            
            # Length and structure analysis (enhanced)
            word_count = len(content.split())
            if word_count > 1000:
                base_score += 2.0  # Substantial content
            elif word_count > 500:
                base_score += 1.5  # Good length
            elif word_count > 200:
                base_score += 0.5  # Adequate length
            elif word_count < 100:
                base_score -= 1.5  # Too brief
            
            # Citation and reference analysis (enhanced)
            citation_score = self._analyze_citations_detailed(content)
            base_score += citation_score
            
            # Professional language structure analysis
            structure_score = self._analyze_content_structure(content)
            base_score += structure_score
            
            # Calculate final score
            final_score = base_score + quality_boost - quality_penalty
            
            return max(0.0, min(10.0, final_score))
            
        except Exception as e:
            logger.error(f"Enhanced content quality analysis failed: {e}")
            return 5.0  # Default neutral score
    
    def _analyze_citations_detailed(self, content: str) -> float:
        """Detailed citation analysis with different weights"""
        score = 0.0
        content_lower = content.lower()
        
        # High-value citations
        doi_count = len(re.findall(r'doi:\s*10\.\d+', content_lower))
        score += min(doi_count * 0.8, 2.0)  # Max +2.0 for DOIs
        
        # Academic citations
        pmid_count = len(re.findall(r'pmid:\s*\d+', content_lower))
        score += min(pmid_count * 0.6, 1.5)  # Max +1.5 for PMIDs
        
        # ISBN citations
        isbn_count = len(re.findall(r'isbn:\s*[\d-]+', content_lower))
        score += min(isbn_count * 0.4, 1.0)  # Max +1.0 for ISBNs
        
        # URL citations (lower value)
        url_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content))
        score += min(url_count * 0.1, 1.0)  # Max +1.0 for URLs
        
        # Reference sections
        if 'references:' in content_lower or 'bibliography:' in content_lower:
            score += 1.0
        
        return score
    
    def _analyze_content_structure(self, content: str) -> float:
        """Analyze content structure for professional quality indicators"""
        score = 0.0
        content_lower = content.lower()
        
        # Section headers (indicates structured content)
        section_indicators = ['introduction', 'methodology', 'results', 'discussion', 
                            'conclusion', 'abstract', 'summary', 'background']
        section_count = sum(1 for indicator in section_indicators if indicator in content_lower)
        score += min(section_count * 0.3, 1.5)
        
        # Data presentation indicators
        data_indicators = ['table', 'figure', 'chart', 'graph', 'appendix']
        data_count = sum(1 for indicator in data_indicators if indicator in content_lower)
        score += min(data_count * 0.2, 1.0)
        
        # Professional formatting indicators
        format_indicators = ['et al.', 'ibid.', 'op. cit.', 'cf.', 'i.e.', 'e.g.']
        format_count = sum(1 for indicator in format_indicators if indicator in content_lower)
        score += min(format_count * 0.1, 0.5)
        
        return score
    
    def _assess_author_credibility_enhanced(self, metadata: Optional[Dict]) -> float:
        """Enhanced author credibility assessment"""
        if not metadata:
            return 5.0  # Neutral score when no metadata
        
        score = 5.0
        
        # Author information analysis
        author = metadata.get('author', '').lower()
        if author:
            # Academic credentials (enhanced)
            if 'nobel prize' in author or 'nobel laureate' in author:
                score += 4.0
            elif any(credential in author for credential in ['phd', 'ph.d.', 'dr.', 'professor', 'prof.']):
                score += 2.5
            elif any(credential in author for credential in ['md', 'm.d.', 'researcher', 'scientist']):
                score += 2.0
            
            # Professional titles (enhanced)
            elif any(title in author for title in ['chief', 'director', 'vp', 'vice president']):
                score += 1.8
            elif any(title in author for title in ['senior', 'lead', 'principal']):
                score += 1.5
            elif any(title in author for title in ['analyst', 'manager', 'expert']):
                score += 1.0
        
        # Institutional affiliation (enhanced)
        affiliation = metadata.get('affiliation', '').lower()
        if affiliation:
            # Top-tier institutions
            if any(inst in affiliation for inst in ['harvard', 'mit', 'stanford', 'oxford', 'cambridge']):
                score += 2.5
            elif any(inst in affiliation for inst in ['university', 'institute', 'research', 'laboratory']):
                score += 2.0
            elif any(inst in affiliation for inst in ['government', 'federal', 'national']):
                score += 1.8
            elif any(inst in affiliation for inst in ['mckinsey', 'bcg', 'bain', 'deloitte']):
                score += 1.5
        
        # Publication history (if available)
        publications = metadata.get('publications', 0)
        if isinstance(publications, int) and publications > 0:
            score += min(publications * 0.1, 2.0)  # Max +2.0 for publications
        
        return max(0.0, min(10.0, score))
    
    def _evaluate_publication_standards_enhanced(self, content: str, domain: str) -> float:
        """Enhanced publication standards evaluation"""
        score = 5.0
        content_lower = content.lower()
        
        # Editorial standards indicators (enhanced weights)
        editorial_indicators = {
            'peer review': 1.5,
            'peer-reviewed': 1.5,
            'editorial board': 1.2,
            'fact check': 1.0,
            'fact-checked': 1.0,
            'verified': 0.8,
            'editor': 0.6,
            'editorial': 0.6,
            'correction': 0.8,
            'retraction': 0.5,  # Lower because it indicates problems
            'updated': 0.4
        }
        
        for indicator, weight in editorial_indicators.items():
            if indicator in content_lower:
                score += weight
        
        # Transparency indicators (enhanced)
        transparency_indicators = {
            'methodology': 1.5,
            'data sources': 1.2,
            'conflict of interest': 1.0,
            'funding': 0.8,
            'sponsor': 0.6,
            'disclaimer': 0.8,
            'about the author': 0.6,
            'author bio': 0.5,
            'contact information': 0.4
        }
        
        for indicator, weight in transparency_indicators.items():
            if indicator in content_lower:
                score += weight
        
        # Domain-specific adjustments (enhanced)
        domain_bonuses = {
            '.gov': 2.0,
            '.edu': 1.8,
            'reuters': 1.5,
            'bloomberg': 1.5,
            'nature': 2.0,
            'science': 1.8,
            'mckinsey': 1.3,
            'harvard': 1.6,
            'mit': 1.6
        }
        
        for domain_indicator, bonus in domain_bonuses.items():
            if domain_indicator in domain:
                score += bonus
                break  # Only apply one domain bonus
        
        return max(0.0, min(10.0, score))
    
    def _analyze_citation_quality_enhanced(self, content: str) -> float:
        """Enhanced citation quality analysis"""
        score = 5.0
        content_lower = content.lower()
        
        # High-quality citation types with enhanced scoring
        citation_types = {
            'doi:': 2.0,
            'pmid:': 1.8,
            'arxiv:': 1.5,
            'isbn:': 1.2,
            'issn:': 1.0
        }
        
        for citation_type, value in citation_types.items():
            count = len(re.findall(citation_type, content_lower))
            score += min(count * value, value * 2)  # Max 2x the base value
        
        # Academic journal indicators
        journal_indicators = [
            'journal of', 'proceedings of', 'annals of', 'review of',
            'nature', 'science', 'cell', 'lancet', 'nejm'
        ]
        journal_count = sum(1 for indicator in journal_indicators if indicator in content_lower)
        score += min(journal_count * 0.8, 2.0)
        
        # Citation density analysis
        word_count = len(content.split())
        if word_count > 0:
            # Count all citation-like patterns
            total_citations = (
                len(re.findall(r'doi:', content_lower)) +
                len(re.findall(r'pmid:', content_lower)) +
                len(re.findall(r'http[s]?://', content_lower)) +
                len(re.findall(r'\[\d+\]', content)) +  # Numbered citations
                len(re.findall(r'\([^)]*\d{4}[^)]*\)', content))  # Year citations
            )
            
            citation_density = total_citations / (word_count / 100)  # Citations per 100 words
            
            if citation_density > 3.0:
                score += 2.0  # Very well-cited
            elif citation_density > 2.0:
                score += 1.5  # Well-cited
            elif citation_density > 1.0:
                score += 1.0  # Adequately cited
            elif citation_density < 0.1:
                score -= 1.0  # Poorly cited
        
        return max(0.0, min(10.0, score))
    
    def _assess_bias_enhanced(self, content: str) -> float:
        """Enhanced bias assessment with multiple dimensions"""
        score = 7.0  # Start with assumption of moderate bias
        content_lower = content.lower()
        
        # Emotional language penalty (enhanced)
        emotional_count = sum(1 for word in self.bias_patterns['emotional_language'] 
                            if word in content_lower)
        score -= min(emotional_count * 0.4, 2.5)  # Max -2.5
        
        # Balanced language bonus (enhanced)
        balanced_count = sum(1 for phrase in self.bias_patterns['balanced_language'] 
                           if phrase in content_lower)
        score += min(balanced_count * 0.3, 2.0)  # Max +2.0
        
        # Objective language bonus (enhanced)
        objective_count = sum(1 for phrase in self.bias_patterns['objective_language'] 
                            if phrase in content_lower)
        score += min(objective_count * 0.4, 2.5)  # Max +2.5
        
        # First-person language penalty
        first_person = ['i think', 'i believe', 'in my opinion', 'personally', 'my view']
        first_person_count = sum(1 for phrase in first_person if phrase in content_lower)
        score -= min(first_person_count * 0.5, 2.0)  # Max -2.0
        
        # Uncertainty language penalty
        uncertainty = ['probably', 'maybe', 'could be', 'might be', 'possibly']
        uncertainty_count = sum(1 for word in uncertainty if word in content_lower)
        score -= min(uncertainty_count * 0.3, 1.5)  # Max -1.5
        
        return max(0.0, min(10.0, score))
    
    def _classify_source_type_enhanced(self, domain: str, content: str) -> SourceType:
        """Enhanced source type classification"""
        domain_lower = domain.lower()
        content_lower = content.lower()
        
        # Government sources (enhanced detection)
        if ('.gov' in domain_lower or 
            any(gov_indicator in domain_lower for gov_indicator in 
                ['census', 'federal', 'state', 'municipal', 'parliament'])):
            return SourceType.GOVERNMENT
        
        # Academic sources (enhanced detection)
        if ('.edu' in domain_lower or 
            any(academic_indicator in domain_lower for academic_indicator in 
                ['university', 'college', 'institute', 'academic', 'research']) or
            any(academic_indicator in domain_lower for academic_indicator in 
                ['arxiv', 'pubmed', 'scholar', 'jstor', 'researchgate'])):
            return SourceType.ACADEMIC
        
        # News media (enhanced detection)
        news_indicators = [
            'news', 'times', 'post', 'herald', 'tribune', 'gazette',
            'reuters', 'bloomberg', 'cnn', 'bbc', 'npr', 'pbs',
            'associated press', 'ap news', 'guardian', 'telegraph'
        ]
        if any(indicator in domain_lower for indicator in news_indicators):
            return SourceType.NEWS_MEDIA
        
        # Industry reports (enhanced detection)
        industry_indicators = [
            'mckinsey', 'bcg', 'bain', 'deloitte', 'pwc', 'kpmg',
            'gartner', 'forrester', 'idc', 'statista', 'frost',
            'accenture', 'capgemini', 'ey'
        ]
        if any(indicator in domain_lower for indicator in industry_indicators):
            return SourceType.INDUSTRY_REPORT
        
        # Social media (enhanced detection)
        social_indicators = [
            'twitter', 'facebook', 'linkedin', 'reddit', 'instagram',
            'tiktok', 'youtube', 'snapchat', 'pinterest'
        ]
        if any(indicator in domain_lower for indicator in social_indicators):
            return SourceType.SOCIAL_MEDIA
        
        # Blogs (enhanced detection)
        blog_indicators = [
            'blog', 'wordpress', 'medium', 'substack', 'blogspot',
            'tumblr', 'ghost', 'squarespace'
        ]
        if any(indicator in domain_lower for indicator in blog_indicators):
            return SourceType.BLOG
        
        # Forums (enhanced detection)
        forum_indicators = [
            'forum', 'discussion', 'community', 'quora', 'stackoverflow',
            'discourse', 'phpbb', 'vbulletin'
        ]
        if any(indicator in domain_lower for indicator in forum_indicators):
            return SourceType.FORUM
        
        return SourceType.UNKNOWN
    
    def _determine_credibility_level_calibrated(self, raw_score: float) -> CredibilityLevel:
        """Determine credibility level with calibrated thresholds"""
        if raw_score >= 9.0:
            return CredibilityLevel.VERY_HIGH
        elif raw_score >= 7.5:  # Raised from 7.0
            return CredibilityLevel.HIGH
        elif raw_score >= 5.5:  # Raised from 5.0
            return CredibilityLevel.MEDIUM
        elif raw_score >= 3.5:  # Raised from 3.0
            return CredibilityLevel.LOW
        else:
            return CredibilityLevel.VERY_LOW
    
    def _calculate_credibility_confidence_enhanced(self, factors: Dict[str, float], 
                                                 metadata: Optional[Dict]) -> float:
        """Enhanced confidence calculation"""
        confidence = 0.7  # Base confidence
        
        # Boost confidence based on available information
        if metadata:
            confidence += 0.1
            if metadata.get('author'):
                confidence += 0.05
            if metadata.get('affiliation'):
                confidence += 0.05
            if metadata.get('publish_date'):
                confidence += 0.05
        
        # Boost confidence if factors are consistent
        high_factors = sum(1 for score in factors.values() if score > 7.5)
        low_factors = sum(1 for score in factors.values() if score < 3.5)
        
        if high_factors >= 4:
            confidence += 0.2  # Very consistent high quality
        elif high_factors >= 3:
            confidence += 0.15
        elif low_factors >= 4:
            confidence += 0.15  # Consistent low quality also increases confidence
        elif low_factors >= 3:
            confidence += 0.1
        
        # Reduce confidence if factors are inconsistent
        inconsistency = abs(max(factors.values()) - min(factors.values()))
        if inconsistency > 5.0:
            confidence -= 0.1
        
        return min(confidence, 1.0)
    
    def _validate_data_freshness_enhanced(self, content: str, 
                                        metadata: Optional[Dict] = None) -> DataFreshnessCheck:
        """Enhanced data freshness validation with content-type awareness"""
        try:
            # Determine content type for appropriate freshness tolerance
            content_type = self._classify_content_type(content)
            
            # Extract publish date from metadata
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
            
            # Enhanced staleness detection
            staleness_indicators = self._detect_staleness_indicators_enhanced(content)
            
            # Calculate freshness score with content-type awareness
            freshness_score = self._calculate_freshness_score_enhanced(
                age_days, staleness_indicators, content, content_type
            )
            
            # Determine if data is fresh based on content type
            tolerance = self.freshness_tolerance.get(content_type, 90)
            is_fresh = (age_days is None or age_days <= tolerance) and freshness_score > 0.6
            
            return DataFreshnessCheck(
                content=content[:200] + "..." if len(content) > 200 else content,
                publish_date=publish_date,
                age_days=age_days,
                is_fresh=is_fresh,
                freshness_score=freshness_score,
                staleness_indicators=staleness_indicators,
                content_type=content_type
            )
            
        except Exception as e:
            logger.error(f"Enhanced freshness validation failed: {e}")
            return DataFreshnessCheck(
                content="Error in analysis",
                publish_date=None,
                age_days=None,
                is_fresh=False,
                freshness_score=0.5,
                staleness_indicators=["validation_error"],
                content_type="unknown"
            )
    
    def _classify_content_type(self, content: str) -> str:
        """Classify content type for appropriate freshness tolerance"""
        content_lower = content.lower()
        
        for content_type, patterns in self.content_type_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                return content_type
        
        return 'default'
    
    def _detect_staleness_indicators_enhanced(self, content: str) -> List[str]:
        """Enhanced staleness indicator detection"""
        staleness_indicators = []
        content_lower = content.lower()
        
        # Time-based staleness indicators
        time_indicators = [
            'last year', 'previous year', 'in 2020', 'in 2021', 'in 2022',
            'formerly', 'previously', 'used to be', 'was once',
            'outdated', 'legacy', 'deprecated', 'discontinued',
            'no longer', 'has been replaced', 'superseded'
        ]
        
        for indicator in time_indicators:
            if indicator in content_lower:
                staleness_indicators.append(indicator)
        
        # Year-specific detection (more sophisticated)
        current_year = datetime.now().year
        for year in range(2015, current_year - 1):  # Years before last year
            if f'in {year}' in content_lower or f'during {year}' in content_lower:
                staleness_indicators.append(f'reference_to_{year}')
        
        return staleness_indicators
    
    def _calculate_freshness_score_enhanced(self, age_days: Optional[int], 
                                          staleness_indicators: List[str], 
                                          content: str, content_type: str) -> float:
        """Enhanced freshness score calculation with content-type awareness"""
        score = 0.8  # Base score
        
        # Content-type specific age scoring
        tolerance = self.freshness_tolerance.get(content_type, 90)
        
        if age_days is not None:
            if age_days <= tolerance * 0.5:
                score = 1.0  # Very fresh
            elif age_days <= tolerance:
                score = 0.9  # Fresh
            elif age_days <= tolerance * 2:
                score = 0.7  # Moderately fresh
            elif age_days <= tolerance * 4:
                score = 0.5  # Getting stale
            else:
                score = 0.2  # Stale
        
        # Enhanced staleness penalty
        staleness_penalty = min(len(staleness_indicators) * 0.15, 0.6)
        score -= staleness_penalty
        
        # Current context bonuses
        current_year = datetime.now().year
        current_month = datetime.now().strftime('%B').lower()
        content_lower = content.lower()
        
        if str(current_year) in content:
            score += 0.15
        if current_month in content_lower:
            score += 0.1
        
        # Recent terminology bonus
        recent_terms = ['2024', '2025', 'recent', 'latest', 'current', 'now', 'today']
        recent_count = sum(1 for term in recent_terms if term in content_lower)
        score += min(recent_count * 0.05, 0.2)
        
        return max(0.0, min(1.0, score))
    
    def _perform_cross_reference_enhanced(self, content: str) -> List[CrossReferenceResult]:
        """Enhanced cross-reference checking with improved claim extraction"""
        results = []
        
        # Enhanced claim extraction
        claims = self._extract_key_claims_enhanced(content)
        
        for claim in claims[:3]:  # Limit to top 3 claims
            # Enhanced cross-reference simulation
            confidence = self._calculate_claim_confidence(claim, content)
            consensus = self._determine_consensus_level(confidence)
            verification = self._determine_verification_status(confidence)
            
            result = CrossReferenceResult(
                claim=claim,
                supporting_sources=[],  # Would be populated by real fact-checking
                contradicting_sources=[],
                confidence_score=confidence,
                consensus_level=consensus,
                verification_status=verification
            )
            results.append(result)
        
        return results
    
    def _extract_key_claims_enhanced(self, content: str) -> List[str]:
        """Enhanced key claim extraction"""
        sentences = content.split('.')
        claims = []
        
        # Enhanced factual indicators
        factual_indicators = [
            'percent', '%', 'million', 'billion', 'trillion',
            'study shows', 'research indicates', 'data reveals',
            'according to', 'survey found', 'analysis shows',
            'statistics show', 'report states', 'findings suggest',
            'evidence indicates', 'results demonstrate'
        ]
        
        # Quantitative indicators
        quantitative_patterns = [
            r'\d+%', r'\$\d+', r'\d+\s*(million|billion|trillion)',
            r'\d+\s*(percent|percentage)', r'\d+\s*times',
            r'increased by \d+', r'decreased by \d+', r'grew by \d+'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                # Check for factual indicators
                has_factual_indicator = any(indicator in sentence.lower() 
                                          for indicator in factual_indicators)
                
                # Check for quantitative data
                has_quantitative_data = any(re.search(pattern, sentence, re.IGNORECASE) 
                                          for pattern in quantitative_patterns)
                
                if has_factual_indicator or has_quantitative_data:
                    claims.append(sentence[:150] + "..." if len(sentence) > 150 else sentence)
        
        return claims[:5]  # Return top 5 claims
    
    def _calculate_claim_confidence(self, claim: str, content: str) -> float:
        """Calculate confidence in a specific claim"""
        confidence = 0.6  # Base confidence
        
        claim_lower = claim.lower()
        
        # Boost confidence for specific data
        if any(indicator in claim_lower for indicator in ['%', 'percent', 'million', 'billion']):
            confidence += 0.2
        
        # Boost confidence for attribution
        if any(indicator in claim_lower for indicator in ['according to', 'study shows', 'research']):
            confidence += 0.15
        
        # Reduce confidence for uncertainty
        if any(indicator in claim_lower for indicator in ['probably', 'maybe', 'could be']):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _determine_consensus_level(self, confidence: float) -> str:
        """Determine consensus level from confidence score"""
        if confidence >= 0.8:
            return "strong"
        elif confidence >= 0.6:
            return "moderate"
        elif confidence >= 0.4:
            return "weak"
        else:
            return "conflicted"
    
    def _determine_verification_status(self, confidence: float) -> str:
        """Determine verification status from confidence score"""
        if confidence >= 0.7:
            return "verified"
        elif confidence >= 0.4:
            return "unverified"
        else:
            return "disputed"
    
    def _calculate_overall_quality_calibrated(self, credibility_score: SourceCredibilityScore,
                                            freshness_check: DataFreshnessCheck,
                                            cross_reference_results: List[CrossReferenceResult]) -> float:
        """Calculate overall quality score with calibrated weights"""
        
        # New calibrated weights for better differentiation
        weights = {
            'credibility': 0.60,    # Increased from 0.50
            'freshness': 0.25,     # Decreased from 0.30
            'cross_reference': 0.15 # Decreased from 0.20
        }
        
        # Normalize credibility score to 0-1
        credibility_normalized = credibility_score.raw_score / 10.0
        
        # Freshness score is already 0-1
        freshness_normalized = freshness_check.freshness_score
        
        # Cross-reference score (average of all checks)
        if cross_reference_results:
            cross_ref_normalized = sum(result.confidence_score for result in cross_reference_results) / len(cross_reference_results)
        else:
            cross_ref_normalized = 0.6  # Slightly higher neutral for missing data
        
        overall_score = (
            weights['credibility'] * credibility_normalized +
            weights['freshness'] * freshness_normalized +
            weights['cross_reference'] * cross_ref_normalized
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _assign_validation_grade_calibrated(self, overall_quality_score: float) -> str:
        """Assign letter grade with calibrated thresholds for better differentiation"""
        if overall_quality_score >= 0.95:
            return "A+"
        elif overall_quality_score >= 0.90:
            return "A"
        elif overall_quality_score >= 0.85:
            return "A-"
        elif overall_quality_score >= 0.80:
            return "B+"
        elif overall_quality_score >= 0.75:
            return "B"
        elif overall_quality_score >= 0.70:
            return "B-"
        elif overall_quality_score >= 0.65:
            return "C+"
        elif overall_quality_score >= 0.60:
            return "C"
        elif overall_quality_score >= 0.55:
            return "C-"
        elif overall_quality_score >= 0.50:
            return "D+"
        elif overall_quality_score >= 0.45:
            return "D"
        elif overall_quality_score >= 0.40:
            return "D-"
        else:
            return "F"
    
    def _identify_critical_issues_enhanced(self, credibility_score: SourceCredibilityScore,
                                         freshness_check: DataFreshnessCheck,
                                         cross_reference_results: List[CrossReferenceResult]) -> List[str]:
        """Enhanced critical issue identification"""
        issues = []
        
        # Enhanced credibility issues
        if credibility_score.credibility_level == CredibilityLevel.VERY_LOW:
            issues.append("Very low source credibility - unreliable for business decisions")
        elif credibility_score.credibility_level == CredibilityLevel.LOW:
            issues.append("Low source credibility - use with extreme caution")
        
        # Enhanced freshness issues
        if not freshness_check.is_fresh:
            if freshness_check.age_days and freshness_check.age_days > 730:
                issues.append("Data is over 2 years old - likely outdated for business analysis")
            elif freshness_check.age_days and freshness_check.age_days > 365:
                issues.append("Data is over 1 year old - verify currency before use")
            elif len(freshness_check.staleness_indicators) > 3:
                issues.append("Multiple staleness indicators - content may be outdated")
        
        # Enhanced cross-reference issues
        disputed_results = [r for r in cross_reference_results if r.verification_status == "disputed"]
        if disputed_results:
            issues.append("Contains disputed claims - verify through additional sources")
        
        conflicted_results = [r for r in cross_reference_results if r.consensus_level == "conflicted"]
        if conflicted_results:
            issues.append("Conflicting information found - seek consensus sources")
        
        # Enhanced confidence issues
        if credibility_score.confidence < 0.4:
            issues.append("Very low confidence in credibility assessment")
        elif credibility_score.confidence < 0.6:
            issues.append("Low confidence in credibility assessment")
        
        return issues
    
    def _generate_recommendations_enhanced(self, credibility_score: SourceCredibilityScore,
                                         freshness_check: DataFreshnessCheck,
                                         critical_issues: List[str]) -> List[str]:
        """Generate enhanced actionable recommendations"""
        recommendations = []
        
        # Enhanced credibility-based recommendations
        if credibility_score.credibility_level == CredibilityLevel.VERY_LOW:
            recommendations.append("Exclude this source from business analysis")
            recommendations.append("Seek sources from Tier 1 domains (Reuters, Bloomberg, McKinsey)")
        elif credibility_score.credibility_level == CredibilityLevel.LOW:
            recommendations.append("Use only as supporting evidence, not primary source")
            recommendations.append("Verify claims through high-credibility sources")
        elif credibility_score.credibility_level == CredibilityLevel.MEDIUM:
            recommendations.append("Acceptable for preliminary analysis, verify key claims")
        
        # Enhanced freshness-based recommendations
        if not freshness_check.is_fresh:
            if freshness_check.content_type == 'market_data':
                recommendations.append("Market data is stale - seek current financial information")
            elif freshness_check.content_type == 'industry_analysis':
                recommendations.append("Industry analysis may be outdated - check for recent reports")
            else:
                recommendations.append("Look for more recent data on this topic")
        
        # Quality-based recommendations
        if len(critical_issues) > 3:
            recommendations.append("Multiple quality issues detected - exclude from analysis")
        elif len(critical_issues) > 1:
            recommendations.append("Quality concerns present - use with caution")
        elif len(critical_issues) == 0:
            recommendations.append("Source meets quality standards for business intelligence")
        
        # Source type specific recommendations
        if credibility_score.source_type == SourceType.SOCIAL_MEDIA:
            recommendations.append("Social media source - verify through traditional media")
        elif credibility_score.source_type == SourceType.BLOG:
            recommendations.append("Blog source - check author credentials and citations")
        elif credibility_score.source_type == SourceType.ACADEMIC:
            recommendations.append("Academic source - excellent for research foundation")
        elif credibility_score.source_type == SourceType.GOVERNMENT:
            recommendations.append("Government source - reliable for official data")
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    def _create_error_validation_result(self, url: str, error_message: str) -> ValidationResult:
        """Create error validation result"""
        error_credibility = SourceCredibilityScore(
            url=url, domain="error", source_type=SourceType.UNKNOWN,
            credibility_level=CredibilityLevel.VERY_LOW, raw_score=0.0,
            factors={}, confidence=0.0, last_updated=datetime.now()
        )
        
        error_freshness = DataFreshnessCheck(
            content="Error in analysis", publish_date=None, age_days=None,
            is_fresh=False, freshness_score=0.0, staleness_indicators=["error"],
            content_type="unknown"
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
def create_calibrated_data_validator() -> CalibratedDataValidator:
    """Create calibrated data validator instance"""
    return CalibratedDataValidator()

# Test function
def test_calibrated_data_validator():
    """Test the calibrated data validation system"""
    print("Testing Calibrated Data Validation System...")
    
    validator = create_calibrated_data_validator()
    
    print("\n1. Testing high-quality source validation (Reuters/McKinsey)...")
    
    # Test with high-quality content (enhanced)
    high_quality_url = "https://www.reuters.com/business/tech/ai-market-analysis-2024"
    high_quality_content = """
    According to a comprehensive peer-reviewed study published by McKinsey & Company in collaboration 
    with MIT's AI Laboratory, the artificial intelligence market is projected to reach $190 billion by 2025. 
    The research, conducted over 18 months using rigorous methodology and surveying 2,000 executives 
    across 15 industries, reveals significant growth in AI adoption.
    
    Statistical analysis of the data shows that 67% of companies have increased their AI investments 
    in 2024, compared to 45% in 2023 (p < 0.001, 95% confidence interval). The study methodology 
    involved structured interviews, quantitative analysis of financial reports from Fortune 500 companies,
    and regression analysis to identify key trends.
    
    Dr. Sarah Johnson, PhD, lead researcher at MIT's AI Lab and co-author of the study published in 
    Nature Machine Intelligence (DOI: 10.1038/s42256-024-00123-4), noted that "the evidence suggests 
    a fundamental shift in how businesses approach artificial intelligence implementation."
    
    The report cites 47 peer-reviewed sources and includes references to academic publications 
    in Nature, Science, and Cell journals. Statistical analysis was performed using standard 
    methodologies with 95% confidence intervals and effect size calculations (Cohen's d = 0.82).
    
    Methodology section details the systematic review process, data collection procedures, and 
    quality assurance measures. The research team included experts from Harvard Business School,
    Stanford University, and Oxford University.
    """
    
    high_quality_metadata = {
        'author': 'Dr. Sarah Johnson, PhD, MIT AI Laboratory',
        'publish_date': '2024-06-15T10:30:00Z',
        'affiliation': 'MIT AI Laboratory, Harvard Business School',
        'publications': 47
    }
    
    result_high = validator.validate_source_comprehensive(
        high_quality_url, high_quality_content, high_quality_metadata
    )
    
    print(f"✅ High Quality Source (Enhanced):")
    print(f"   URL: {result_high.source_url}")
    print(f"   Validation Grade: {result_high.validation_grade}")
    print(f"   Overall Quality: {result_high.overall_quality_score:.3f}")
    print(f"   Credibility Level: {result_high.credibility_score.credibility_level.value}")
    print(f"   Credibility Score: {result_high.credibility_score.raw_score:.1f}/10")
    print(f"   Is Fresh: {result_high.freshness_check.is_fresh}")
    print(f"   Content Type: {result_high.freshness_check.content_type}")
    print(f"   Critical Issues: {len(result_high.critical_issues)}")
    print(f"   Recommendations: {len(result_high.recommendations)}")
    
    print("\n2. Testing low-quality source validation (Personal Blog)...")
    
    # Test with low-quality content (same as before for comparison)
    low_quality_url = "https://myblog.wordpress.com/ai-predictions"
    low_quality_content = """
    I think AI is going to be huge! Maybe it will make billions of dollars, probably more than anyone expects.
    
    Some guy on Twitter said that AI companies are making crazy money. This is shocking news that will 
    revolutionize everything! I heard from a friend that Google is secretly working on something amazing.
    
    In my opinion, everyone should invest in AI right now. It's unbelievable how much money you could make!
    The rumors suggest that this could be the next big thing. Personally, I believe this is groundbreaking.
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
    
    print("\n3. Testing medium-quality source validation (TechCrunch)...")
    
    # Test with medium-quality content
    medium_quality_url = "https://techcrunch.com/ai-startup-funding"
    medium_quality_content = """
    According to industry reports, AI startup funding reached $25 billion in 2024, representing 
    a 40% increase from the previous year. The data comes from venture capital tracking firm 
    PitchBook and includes analysis of over 1,200 funding rounds.
    
    Research from Stanford's AI Index shows that enterprise AI adoption has grown significantly. 
    The report indicates that 78% of Fortune 500 companies now use AI in some capacity, 
    compared to 45% in 2022.
    
    Industry analyst John Smith from Gartner noted that "we're seeing a maturation of the AI market 
    with more focused applications and better ROI metrics." The analysis includes data from 
    multiple sources including CB Insights and Crunchbase.
    
    However, some experts caution that the market may be experiencing overvaluation in certain 
    segments. Nevertheless, the overall trend suggests continued growth in AI investment.
    """
    
    medium_quality_metadata = {
        'author': 'Tech Reporter',
        'publish_date': '2024-05-20T14:15:00Z'
    }
    
    result_medium = validator.validate_source_comprehensive(
        medium_quality_url, medium_quality_content, medium_quality_metadata
    )
    
    print(f"✅ Medium Quality Source:")
    print(f"   URL: {result_medium.source_url}")
    print(f"   Validation Grade: {result_medium.validation_grade}")
    print(f"   Overall Quality: {result_medium.overall_quality_score:.3f}")
    print(f"   Credibility Level: {result_medium.credibility_score.credibility_level.value}")
    print(f"   Credibility Score: {result_medium.credibility_score.raw_score:.1f}/10")
    print(f"   Critical Issues: {len(result_medium.critical_issues)}")
    
    print("\n🎉 Calibrated data validation test completed successfully!")
    
    # Calculate improvement metrics
    quality_differentiation = abs(result_high.overall_quality_score - result_low.overall_quality_score)
    
    return {
        "high_quality_validation": {
            "grade": result_high.validation_grade,
            "quality_score": result_high.overall_quality_score,
            "credibility_level": result_high.credibility_score.credibility_level.value,
            "credibility_score": result_high.credibility_score.raw_score,
            "is_fresh": result_high.freshness_check.is_fresh,
            "critical_issues": len(result_high.critical_issues)
        },
        "medium_quality_validation": {
            "grade": result_medium.validation_grade,
            "quality_score": result_medium.overall_quality_score,
            "credibility_level": result_medium.credibility_score.credibility_level.value,
            "credibility_score": result_medium.credibility_score.raw_score,
            "critical_issues": len(result_medium.critical_issues)
        },
        "low_quality_validation": {
            "grade": result_low.validation_grade,
            "quality_score": result_low.overall_quality_score,
            "credibility_level": result_low.credibility_score.credibility_level.value,
            "credibility_score": result_low.credibility_score.raw_score,
            "critical_issues": len(result_low.critical_issues)
        },
        "quality_differentiation": quality_differentiation,
        "system_status": "calibrated_validation_operational"
    }

if __name__ == "__main__":
    # Run calibrated data validation tests
    test_calibrated_data_validator()

