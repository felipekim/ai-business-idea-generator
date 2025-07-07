"""
Automated Reporting System
Generates automated quality reports, insights, and weekly summaries
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of automated reports"""
    WEEKLY_SUMMARY = "weekly_summary"
    QUALITY_ASSESSMENT = "quality_assessment"
    TREND_ANALYSIS = "trend_analysis"
    PERFORMANCE_METRICS = "performance_metrics"
    OPPORTUNITY_INSIGHTS = "opportunity_insights"

class ReportFormat(Enum):
    """Report output formats"""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"

@dataclass
class ReportConfig:
    """Configuration for automated reporting"""
    enabled_reports: List[ReportType] = field(default_factory=lambda: list(ReportType))
    default_format: ReportFormat = ReportFormat.MARKDOWN
    include_charts: bool = False
    include_recommendations: bool = True
    max_ideas_in_summary: int = 10
    
    # Report scheduling
    weekly_report_day: str = "sunday"  # Day to generate weekly reports
    quality_report_threshold: int = 5  # Generate after N assessments
    
    # Content preferences
    include_financial_details: bool = True
    include_market_analysis: bool = True
    include_competitive_insights: bool = True
    include_implementation_notes: bool = True

@dataclass
class ReportData:
    """Report data structure"""
    report_id: str
    report_type: ReportType
    title: str
    content: str
    format: ReportFormat
    generated_at: datetime
    data_period: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'report_id': self.report_id,
            'report_type': self.report_type.value,
            'title': self.title,
            'content': self.content,
            'format': self.format.value,
            'generated_at': self.generated_at.isoformat(),
            'data_period': self.data_period,
            'metadata': self.metadata
        }

class AutomatedReportingSystem:
    """Automated reporting system for business intelligence"""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.report_history: List[ReportData] = []
        self.report_templates = self._initialize_templates()
        
        logger.info("Automated Reporting System initialized")
    
    def generate_weekly_summary(self, ideas: List[Dict[str, Any]], 
                              quality_metrics: List[Any],
                              period_start: datetime,
                              period_end: datetime) -> ReportData:
        """Generate weekly summary report"""
        logger.info("Generating weekly summary report")
        
        try:
            # Analyze the week's data
            total_ideas = len(ideas)
            approved_ideas = [
                idea for i, idea in enumerate(ideas)
                if i < len(quality_metrics) and quality_metrics[i].validation_result.value == "approved"
            ]
            
            avg_score = statistics.mean([qm.overall_score for qm in quality_metrics]) if quality_metrics else 0
            
            # Generate content
            content = self._generate_weekly_summary_content(
                ideas, quality_metrics, approved_ideas, avg_score, period_start, period_end
            )
            
            # Create report
            report = ReportData(
                report_id=f"weekly_summary_{period_start.strftime('%Y%m%d')}",
                report_type=ReportType.WEEKLY_SUMMARY,
                title=f"Weekly Business Ideas Summary - {period_start.strftime('%B %d, %Y')}",
                content=content,
                format=self.config.default_format,
                generated_at=datetime.utcnow(),
                data_period={
                    'start': period_start.isoformat(),
                    'end': period_end.isoformat(),
                    'total_ideas': total_ideas,
                    'approved_ideas': len(approved_ideas)
                }
            )
            
            self.report_history.append(report)
            logger.info(f"Weekly summary generated: {total_ideas} ideas, {len(approved_ideas)} approved")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating weekly summary: {e}")
            raise
    
    def generate_quality_assessment_report(self, quality_metrics: List[Any]) -> ReportData:
        """Generate quality assessment report"""
        logger.info("Generating quality assessment report")
        
        try:
            # Analyze quality data
            total_assessments = len(quality_metrics)
            approved_count = sum(1 for qm in quality_metrics if qm.validation_result.value == "approved")
            rejected_count = sum(1 for qm in quality_metrics if qm.validation_result.value == "rejected")
            
            approval_rate = (approved_count / total_assessments * 100) if total_assessments > 0 else 0
            avg_score = statistics.mean([qm.overall_score for qm in quality_metrics]) if quality_metrics else 0
            
            # Generate content
            content = self._generate_quality_assessment_content(
                quality_metrics, total_assessments, approved_count, rejected_count, approval_rate, avg_score
            )
            
            # Create report
            report = ReportData(
                report_id=f"quality_assessment_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                report_type=ReportType.QUALITY_ASSESSMENT,
                title="Quality Assessment Report",
                content=content,
                format=self.config.default_format,
                generated_at=datetime.utcnow(),
                data_period={
                    'assessments_analyzed': total_assessments,
                    'approval_rate': approval_rate,
                    'average_score': avg_score
                }
            )
            
            self.report_history.append(report)
            logger.info(f"Quality assessment report generated: {approval_rate:.1f}% approval rate")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating quality assessment report: {e}")
            raise
    
    def generate_trend_analysis_report(self, trending_topics: List[str],
                                     ideas: List[Dict[str, Any]]) -> ReportData:
        """Generate trend analysis report"""
        logger.info("Generating trend analysis report")
        
        try:
            # Analyze trends
            topic_frequency = Counter(trending_topics)
            top_trends = topic_frequency.most_common(10)
            
            # Analyze idea categories
            idea_categories = [idea.get('niche_category', 'general') for idea in ideas]
            category_frequency = Counter(idea_categories)
            
            # Generate content
            content = self._generate_trend_analysis_content(top_trends, category_frequency, ideas)
            
            # Create report
            report = ReportData(
                report_id=f"trend_analysis_{datetime.utcnow().strftime('%Y%m%d')}",
                report_type=ReportType.TREND_ANALYSIS,
                title="Market Trends & Opportunity Analysis",
                content=content,
                format=self.config.default_format,
                generated_at=datetime.utcnow(),
                data_period={
                    'trending_topics_analyzed': len(trending_topics),
                    'ideas_analyzed': len(ideas),
                    'top_trend': top_trends[0][0] if top_trends else None
                }
            )
            
            self.report_history.append(report)
            logger.info(f"Trend analysis report generated: {len(top_trends)} trends analyzed")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating trend analysis report: {e}")
            raise
    
    def generate_opportunity_insights(self, ideas: List[Dict[str, Any]],
                                    quality_metrics: List[Any]) -> ReportData:
        """Generate opportunity insights report"""
        logger.info("Generating opportunity insights report")
        
        try:
            # Find top opportunities
            top_opportunities = []
            for i, idea in enumerate(ideas):
                if i < len(quality_metrics):
                    qm = quality_metrics[i]
                    if qm.validation_result.value == "approved" and qm.overall_score >= 8.0:
                        top_opportunities.append((idea, qm))
            
            # Sort by score
            top_opportunities.sort(key=lambda x: x[1].overall_score, reverse=True)
            top_opportunities = top_opportunities[:5]  # Top 5
            
            # Generate insights
            content = self._generate_opportunity_insights_content(top_opportunities, ideas, quality_metrics)
            
            # Create report
            report = ReportData(
                report_id=f"opportunity_insights_{datetime.utcnow().strftime('%Y%m%d')}",
                report_type=ReportType.OPPORTUNITY_INSIGHTS,
                title="Top Business Opportunities & Strategic Insights",
                content=content,
                format=self.config.default_format,
                generated_at=datetime.utcnow(),
                data_period={
                    'top_opportunities_count': len(top_opportunities),
                    'total_ideas_analyzed': len(ideas)
                }
            )
            
            self.report_history.append(report)
            logger.info(f"Opportunity insights generated: {len(top_opportunities)} top opportunities")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating opportunity insights: {e}")
            raise
    
    def _generate_weekly_summary_content(self, ideas: List[Dict[str, Any]],
                                       quality_metrics: List[Any],
                                       approved_ideas: List[Dict[str, Any]],
                                       avg_score: float,
                                       period_start: datetime,
                                       period_end: datetime) -> str:
        """Generate weekly summary content"""
        
        content = f"""# Weekly Business Ideas Summary
**Period:** {period_start.strftime('%B %d, %Y')} - {period_end.strftime('%B %d, %Y')}
**Generated:** {datetime.utcnow().strftime('%B %d, %Y at %I:%M %p')}

## 📊 Executive Summary

- **Total Ideas Generated:** {len(ideas)}
- **Ideas Approved:** {len(approved_ideas)}
- **Average Quality Score:** {avg_score:.1f}/10
- **Approval Rate:** {(len(approved_ideas)/len(ideas)*100):.1f}%

## 🎯 Top Approved Ideas

"""
        
        # Add top approved ideas
        for i, idea in enumerate(approved_ideas[:5], 1):
            qm = quality_metrics[ideas.index(idea)] if idea in ideas else None
            score = qm.overall_score if qm else 0
            
            content += f"""### {i}. {idea.get('name', 'Untitled Idea')}
**Score:** {score:.1f}/10 | **Category:** {idea.get('niche_category', 'General')}

**Problem:** {idea.get('problem_statement', 'Not specified')[:150]}...

**Solution:** {idea.get('ai_solution', 'Not specified')[:150]}...

**Market Opportunity:** {self._extract_market_size(idea)}

---

"""
        
        # Add quality insights
        if quality_metrics:
            quality_levels = [qm.quality_level.value for qm in quality_metrics]
            quality_counter = Counter(quality_levels)
            
            content += f"""## 📈 Quality Analysis

**Quality Distribution:**
"""
            for level, count in quality_counter.most_common():
                percentage = (count / len(quality_metrics) * 100)
                content += f"- {level.title()}: {count} ideas ({percentage:.1f}%)\n"
        
        # Add recommendations
        content += f"""

## 🎯 Recommendations

"""
        
        if len(approved_ideas) >= 5:
            content += "- **Excellent week!** Multiple high-quality opportunities identified.\n"
        elif len(approved_ideas) >= 3:
            content += "- **Good progress** with several viable opportunities.\n"
        else:
            content += "- **Focus needed** on improving idea quality and market validation.\n"
        
        if avg_score >= 8.0:
            content += "- **High quality standards** maintained across generated ideas.\n"
        elif avg_score >= 7.0:
            content += "- **Quality is good** but there's room for improvement.\n"
        else:
            content += "- **Quality improvement needed** - consider refining generation criteria.\n"
        
        content += """
## 📅 Next Week Focus

- Continue monitoring trending opportunities
- Enhance market validation for emerging sectors
- Optimize idea generation based on this week's insights

---
*This report was automatically generated by the AI Business Intelligence Pipeline*
"""
        
        return content
    
    def _generate_quality_assessment_content(self, quality_metrics: List[Any],
                                           total_assessments: int,
                                           approved_count: int,
                                           rejected_count: int,
                                           approval_rate: float,
                                           avg_score: float) -> str:
        """Generate quality assessment content"""
        
        content = f"""# Quality Assessment Report
**Generated:** {datetime.utcnow().strftime('%B %d, %Y at %I:%M %p')}

## 📊 Quality Metrics Overview

- **Total Assessments:** {total_assessments}
- **Approved Ideas:** {approved_count}
- **Rejected Ideas:** {rejected_count}
- **Approval Rate:** {approval_rate:.1f}%
- **Average Score:** {avg_score:.2f}/10

## 🎯 Quality Gate Performance

"""
        
        # Analyze gate performance
        if quality_metrics:
            gate_scores = defaultdict(list)
            for qm in quality_metrics:
                for gate, score in qm.gate_scores.items():
                    gate_scores[gate.value].append(score)
            
            for gate, scores in gate_scores.items():
                avg_gate_score = statistics.mean(scores)
                content += f"- **{gate.replace('_', ' ').title()}:** {avg_gate_score:.1f}/10\n"
        
        # Common issues
        content += f"""

## ⚠️ Common Quality Issues

"""
        
        if quality_metrics:
            all_issues = []
            for qm in quality_metrics:
                all_issues.extend(qm.quality_issues)
            
            issue_counter = Counter(all_issues)
            for issue, count in issue_counter.most_common(5):
                content += f"- {issue} ({count} occurrences)\n"
        
        # Recommendations
        content += f"""

## 🎯 Quality Improvement Recommendations

"""
        
        if approval_rate >= 80:
            content += "- **Excellent quality standards** - maintain current processes\n"
        elif approval_rate >= 60:
            content += "- **Good quality** - focus on addressing common issues\n"
        else:
            content += "- **Quality improvement needed** - review generation and validation processes\n"
        
        if avg_score >= 8.0:
            content += "- **High scoring ideas** - consider raising quality thresholds\n"
        elif avg_score >= 7.0:
            content += "- **Solid performance** - optimize weak quality gates\n"
        else:
            content += "- **Score improvement needed** - enhance research and validation depth\n"
        
        content += """
---
*This quality assessment was automatically generated*
"""
        
        return content
    
    def _generate_trend_analysis_content(self, top_trends: List[Tuple[str, int]],
                                       category_frequency: Counter,
                                       ideas: List[Dict[str, Any]]) -> str:
        """Generate trend analysis content"""
        
        content = f"""# Market Trends & Opportunity Analysis
**Generated:** {datetime.utcnow().strftime('%B %d, %Y at %I:%M %p')}

## 🔥 Top Trending Topics

"""
        
        for i, (trend, frequency) in enumerate(top_trends[:10], 1):
            content += f"{i}. **{trend}** ({frequency} mentions)\n"
        
        content += f"""

## 📈 Idea Categories Distribution

"""
        
        for category, count in category_frequency.most_common():
            percentage = (count / len(ideas) * 100) if ideas else 0
            content += f"- **{category.title()}:** {count} ideas ({percentage:.1f}%)\n"
        
        content += f"""

## 🎯 Market Opportunity Insights

"""
        
        # Analyze market opportunities
        if ideas:
            high_market_ideas = []
            for idea in ideas:
                market_research = idea.get('market_research', {})
                market_size = market_research.get('market_size', {}).get('total_addressable_market', 0)
                if market_size > 10000000:  # $10M+
                    high_market_ideas.append((idea.get('name', 'Unknown'), market_size))
            
            if high_market_ideas:
                content += "**Large Market Opportunities (>$10M TAM):**\n"
                for name, size in sorted(high_market_ideas, key=lambda x: x[1], reverse=True)[:5]:
                    content += f"- {name}: ${size:,.0f}\n"
        
        content += f"""

## 🚀 Strategic Recommendations

- **Emerging Trends:** Focus on {top_trends[0][0] if top_trends else 'AI automation'} opportunities
- **Market Focus:** Prioritize {category_frequency.most_common(1)[0][0] if category_frequency else 'technology'} sector development
- **Opportunity Size:** Target large addressable markets for maximum impact

---
*This trend analysis was automatically generated*
"""
        
        return content
    
    def _generate_opportunity_insights_content(self, top_opportunities: List[Tuple[Dict[str, Any], Any]],
                                             ideas: List[Dict[str, Any]],
                                             quality_metrics: List[Any]) -> str:
        """Generate opportunity insights content"""
        
        content = f"""# Top Business Opportunities & Strategic Insights
**Generated:** {datetime.utcnow().strftime('%B %d, %Y at %I:%M %p')}

## 🏆 Top 5 Investment-Grade Opportunities

"""
        
        for i, (idea, qm) in enumerate(top_opportunities, 1):
            financial_analysis = idea.get('financial_analysis', {})
            revenue_projections = financial_analysis.get('revenue_projections', {})
            year_1_revenue = revenue_projections.get('revenue_by_year', {}).get('year_1', {}).get('annual_revenue', 0)
            
            content += f"""### {i}. {idea.get('name', 'Untitled Opportunity')}
**Quality Score:** {qm.overall_score:.1f}/10 | **Investment Grade:** {qm.quality_level.value.title()}

**Market Opportunity:** {self._extract_market_size(idea)}
**Year 1 Revenue Potential:** ${year_1_revenue:,.0f}
**Implementation Complexity:** {self._assess_complexity(idea)}

**Key Strengths:**
"""
            
            # Add strengths based on high gate scores
            for gate, score in qm.gate_scores.items():
                if score >= 8.0:
                    content += f"- {gate.value.replace('_', ' ').title()}: {score:.1f}/10\n"
            
            content += "\n---\n\n"
        
        # Strategic insights
        content += f"""## 🎯 Strategic Investment Insights

"""
        
        if top_opportunities:
            avg_top_score = statistics.mean([qm.overall_score for _, qm in top_opportunities])
            content += f"- **Portfolio Quality:** Average score of top opportunities is {avg_top_score:.1f}/10\n"
            
            # Analyze sectors
            sectors = [idea.get('niche_category', 'general') for idea, _ in top_opportunities]
            sector_counter = Counter(sectors)
            top_sector = sector_counter.most_common(1)[0][0] if sector_counter else 'technology'
            content += f"- **Sector Focus:** {top_sector.title()} shows strongest opportunities\n"
            
            # Financial potential
            total_revenue_potential = sum([
                idea.get('financial_analysis', {}).get('revenue_projections', {})
                .get('revenue_by_year', {}).get('year_1', {}).get('annual_revenue', 0)
                for idea, _ in top_opportunities
            ])
            content += f"- **Revenue Potential:** Combined Year 1 potential of ${total_revenue_potential:,.0f}\n"
        
        content += f"""

## 📊 Portfolio Recommendations

"""
        
        if len(top_opportunities) >= 3:
            content += "- **Diversification:** Strong portfolio with multiple high-quality opportunities\n"
            content += "- **Investment Ready:** Multiple ideas meet investment-grade criteria\n"
        elif len(top_opportunities) >= 1:
            content += "- **Selective Focus:** Concentrate on top-scoring opportunities\n"
            content += "- **Quality over Quantity:** Maintain high standards for opportunity selection\n"
        else:
            content += "- **Pipeline Development:** Focus on improving idea generation quality\n"
            content += "- **Market Research:** Enhance validation processes for better opportunities\n"
        
        content += """
---
*This opportunity analysis was automatically generated*
"""
        
        return content
    
    def _extract_market_size(self, idea: Dict[str, Any]) -> str:
        """Extract and format market size information"""
        market_research = idea.get('market_research', {})
        market_size = market_research.get('market_size', {})
        tam = market_size.get('total_addressable_market', 0)
        
        if tam >= 1000000000:  # $1B+
            return f"${tam/1000000000:.1f}B TAM"
        elif tam >= 1000000:  # $1M+
            return f"${tam/1000000:.0f}M TAM"
        else:
            return "Market size not specified"
    
    def _assess_complexity(self, idea: Dict[str, Any]) -> str:
        """Assess implementation complexity"""
        implementation_plan = idea.get('implementation_plan', {})
        phases = len([key for key in implementation_plan.keys() if 'phase' in key.lower()])
        
        if phases <= 2:
            return "Low complexity"
        elif phases <= 4:
            return "Medium complexity"
        else:
            return "High complexity"
    
    def _initialize_templates(self) -> Dict[ReportType, str]:
        """Initialize report templates"""
        return {
            ReportType.WEEKLY_SUMMARY: "weekly_summary_template",
            ReportType.QUALITY_ASSESSMENT: "quality_assessment_template",
            ReportType.TREND_ANALYSIS: "trend_analysis_template",
            ReportType.OPPORTUNITY_INSIGHTS: "opportunity_insights_template"
        }
    
    def get_report_history(self, report_type: Optional[ReportType] = None,
                          days: int = 30) -> List[ReportData]:
        """Get report history"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        filtered_reports = [
            report for report in self.report_history
            if report.generated_at >= cutoff_date
        ]
        
        if report_type:
            filtered_reports = [
                report for report in filtered_reports
                if report.report_type == report_type
            ]
        
        return sorted(filtered_reports, key=lambda r: r.generated_at, reverse=True)
    
    def export_report(self, report: ReportData, file_path: str) -> bool:
        """Export report to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if report.format == ReportFormat.JSON:
                    json.dump(report.to_dict(), f, indent=2)
                else:
                    f.write(report.content)
            
            logger.info(f"Report exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return False

