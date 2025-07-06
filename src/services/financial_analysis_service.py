import os
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import math

class FinancialAnalysisService:
    def __init__(self):
        # Cost databases for different tools and services
        self.no_code_tools = {
            "zapier": {"monthly": 29, "annual": 290, "category": "automation"},
            "bubble": {"monthly": 29, "annual": 290, "category": "app_development"},
            "webflow": {"monthly": 23, "annual": 230, "category": "website"},
            "airtable": {"monthly": 20, "annual": 200, "category": "database"},
            "notion": {"monthly": 10, "annual": 100, "category": "workspace"},
            "typeform": {"monthly": 35, "annual": 350, "category": "forms"},
            "calendly": {"monthly": 10, "annual": 100, "category": "scheduling"},
            "mailchimp": {"monthly": 20, "annual": 200, "category": "email_marketing"},
            "stripe": {"transaction_fee": 0.029, "category": "payments"},
            "twilio": {"per_message": 0.0075, "category": "communications"},
            "sendgrid": {"monthly": 15, "annual": 150, "category": "email_service"}
        }
        
        self.ai_platforms = {
            "openai_api": {"per_1k_tokens": 0.002, "category": "language_model"},
            "anthropic": {"per_1k_tokens": 0.008, "category": "language_model"},
            "huggingface": {"monthly": 9, "annual": 90, "category": "ml_platform"},
            "replicate": {"per_prediction": 0.0023, "category": "ml_inference"},
            "elevenlabs": {"monthly": 22, "annual": 220, "category": "voice_ai"},
            "stability_ai": {"per_image": 0.02, "category": "image_generation"},
            "pinecone": {"monthly": 70, "annual": 700, "category": "vector_database"},
            "langchain": {"monthly": 0, "annual": 0, "category": "framework"}
        }
        
        self.business_services = {
            "domain_registration": {"annual": 15, "category": "infrastructure"},
            "hosting_basic": {"monthly": 10, "annual": 100, "category": "infrastructure"},
            "hosting_premium": {"monthly": 50, "annual": 500, "category": "infrastructure"},
            "ssl_certificate": {"annual": 50, "category": "security"},
            "business_registration": {"one_time": 500, "category": "legal"},
            "trademark": {"one_time": 1000, "category": "legal"},
            "business_insurance": {"annual": 500, "category": "legal"},
            "accounting_software": {"monthly": 30, "annual": 300, "category": "operations"},
            "google_workspace": {"monthly": 12, "annual": 120, "category": "productivity"},
            "design_tools": {"monthly": 20, "annual": 200, "category": "design"}
        }
        
        self.marketing_costs = {
            "google_ads": {"daily_budget": 50, "category": "paid_advertising"},
            "facebook_ads": {"daily_budget": 30, "category": "paid_advertising"},
            "content_creation": {"monthly": 500, "category": "content"},
            "seo_tools": {"monthly": 100, "annual": 1000, "category": "seo"},
            "social_media_management": {"monthly": 50, "annual": 500, "category": "social"},
            "email_marketing": {"monthly": 50, "annual": 500, "category": "email"},
            "influencer_marketing": {"campaign": 1000, "category": "influencer"},
            "pr_services": {"monthly": 2000, "category": "pr"}
        }
        
        # Industry benchmarks
        self.industry_benchmarks = {
            "saas": {
                "customer_acquisition_cost": {"min": 100, "max": 500, "avg": 250},
                "lifetime_value": {"min": 500, "max": 5000, "avg": 1500},
                "churn_rate_monthly": {"min": 0.02, "max": 0.10, "avg": 0.05},
                "gross_margin": {"min": 0.70, "max": 0.90, "avg": 0.80},
                "revenue_growth_rate": {"min": 0.20, "max": 1.00, "avg": 0.50}
            },
            "marketplace": {
                "customer_acquisition_cost": {"min": 50, "max": 200, "avg": 100},
                "lifetime_value": {"min": 200, "max": 2000, "avg": 800},
                "churn_rate_monthly": {"min": 0.05, "max": 0.15, "avg": 0.08},
                "gross_margin": {"min": 0.15, "max": 0.30, "avg": 0.20},
                "revenue_growth_rate": {"min": 0.30, "max": 1.50, "avg": 0.70}
            },
            "ecommerce": {
                "customer_acquisition_cost": {"min": 30, "max": 150, "avg": 75},
                "lifetime_value": {"min": 100, "max": 1000, "avg": 300},
                "churn_rate_monthly": {"min": 0.10, "max": 0.25, "avg": 0.15},
                "gross_margin": {"min": 0.20, "max": 0.50, "avg": 0.35},
                "revenue_growth_rate": {"min": 0.15, "max": 0.80, "avg": 0.40}
            },
            "consulting": {
                "customer_acquisition_cost": {"min": 200, "max": 1000, "avg": 500},
                "lifetime_value": {"min": 2000, "max": 20000, "avg": 8000},
                "churn_rate_monthly": {"min": 0.01, "max": 0.05, "avg": 0.02},
                "gross_margin": {"min": 0.60, "max": 0.85, "avg": 0.75},
                "revenue_growth_rate": {"min": 0.10, "max": 0.50, "avg": 0.25}
            }
        }
    
    def analyze_financial_projections(self, business_idea: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive financial analysis for a business idea"""
        try:
            # Determine business model and industry
            business_model = self._determine_business_model(business_idea)
            industry = self._determine_industry(business_idea)
            
            # Calculate detailed cost breakdown
            cost_breakdown = self._calculate_detailed_costs(business_idea)
            
            # Generate revenue projections
            revenue_projections = self._generate_revenue_projections(business_idea, business_model, industry)
            
            # Calculate unit economics
            unit_economics = self._calculate_unit_economics(business_idea, business_model, industry)
            
            # Perform sensitivity analysis
            sensitivity_analysis = self._perform_sensitivity_analysis(revenue_projections, cost_breakdown)
            
            # Calculate key financial metrics
            financial_metrics = self._calculate_financial_metrics(revenue_projections, cost_breakdown, unit_economics)
            
            # Generate funding requirements
            funding_analysis = self._analyze_funding_requirements(cost_breakdown, revenue_projections)
            
            # Risk assessment
            risk_assessment = self._assess_financial_risks(business_idea, financial_metrics)
            
            return {
                "business_model": business_model,
                "industry_category": industry,
                "cost_breakdown": cost_breakdown,
                "revenue_projections": revenue_projections,
                "unit_economics": unit_economics,
                "financial_metrics": financial_metrics,
                "sensitivity_analysis": sensitivity_analysis,
                "funding_analysis": funding_analysis,
                "risk_assessment": risk_assessment,
                "break_even_analysis": self._calculate_break_even(revenue_projections, cost_breakdown),
                "roi_analysis": self._calculate_roi_analysis(revenue_projections, cost_breakdown),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Error in financial analysis: {e}")
            return self._get_fallback_financial_analysis()
    
    def _determine_business_model(self, business_idea: Dict[str, Any]) -> str:
        """Determine the business model from the idea description"""
        revenue_model = business_idea.get('revenue_model', {})
        primary_revenue = str(revenue_model.get('primary', '')).lower()
        
        if 'subscription' in primary_revenue or 'saas' in primary_revenue:
            return 'saas'
        elif 'marketplace' in primary_revenue or 'commission' in primary_revenue:
            return 'marketplace'
        elif 'ecommerce' in primary_revenue or 'product' in primary_revenue:
            return 'ecommerce'
        elif 'consulting' in primary_revenue or 'service' in primary_revenue:
            return 'consulting'
        elif 'advertising' in primary_revenue or 'freemium' in primary_revenue:
            return 'advertising'
        else:
            return 'saas'  # Default to SaaS for AI businesses
    
    def _determine_industry(self, business_idea: Dict[str, Any]) -> str:
        """Determine the industry category"""
        niche = business_idea.get('niche_category', '').lower()
        target_audience = business_idea.get('target_audience', '').lower()
        
        if 'healthcare' in niche or 'medical' in niche:
            return 'healthcare'
        elif 'education' in niche or 'learning' in niche:
            return 'education'
        elif 'finance' in niche or 'fintech' in niche:
            return 'fintech'
        elif 'real estate' in niche:
            return 'real_estate'
        elif 'ecommerce' in niche or 'retail' in niche:
            return 'ecommerce'
        elif 'enterprise' in target_audience or 'business' in target_audience:
            return 'b2b'
        else:
            return 'general'
    
    def _calculate_detailed_costs(self, business_idea: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed cost breakdown"""
        implementation_plan = business_idea.get('implementation_plan', {})
        no_code_tools = implementation_plan.get('no_code_tools', [])
        ai_platforms = implementation_plan.get('ai_platforms', [])
        
        # Calculate tool costs
        tool_costs = self._calculate_tool_costs(no_code_tools, ai_platforms)
        
        # Calculate business setup costs
        setup_costs = self._calculate_setup_costs()
        
        # Calculate marketing costs
        marketing_costs = self._calculate_marketing_costs(business_idea)
        
        # Calculate operational costs
        operational_costs = self._calculate_operational_costs(business_idea)
        
        # Calculate total launch costs
        total_launch_costs = (
            tool_costs['annual_total'] +
            setup_costs['total'] +
            marketing_costs['launch_budget'] +
            operational_costs['setup_costs']
        )
        
        # Calculate ongoing monthly costs
        monthly_costs = (
            tool_costs['monthly_total'] +
            marketing_costs['monthly_budget'] +
            operational_costs['monthly_costs']
        )
        
        return {
            "launch_costs": {
                "tools_and_platforms": tool_costs['annual_total'],
                "business_setup": setup_costs['total'],
                "initial_marketing": marketing_costs['launch_budget'],
                "operational_setup": operational_costs['setup_costs'],
                "contingency": total_launch_costs * 0.20,  # 20% contingency
                "total": total_launch_costs * 1.20
            },
            "monthly_costs": {
                "tools_and_platforms": tool_costs['monthly_total'],
                "marketing": marketing_costs['monthly_budget'],
                "operations": operational_costs['monthly_costs'],
                "total": monthly_costs
            },
            "annual_costs": {
                "tools_and_platforms": tool_costs['annual_total'],
                "marketing": marketing_costs['monthly_budget'] * 12,
                "operations": operational_costs['monthly_costs'] * 12,
                "total": (tool_costs['annual_total'] + 
                         marketing_costs['monthly_budget'] * 12 + 
                         operational_costs['monthly_costs'] * 12)
            },
            "cost_breakdown_details": {
                "tools": tool_costs,
                "setup": setup_costs,
                "marketing": marketing_costs,
                "operations": operational_costs
            }
        }
    
    def _calculate_tool_costs(self, no_code_tools: List[str], ai_platforms: List[str]) -> Dict[str, Any]:
        """Calculate costs for no-code tools and AI platforms"""
        monthly_total = 0
        annual_total = 0
        tool_details = {}
        
        # No-code tools
        for tool in no_code_tools:
            tool_name = tool.lower().replace(' ', '_')
            if tool_name in self.no_code_tools:
                cost_data = self.no_code_tools[tool_name]
                monthly_cost = cost_data.get('monthly', 0)
                annual_cost = cost_data.get('annual', monthly_cost * 12)
                
                monthly_total += monthly_cost
                annual_total += annual_cost
                
                tool_details[tool] = {
                    "monthly": monthly_cost,
                    "annual": annual_cost,
                    "category": cost_data.get('category', 'general')
                }
        
        # AI platforms
        for platform in ai_platforms:
            platform_name = platform.lower().replace(' ', '_')
            if platform_name in self.ai_platforms:
                cost_data = self.ai_platforms[platform_name]
                
                # Estimate usage-based costs
                if 'per_1k_tokens' in cost_data:
                    # Estimate token usage (conservative)
                    estimated_monthly_tokens = 100000  # 100k tokens per month
                    monthly_cost = (estimated_monthly_tokens / 1000) * cost_data['per_1k_tokens']
                elif 'per_prediction' in cost_data:
                    # Estimate API calls
                    estimated_monthly_calls = 1000
                    monthly_cost = estimated_monthly_calls * cost_data['per_prediction']
                else:
                    monthly_cost = cost_data.get('monthly', 0)
                
                annual_cost = cost_data.get('annual', monthly_cost * 12)
                
                monthly_total += monthly_cost
                annual_total += annual_cost
                
                tool_details[platform] = {
                    "monthly": monthly_cost,
                    "annual": annual_cost,
                    "category": cost_data.get('category', 'ai')
                }
        
        return {
            "monthly_total": monthly_total,
            "annual_total": annual_total,
            "tool_details": tool_details
        }
    
    def _calculate_setup_costs(self) -> Dict[str, Any]:
        """Calculate one-time business setup costs"""
        setup_items = {
            "business_registration": 500,
            "domain_and_hosting": 200,
            "legal_consultation": 1000,
            "initial_design": 500,
            "business_insurance": 500
        }
        
        total = sum(setup_items.values())
        
        return {
            "items": setup_items,
            "total": total
        }
    
    def _calculate_marketing_costs(self, business_idea: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate marketing costs based on business model"""
        target_audience = business_idea.get('target_audience', '').lower()
        
        # Adjust marketing budget based on target audience
        if 'enterprise' in target_audience:
            launch_budget = 5000
            monthly_budget = 2000
        elif 'small business' in target_audience:
            launch_budget = 3000
            monthly_budget = 1000
        else:
            launch_budget = 2000
            monthly_budget = 500
        
        marketing_breakdown = {
            "content_creation": monthly_budget * 0.30,
            "paid_advertising": monthly_budget * 0.40,
            "seo_tools": monthly_budget * 0.15,
            "social_media": monthly_budget * 0.15
        }
        
        return {
            "launch_budget": launch_budget,
            "monthly_budget": monthly_budget,
            "breakdown": marketing_breakdown
        }
    
    def _calculate_operational_costs(self, business_idea: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate operational costs"""
        setup_costs = 1000  # Accounting, productivity tools, etc.
        monthly_costs = 200  # Ongoing operational expenses
        
        operational_breakdown = {
            "accounting_software": 30,
            "productivity_tools": 50,
            "customer_support": 50,
            "miscellaneous": 70
        }
        
        return {
            "setup_costs": setup_costs,
            "monthly_costs": monthly_costs,
            "breakdown": operational_breakdown
        }
    
    def _generate_revenue_projections(self, business_idea: Dict[str, Any], business_model: str, industry: str) -> Dict[str, Any]:
        """Generate detailed revenue projections"""
        revenue_model = business_idea.get('revenue_model', {})
        pricing = revenue_model.get('pricing', '')
        
        # Extract pricing information
        price_per_customer = self._extract_pricing(pricing)
        
        # Get industry benchmarks
        benchmarks = self.industry_benchmarks.get(business_model, self.industry_benchmarks['saas'])
        
        # Generate customer acquisition projections
        customer_projections = self._project_customer_growth(business_model, benchmarks)
        
        # Calculate revenue for each year
        revenue_projections = {}
        for year in range(1, 6):
            customers = customer_projections[f'year_{year}']['total_customers']
            monthly_revenue = customers * price_per_customer
            annual_revenue = monthly_revenue * 12
            
            # Apply churn
            churn_rate = benchmarks['churn_rate_monthly']['avg']
            retention_factor = (1 - churn_rate) ** 12  # Annual retention
            adjusted_revenue = annual_revenue * retention_factor
            
            revenue_projections[f'year_{year}'] = {
                "customers": customers,
                "monthly_revenue": monthly_revenue,
                "annual_revenue": adjusted_revenue,
                "average_revenue_per_customer": price_per_customer * 12 * retention_factor
            }
        
        return {
            "pricing_model": pricing,
            "price_per_customer_monthly": price_per_customer,
            "customer_projections": customer_projections,
            "revenue_by_year": revenue_projections,
            "total_5_year_revenue": sum(revenue_projections[f'year_{i}']['annual_revenue'] for i in range(1, 6))
        }
    
    def _extract_pricing(self, pricing_text: str) -> float:
        """Extract pricing from text description"""
        import re
        
        # Look for dollar amounts
        price_matches = re.findall(r'\$(\d+(?:\.\d{2})?)', pricing_text)
        if price_matches:
            return float(price_matches[0])
        
        # Default pricing based on common patterns
        if 'enterprise' in pricing_text.lower():
            return 200.0
        elif 'premium' in pricing_text.lower():
            return 99.0
        elif 'basic' in pricing_text.lower():
            return 29.0
        else:
            return 49.0  # Default
    
    def _project_customer_growth(self, business_model: str, benchmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Project customer growth over 5 years"""
        # Starting assumptions
        year_1_customers = 50  # Conservative start
        growth_rate = benchmarks['revenue_growth_rate']['avg']
        
        projections = {}
        for year in range(1, 6):
            if year == 1:
                customers = year_1_customers
            else:
                customers = int(projections[f'year_{year-1}']['total_customers'] * (1 + growth_rate))
            
            new_customers = customers - projections.get(f'year_{year-1}', {}).get('total_customers', 0)
            
            projections[f'year_{year}'] = {
                "new_customers": max(new_customers, year_1_customers if year == 1 else 0),
                "total_customers": customers,
                "growth_rate": growth_rate if year > 1 else 0
            }
        
        return projections
    
    def _calculate_unit_economics(self, business_idea: Dict[str, Any], business_model: str, industry: str) -> Dict[str, Any]:
        """Calculate unit economics (LTV, CAC, etc.)"""
        benchmarks = self.industry_benchmarks.get(business_model, self.industry_benchmarks['saas'])
        
        # Customer Acquisition Cost
        cac = benchmarks['customer_acquisition_cost']['avg']
        
        # Customer Lifetime Value
        ltv = benchmarks['lifetime_value']['avg']
        
        # Gross margin
        gross_margin = benchmarks['gross_margin']['avg']
        
        # Payback period (months)
        revenue_model = business_idea.get('revenue_model', {})
        monthly_revenue_per_customer = self._extract_pricing(revenue_model.get('pricing', ''))
        payback_period = cac / (monthly_revenue_per_customer * gross_margin) if monthly_revenue_per_customer > 0 else 12
        
        return {
            "customer_acquisition_cost": cac,
            "customer_lifetime_value": ltv,
            "ltv_cac_ratio": ltv / cac if cac > 0 else 0,
            "gross_margin_percentage": gross_margin * 100,
            "payback_period_months": payback_period,
            "monthly_revenue_per_customer": monthly_revenue_per_customer,
            "annual_revenue_per_customer": monthly_revenue_per_customer * 12,
            "unit_economics_health": "Healthy" if ltv / cac > 3 else "Concerning" if ltv / cac > 1 else "Poor"
        }
    
    def _perform_sensitivity_analysis(self, revenue_projections: Dict[str, Any], cost_breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sensitivity analysis on key variables"""
        base_revenue_5_year = revenue_projections['total_5_year_revenue']
        base_launch_cost = cost_breakdown['launch_costs']['total']
        
        scenarios = {
            "optimistic": {
                "revenue_multiplier": 1.5,
                "cost_multiplier": 0.8,
                "description": "Best case scenario"
            },
            "pessimistic": {
                "revenue_multiplier": 0.6,
                "cost_multiplier": 1.3,
                "description": "Worst case scenario"
            },
            "realistic": {
                "revenue_multiplier": 1.0,
                "cost_multiplier": 1.0,
                "description": "Base case scenario"
            }
        }
        
        scenario_results = {}
        for scenario_name, scenario in scenarios.items():
            adjusted_revenue = base_revenue_5_year * scenario['revenue_multiplier']
            adjusted_costs = base_launch_cost * scenario['cost_multiplier']
            net_profit = adjusted_revenue - adjusted_costs
            roi = (net_profit / adjusted_costs) * 100 if adjusted_costs > 0 else 0
            
            scenario_results[scenario_name] = {
                "total_revenue": adjusted_revenue,
                "total_costs": adjusted_costs,
                "net_profit": net_profit,
                "roi_percentage": roi,
                "description": scenario['description']
            }
        
        return scenario_results
    
    def _calculate_financial_metrics(self, revenue_projections: Dict[str, Any], cost_breakdown: Dict[str, Any], unit_economics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key financial metrics"""
        total_revenue_5_year = revenue_projections['total_5_year_revenue']
        total_launch_cost = cost_breakdown['launch_costs']['total']
        annual_operating_cost = cost_breakdown['annual_costs']['total']
        
        # Net profit calculation
        total_operating_costs_5_year = annual_operating_cost * 5
        net_profit_5_year = total_revenue_5_year - total_launch_cost - total_operating_costs_5_year
        
        # ROI calculation
        roi = (net_profit_5_year / total_launch_cost) * 100 if total_launch_cost > 0 else 0
        
        # Profit margin
        profit_margin = (net_profit_5_year / total_revenue_5_year) * 100 if total_revenue_5_year > 0 else 0
        
        return {
            "total_revenue_5_year": total_revenue_5_year,
            "total_costs_5_year": total_launch_cost + total_operating_costs_5_year,
            "net_profit_5_year": net_profit_5_year,
            "roi_percentage": roi,
            "profit_margin_percentage": profit_margin,
            "break_even_point_months": self._calculate_break_even_months(revenue_projections, cost_breakdown),
            "cash_flow_positive_month": self._calculate_cash_flow_positive_month(revenue_projections, cost_breakdown)
        }
    
    def _calculate_break_even_months(self, revenue_projections: Dict[str, Any], cost_breakdown: Dict[str, Any]) -> int:
        """Calculate break-even point in months"""
        monthly_revenue_year_1 = revenue_projections['revenue_by_year']['year_1']['monthly_revenue']
        monthly_costs = cost_breakdown['monthly_costs']['total']
        launch_costs = cost_breakdown['launch_costs']['total']
        
        if monthly_revenue_year_1 <= monthly_costs:
            return 60  # More than 5 years
        
        net_monthly_profit = monthly_revenue_year_1 - monthly_costs
        break_even_months = math.ceil(launch_costs / net_monthly_profit)
        
        return min(break_even_months, 60)  # Cap at 5 years
    
    def _calculate_cash_flow_positive_month(self, revenue_projections: Dict[str, Any], cost_breakdown: Dict[str, Any]) -> int:
        """Calculate when cash flow becomes positive"""
        monthly_revenue_year_1 = revenue_projections['revenue_by_year']['year_1']['monthly_revenue']
        monthly_costs = cost_breakdown['monthly_costs']['total']
        
        if monthly_revenue_year_1 > monthly_costs:
            return 1  # Positive from month 1
        else:
            # Assume gradual revenue growth
            growth_rate = 0.1  # 10% monthly growth
            month = 1
            current_revenue = monthly_revenue_year_1
            
            while current_revenue <= monthly_costs and month <= 60:
                current_revenue *= (1 + growth_rate)
                month += 1
            
            return month
    
    def _analyze_funding_requirements(self, cost_breakdown: Dict[str, Any], revenue_projections: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze funding requirements and options"""
        launch_costs = cost_breakdown['launch_costs']['total']
        monthly_costs = cost_breakdown['monthly_costs']['total']
        monthly_revenue_year_1 = revenue_projections['revenue_by_year']['year_1']['monthly_revenue']
        
        # Calculate runway needed
        monthly_burn = monthly_costs - monthly_revenue_year_1
        runway_months = 12 if monthly_burn <= 0 else min(24, max(12, launch_costs / monthly_burn))
        
        # Total funding needed
        total_funding_needed = launch_costs + (max(0, monthly_burn) * runway_months)
        
        # Funding options
        funding_options = []
        
        if total_funding_needed <= 10000:
            funding_options.append({
                "type": "Bootstrapping",
                "amount": total_funding_needed,
                "probability": "High",
                "timeline": "Immediate",
                "requirements": "Personal savings or revenue"
            })
        
        if total_funding_needed <= 50000:
            funding_options.append({
                "type": "Friends & Family",
                "amount": min(25000, total_funding_needed),
                "probability": "Medium",
                "timeline": "1-3 months",
                "requirements": "Personal network"
            })
        
        if total_funding_needed >= 25000:
            funding_options.append({
                "type": "Angel Investment",
                "amount": min(100000, total_funding_needed * 2),
                "probability": "Medium",
                "timeline": "3-6 months",
                "requirements": "Traction and pitch deck"
            })
        
        return {
            "total_funding_needed": total_funding_needed,
            "runway_months": runway_months,
            "monthly_burn_rate": max(0, monthly_burn),
            "funding_options": funding_options,
            "self_funded_feasible": total_funding_needed <= 10000,
            "funding_recommendation": self._get_funding_recommendation(total_funding_needed)
        }
    
    def _get_funding_recommendation(self, funding_needed: float) -> str:
        """Get funding recommendation based on amount needed"""
        if funding_needed <= 5000:
            return "Bootstrap with personal savings - very feasible for solo founder"
        elif funding_needed <= 15000:
            return "Bootstrap or seek small angel investment - manageable risk"
        elif funding_needed <= 50000:
            return "Seek angel investment or accelerator program"
        else:
            return "Consider reducing scope or seeking significant investment"
    
    def _assess_financial_risks(self, business_idea: Dict[str, Any], financial_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess financial risks"""
        risks = []
        risk_score = 0
        
        # Revenue risk
        if financial_metrics['break_even_point_months'] > 24:
            risks.append("Long break-even period increases financial risk")
            risk_score += 2
        
        # Market risk
        competitive_landscape = business_idea.get('competitive_landscape', {})
        if competitive_landscape.get('market_saturation') == 'high':
            risks.append("High market saturation may limit revenue growth")
            risk_score += 2
        
        # Technology risk
        implementation_plan = business_idea.get('implementation_plan', {})
        if len(implementation_plan.get('ai_platforms', [])) > 3:
            risks.append("Complex technology stack increases operational costs")
            risk_score += 1
        
        # Customer acquisition risk
        target_audience = business_idea.get('target_audience', '').lower()
        if 'enterprise' in target_audience:
            risks.append("Enterprise sales cycles may delay revenue")
            risk_score += 1
        
        # Determine overall risk level
        if risk_score <= 2:
            risk_level = "Low"
        elif risk_score <= 4:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            "risk_factors": risks,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "mitigation_strategies": self._get_risk_mitigation_strategies(risks)
        }
    
    def _get_risk_mitigation_strategies(self, risks: List[str]) -> List[str]:
        """Get risk mitigation strategies"""
        strategies = [
            "Start with MVP to validate market demand",
            "Focus on customer development and feedback",
            "Maintain lean operations and low burn rate",
            "Build strategic partnerships for customer acquisition",
            "Diversify revenue streams when possible"
        ]
        return strategies[:3]  # Return top 3 strategies
    
    def _calculate_break_even(self, revenue_projections: Dict[str, Any], cost_breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed break-even analysis"""
        monthly_revenue_year_1 = revenue_projections['revenue_by_year']['year_1']['monthly_revenue']
        monthly_costs = cost_breakdown['monthly_costs']['total']
        launch_costs = cost_breakdown['launch_costs']['total']
        
        break_even_months = self._calculate_break_even_months(revenue_projections, cost_breakdown)
        break_even_revenue = break_even_months * monthly_revenue_year_1
        break_even_customers = break_even_revenue / (revenue_projections['price_per_customer_monthly'] * break_even_months) if revenue_projections['price_per_customer_monthly'] > 0 else 0
        
        return {
            "break_even_months": break_even_months,
            "break_even_revenue": break_even_revenue,
            "break_even_customers": int(break_even_customers),
            "monthly_revenue_needed": monthly_costs,
            "current_monthly_revenue": monthly_revenue_year_1,
            "revenue_gap": max(0, monthly_costs - monthly_revenue_year_1)
        }
    
    def _calculate_roi_analysis(self, revenue_projections: Dict[str, Any], cost_breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed ROI analysis"""
        total_investment = cost_breakdown['launch_costs']['total']
        total_revenue_5_year = revenue_projections['total_5_year_revenue']
        total_operating_costs_5_year = cost_breakdown['annual_costs']['total'] * 5
        
        net_profit = total_revenue_5_year - total_investment - total_operating_costs_5_year
        roi_percentage = (net_profit / total_investment) * 100 if total_investment > 0 else 0
        
        # Annual ROI
        annual_roi = (roi_percentage / 5) if roi_percentage > 0 else 0
        
        return {
            "total_investment": total_investment,
            "total_revenue_5_year": total_revenue_5_year,
            "total_costs_5_year": total_investment + total_operating_costs_5_year,
            "net_profit_5_year": net_profit,
            "roi_percentage_5_year": roi_percentage,
            "annual_roi_percentage": annual_roi,
            "roi_grade": "Excellent" if roi_percentage > 500 else "Good" if roi_percentage > 200 else "Fair" if roi_percentage > 100 else "Poor"
        }
    
    def _get_fallback_financial_analysis(self) -> Dict[str, Any]:
        """Fallback financial analysis if calculations fail"""
        return {
            "business_model": "saas",
            "industry_category": "general",
            "cost_breakdown": {
                "launch_costs": {"total": 8000},
                "monthly_costs": {"total": 500},
                "annual_costs": {"total": 6000}
            },
            "revenue_projections": {
                "total_5_year_revenue": 500000,
                "price_per_customer_monthly": 50
            },
            "unit_economics": {
                "customer_acquisition_cost": 250,
                "customer_lifetime_value": 1500,
                "ltv_cac_ratio": 6.0
            },
            "financial_metrics": {
                "roi_percentage": 300,
                "break_even_point_months": 12,
                "profit_margin_percentage": 25
            },
            "funding_analysis": {
                "total_funding_needed": 8000,
                "self_funded_feasible": True
            },
            "risk_assessment": {
                "risk_level": "Medium",
                "risk_factors": ["Market validation needed"]
            },
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

