from flask import Blueprint, request, jsonify, session
from datetime import datetime, timedelta
import json
import traceback
from src.auth import login_required
from src.models.business_idea import BusinessIdea, ValidationRequest, db
from src.services.enhanced_ai_service import EnhancedAIService
from src.services.market_research_service import MarketResearchService
from src.services.financial_analysis_service import FinancialAnalysisService

enhanced_ideas_bp = Blueprint('enhanced_ideas', __name__)

# Initialize services
enhanced_ai_service = EnhancedAIService()
market_research_service = MarketResearchService()
financial_analysis_service = FinancialAnalysisService()

@enhanced_ideas_bp.route('/generate-enhanced', methods=['POST'])
@login_required
def generate_enhanced_ideas():
    """Generate enhanced business ideas with comprehensive analysis"""
    try:
        data = request.get_json() or {}
        count = min(data.get('count', 5), 10)  # Limit to 10 ideas max
        
        print(f"Generating {count} enhanced business ideas...")
        
        # Generate enhanced ideas
        ideas = enhanced_ai_service.generate_enhanced_business_ideas(count)
        
        # Process each idea with additional analysis
        enhanced_ideas = []
        for idea in ideas:
            try:
                # Conduct market research
                print(f"Conducting market research for: {idea.get('name', 'Unknown')}")
                market_research = market_research_service.research_market_opportunity(idea)
                idea['market_research'] = market_research
                
                # Perform financial analysis
                print(f"Performing financial analysis for: {idea.get('name', 'Unknown')}")
                financial_analysis = financial_analysis_service.analyze_financial_projections(idea)
                idea['financial_analysis'] = financial_analysis
                
                # Calculate enhanced overall score
                idea['enhanced_score'] = calculate_enhanced_overall_score(idea)
                
                # Save to database
                business_idea = BusinessIdea(
                    name=idea.get('name', 'Untitled'),
                    summary=idea.get('tagline', ''),
                    target_audience=idea.get('target_audience', ''),
                    problem_solved=idea.get('problem_statement', ''),
                    ai_solution=idea.get('ai_solution', ''),
                    implementation=json.dumps(idea.get('implementation_plan', {})),
                    revenue_model=json.dumps(idea.get('revenue_model', {})),
                    launch_cost=idea.get('financial_analysis', {}).get('cost_breakdown', {}).get('launch_costs', {}).get('total', 0),
                    revenue_1_year=idea.get('financial_analysis', {}).get('revenue_projections', {}).get('revenue_by_year', {}).get('year_1', {}).get('annual_revenue', 0),
                    revenue_5_year=idea.get('financial_analysis', {}).get('revenue_projections', {}).get('total_5_year_revenue', 0),
                    scores=json.dumps(idea.get('scores', {})),
                    validation_evidence=json.dumps(idea.get('validation_evidence', {})),
                    niche=idea.get('niche_category', 'general'),
                    generated_at=datetime.utcnow(),
                    enhanced_data=json.dumps({
                        'market_research': market_research,
                        'financial_analysis': financial_analysis,
                        'enhanced_score': idea['enhanced_score'],
                        'full_idea_data': idea
                    })
                )
                
                db.session.add(business_idea)
                enhanced_ideas.append(idea)
                
            except Exception as e:
                print(f"Error processing idea {idea.get('name', 'Unknown')}: {e}")
                # Add idea with error flag
                idea['processing_error'] = str(e)
                enhanced_ideas.append(idea)
        
        # Commit all ideas to database
        try:
            db.session.commit()
            print(f"Successfully saved {len(enhanced_ideas)} enhanced ideas to database")
        except Exception as e:
            print(f"Error saving to database: {e}")
            db.session.rollback()
        
        return jsonify({
            'success': True,
            'ideas': enhanced_ideas,
            'count': len(enhanced_ideas),
            'generation_timestamp': datetime.utcnow().isoformat(),
            'enhanced_features': [
                'Real market validation',
                'Comprehensive financial analysis',
                'Competitive intelligence',
                'Industry benchmarks',
                'Risk assessment',
                'Funding analysis'
            ]
        })
        
    except Exception as e:
        print(f"Error generating enhanced ideas: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'fallback_message': 'Enhanced idea generation failed. Please try again.'
        }), 500

@enhanced_ideas_bp.route('/analyze-idea', methods=['POST'])
@login_required
def analyze_existing_idea():
    """Perform enhanced analysis on an existing idea"""
    try:
        data = request.get_json()
        idea_id = data.get('idea_id')
        
        if not idea_id:
            return jsonify({'success': False, 'error': 'Idea ID required'}), 400
        
        # Get idea from database
        business_idea = BusinessIdea.query.get(idea_id)
        if not business_idea:
            return jsonify({'success': False, 'error': 'Idea not found'}), 404
        
        # Convert to enhanced format
        idea_data = {
            'name': business_idea.name,
            'tagline': business_idea.summary,
            'target_audience': business_idea.target_audience,
            'problem_statement': business_idea.problem_solved,
            'ai_solution': business_idea.ai_solution,
            'implementation_plan': json.loads(business_idea.implementation) if business_idea.implementation else {},
            'revenue_model': json.loads(business_idea.revenue_model) if business_idea.revenue_model else {}
        }
        
        # Perform enhanced analysis
        market_research = market_research_service.research_market_opportunity(idea_data)
        financial_analysis = financial_analysis_service.analyze_financial_projections(idea_data)
        enhanced_scores = enhanced_ai_service.calculate_enhanced_scores(idea_data)
        
        # Update database with enhanced data
        business_idea.enhanced_data = json.dumps({
            'market_research': market_research,
            'financial_analysis': financial_analysis,
            'enhanced_scores': enhanced_scores,
            'analysis_timestamp': datetime.utcnow().isoformat()
        })
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'idea_id': idea_id,
            'market_research': market_research,
            'financial_analysis': financial_analysis,
            'enhanced_scores': enhanced_scores,
            'analysis_timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"Error analyzing existing idea: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@enhanced_ideas_bp.route('/market-research', methods=['POST'])
@login_required
def conduct_market_research():
    """Conduct standalone market research"""
    try:
        data = request.get_json()
        keywords = data.get('keywords', [])
        business_context = data.get('business_context', {})
        
        if not keywords:
            return jsonify({'success': False, 'error': 'Keywords required'}), 400
        
        # Conduct market research
        research_data = market_research_service.research_market_opportunity(business_context)
        
        return jsonify({
            'success': True,
            'research_data': research_data,
            'keywords': keywords,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"Error conducting market research: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@enhanced_ideas_bp.route('/financial-analysis', methods=['POST'])
@login_required
def perform_financial_analysis():
    """Perform standalone financial analysis"""
    try:
        data = request.get_json()
        business_idea = data.get('business_idea', {})
        
        if not business_idea:
            return jsonify({'success': False, 'error': 'Business idea data required'}), 400
        
        # Perform financial analysis
        financial_data = financial_analysis_service.analyze_financial_projections(business_idea)
        
        return jsonify({
            'success': True,
            'financial_analysis': financial_data,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"Error performing financial analysis: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@enhanced_ideas_bp.route('/enhanced-ideas', methods=['GET'])
@login_required
def get_enhanced_ideas():
    """Get all enhanced ideas with filtering"""
    try:
        # Query parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 50)
        niche = request.args.get('niche')
        min_score = request.args.get('min_score', type=float)
        sort_by = request.args.get('sort_by', 'generated_at')
        order = request.args.get('order', 'desc')
        
        # Build query
        query = BusinessIdea.query
        
        # Apply filters
        if niche:
            query = query.filter(BusinessIdea.niche == niche)
        
        if min_score:
            # This would require a more complex query to filter by JSON data
            pass
        
        # Apply sorting
        if sort_by == 'generated_at':
            if order == 'desc':
                query = query.order_by(BusinessIdea.generated_at.desc())
            else:
                query = query.order_by(BusinessIdea.generated_at.asc())
        elif sort_by == 'name':
            if order == 'desc':
                query = query.order_by(BusinessIdea.name.desc())
            else:
                query = query.order_by(BusinessIdea.name.asc())
        
        # Paginate
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        # Format results
        ideas = []
        for idea in pagination.items:
            idea_data = {
                'id': idea.id,
                'name': idea.name,
                'summary': idea.summary,
                'target_audience': idea.target_audience,
                'problem_solved': idea.problem_solved,
                'ai_solution': idea.ai_solution,
                'launch_cost': idea.launch_cost,
                'revenue_1_year': idea.revenue_1_year,
                'revenue_5_year': idea.revenue_5_year,
                'niche': idea.niche,
                'generated_at': idea.generated_at.isoformat(),
                'scores': json.loads(idea.scores) if idea.scores else {},
                'validation_evidence': json.loads(idea.validation_evidence) if idea.validation_evidence else {},
                'has_enhanced_data': bool(idea.enhanced_data)
            }
            
            # Add enhanced data if available
            if idea.enhanced_data:
                try:
                    enhanced_data = json.loads(idea.enhanced_data)
                    idea_data['enhanced_data'] = enhanced_data
                except:
                    pass
            
            ideas.append(idea_data)
        
        return jsonify({
            'success': True,
            'ideas': ideas,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            },
            'filters': {
                'niche': niche,
                'min_score': min_score,
                'sort_by': sort_by,
                'order': order
            }
        })
        
    except Exception as e:
        print(f"Error getting enhanced ideas: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@enhanced_ideas_bp.route('/idea/<int:idea_id>/detailed', methods=['GET'])
@login_required
def get_detailed_idea(idea_id):
    """Get detailed view of a specific idea"""
    try:
        business_idea = BusinessIdea.query.get(idea_id)
        if not business_idea:
            return jsonify({'success': False, 'error': 'Idea not found'}), 404
        
        # Build detailed response
        detailed_data = {
            'id': business_idea.id,
            'name': business_idea.name,
            'summary': business_idea.summary,
            'target_audience': business_idea.target_audience,
            'problem_solved': business_idea.problem_solved,
            'ai_solution': business_idea.ai_solution,
            'implementation': json.loads(business_idea.implementation) if business_idea.implementation else {},
            'revenue_model': json.loads(business_idea.revenue_model) if business_idea.revenue_model else {},
            'launch_cost': business_idea.launch_cost,
            'revenue_1_year': business_idea.revenue_1_year,
            'revenue_5_year': business_idea.revenue_5_year,
            'niche': business_idea.niche,
            'generated_at': business_idea.generated_at.isoformat(),
            'scores': json.loads(business_idea.scores) if business_idea.scores else {},
            'validation_evidence': json.loads(business_idea.validation_evidence) if business_idea.validation_evidence else {}
        }
        
        # Add enhanced data if available
        if business_idea.enhanced_data:
            try:
                enhanced_data = json.loads(business_idea.enhanced_data)
                detailed_data['enhanced_data'] = enhanced_data
            except Exception as e:
                print(f"Error parsing enhanced data: {e}")
        
        return jsonify({
            'success': True,
            'idea': detailed_data
        })
        
    except Exception as e:
        print(f"Error getting detailed idea: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def calculate_enhanced_overall_score(idea: dict) -> dict:
    """Calculate enhanced overall score combining multiple factors"""
    try:
        # Get component scores
        ai_scores = idea.get('scores', {})
        market_research = idea.get('market_research', {})
        financial_analysis = idea.get('financial_analysis', {})
        
        # AI scoring (40% weight)
        ai_total = ai_scores.get('total', 6.0)
        
        # Market research scoring (35% weight)
        market_score = market_research.get('market_score', {}).get('total_score', 6.0)
        
        # Financial scoring (25% weight)
        financial_metrics = financial_analysis.get('financial_metrics', {})
        roi = financial_metrics.get('roi_percentage', 100)
        break_even = financial_metrics.get('break_even_point_months', 12)
        
        # Convert financial metrics to 1-10 scale
        roi_score = min(10, max(1, roi / 100))  # 100% ROI = 1 point, 1000% ROI = 10 points
        break_even_score = max(1, 10 - (break_even / 6))  # 6 months = 9 points, 60 months = 1 point
        financial_score = (roi_score + break_even_score) / 2
        
        # Calculate weighted total
        total_score = (ai_total * 0.40) + (market_score * 0.35) + (financial_score * 0.25)
        
        # Determine grade and recommendation
        if total_score >= 8.5:
            grade = 'A+'
            recommendation = 'Exceptional opportunity - proceed immediately'
        elif total_score >= 8.0:
            grade = 'A'
            recommendation = 'Excellent opportunity - high confidence'
        elif total_score >= 7.5:
            grade = 'A-'
            recommendation = 'Very good opportunity - recommended'
        elif total_score >= 7.0:
            grade = 'B+'
            recommendation = 'Good opportunity - validate key assumptions'
        elif total_score >= 6.5:
            grade = 'B'
            recommendation = 'Moderate opportunity - needs improvement'
        elif total_score >= 6.0:
            grade = 'B-'
            recommendation = 'Below average - significant concerns'
        else:
            grade = 'C'
            recommendation = 'Poor opportunity - not recommended'
        
        return {
            'total_score': round(total_score, 2),
            'component_scores': {
                'ai_analysis': ai_total,
                'market_research': market_score,
                'financial_analysis': financial_score
            },
            'weights': {
                'ai_analysis': 0.40,
                'market_research': 0.35,
                'financial_analysis': 0.25
            },
            'grade': grade,
            'recommendation': recommendation,
            'score_breakdown': {
                'ai_reasoning': ai_scores.get('reasoning', {}),
                'market_factors': market_research.get('market_score', {}).get('component_scores', {}),
                'financial_factors': {
                    'roi_score': round(roi_score, 2),
                    'break_even_score': round(break_even_score, 2)
                }
            }
        }
        
    except Exception as e:
        print(f"Error calculating enhanced score: {e}")
        return {
            'total_score': 6.0,
            'grade': 'C',
            'recommendation': 'Score calculation failed - manual review needed',
            'error': str(e)
        }

