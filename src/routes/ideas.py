from flask import Blueprint, request, jsonify
from datetime import datetime, date, timedelta
from src.models.business_idea import db, BusinessIdea, ValidationRequest, DailyStats
from src.services.ai_service import AIService

ideas_bp = Blueprint('ideas', __name__)
ai_service = AIService()

@ideas_bp.route('/ideas', methods=['GET'])
def get_ideas():
    """Get business ideas with optional filtering"""
    try:
        # Query parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 50)
        niche = request.args.get('niche')
        min_score = request.args.get('min_score', type=float)
        sort_by = request.args.get('sort_by', 'created_at')
        order = request.args.get('order', 'desc')
        
        # Build query
        query = BusinessIdea.query
        
        if niche:
            query = query.filter(BusinessIdea.niche.ilike(f'%{niche}%'))
        
        if min_score:
            query = query.filter(BusinessIdea.total_score >= min_score)
        
        # Sorting
        if sort_by == 'score':
            sort_column = BusinessIdea.total_score
        elif sort_by == 'cost':
            sort_column = BusinessIdea.launch_cost
        else:
            sort_column = BusinessIdea.created_at
        
        if order == 'desc':
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())
        
        # Pagination
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        ideas = [idea.to_dict() for idea in pagination.items]
        
        return jsonify({
            'ideas': ideas,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ideas_bp.route('/ideas/<int:idea_id>', methods=['GET'])
def get_idea(idea_id):
    """Get a specific business idea"""
    try:
        idea = BusinessIdea.query.get_or_404(idea_id)
        return jsonify(idea.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ideas_bp.route('/ideas/generate', methods=['POST'])
def generate_ideas():
    """Generate new business ideas"""
    try:
        data = request.get_json() or {}
        count = min(data.get('count', 5), 10)  # Max 10 ideas at once
        
        # Generate ideas using AI service
        raw_ideas = ai_service.generate_business_ideas(count)
        
        saved_ideas = []
        for raw_idea in raw_ideas:
            # Score the idea
            scores = ai_service.score_business_idea(raw_idea)
            
            # Generate validation evidence
            validation_evidence = ai_service.generate_validation_evidence(raw_idea)
            
            # Create database record
            idea = BusinessIdea(
                name=raw_idea.get('name', ''),
                summary=raw_idea.get('summary', ''),
                target_audience=raw_idea.get('target_audience', ''),
                problem_solved=raw_idea.get('problem_solved', ''),
                ai_solution=raw_idea.get('ai_solution', ''),
                implementation=raw_idea.get('implementation', ''),
                revenue_model=raw_idea.get('revenue_model', ''),
                launch_cost=raw_idea.get('launch_cost', 0),
                revenue_1_year=raw_idea.get('revenue_1_year', 0),
                revenue_5_year=raw_idea.get('revenue_5_year', 0),
                cost_to_build_score=scores.get('cost_to_build', 5.0),
                ease_of_implementation_score=scores.get('ease_of_implementation', 5.0),
                market_size_score=scores.get('market_size', 5.0),
                competition_level_score=scores.get('competition_level', 5.0),
                problem_severity_score=scores.get('problem_severity', 5.0),
                founder_fit_score=scores.get('founder_fit', 5.0),
                total_score=scores.get('total', 5.0),
                niche=raw_idea.get('niche', ''),
                tags=''  # Can be populated later
            )
            
            idea.set_validation_evidence(validation_evidence)
            
            db.session.add(idea)
            saved_ideas.append(idea)
        
        db.session.commit()
        
        # Update daily stats
        today = date.today()
        stats = DailyStats.query.filter_by(date=today).first()
        if not stats:
            stats = DailyStats(date=today, ideas_generated=count)
            db.session.add(stats)
        else:
            stats.ideas_generated += count
        
        db.session.commit()
        
        return jsonify({
            'message': f'Generated {len(saved_ideas)} business ideas',
            'ideas': [idea.to_dict() for idea in saved_ideas]
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@ideas_bp.route('/ideas/<int:idea_id>/validate', methods=['POST'])
def request_validation(idea_id):
    """Request deeper validation for an idea"""
    try:
        data = request.get_json() or {}
        user_email = data.get('email')
        
        idea = BusinessIdea.query.get_or_404(idea_id)
        
        # Check daily limit (3 requests per day per idea)
        today = date.today()
        today_requests = ValidationRequest.query.filter(
            ValidationRequest.business_idea_id == idea_id,
            ValidationRequest.requested_at >= datetime.combine(today, datetime.min.time())
        ).count()
        
        if today_requests >= 3:
            return jsonify({'error': 'Daily validation limit reached for this idea'}), 429
        
        # Create validation request
        validation_request = ValidationRequest(
            business_idea_id=idea_id,
            user_email=user_email,
            status='pending'
        )
        
        db.session.add(validation_request)
        db.session.commit()
        
        # Update daily stats
        stats = DailyStats.query.filter_by(date=today).first()
        if not stats:
            stats = DailyStats(date=today, validation_requests=1)
            db.session.add(stats)
        else:
            stats.validation_requests += 1
        
        db.session.commit()
        
        return jsonify({
            'message': 'Validation request submitted successfully',
            'request_id': validation_request.id,
            'estimated_completion': '24 hours'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@ideas_bp.route('/ideas/today', methods=['GET'])
def get_todays_ideas():
    """Get today's featured ideas"""
    try:
        today = date.today()
        ideas = BusinessIdea.query.filter(
            BusinessIdea.created_at >= datetime.combine(today, datetime.min.time())
        ).order_by(BusinessIdea.total_score.desc()).limit(5).all()
        
        return jsonify({
            'date': today.isoformat(),
            'ideas': [idea.to_dict() for idea in ideas]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ideas_bp.route('/ideas/niches', methods=['GET'])
def get_niches():
    """Get available niches"""
    try:
        niches = db.session.query(BusinessIdea.niche).distinct().all()
        niche_list = [niche[0] for niche in niches if niche[0]]
        
        return jsonify({
            'niches': sorted(niche_list)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ideas_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get platform statistics"""
    try:
        total_ideas = BusinessIdea.query.count()
        total_validations = ValidationRequest.query.count()
        avg_score = db.session.query(db.func.avg(BusinessIdea.total_score)).scalar() or 0
        
        # Recent activity (last 7 days)
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_ideas = BusinessIdea.query.filter(
            BusinessIdea.created_at >= seven_days_ago
        ).count()
        
        return jsonify({
            'total_ideas': total_ideas,
            'total_validations': total_validations,
            'average_score': round(avg_score, 2),
            'recent_ideas': recent_ideas
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

