from src.models.user import db
from datetime import datetime
import json

class BusinessIdea(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    summary = db.Column(db.Text, nullable=False)
    target_audience = db.Column(db.String(500), nullable=False)
    problem_solved = db.Column(db.Text, nullable=False)
    ai_solution = db.Column(db.Text, nullable=False)
    implementation = db.Column(db.Text, nullable=False)
    revenue_model = db.Column(db.Text, nullable=False)
    launch_cost = db.Column(db.Integer, nullable=False)  # in dollars
    revenue_1_year = db.Column(db.Integer, nullable=False)  # in dollars
    revenue_5_year = db.Column(db.Integer, nullable=False)  # in dollars
    
    # Scoring dimensions (1-10 scale) - keeping for backward compatibility
    cost_to_build_score = db.Column(db.Float, nullable=True, default=6.0)
    ease_of_implementation_score = db.Column(db.Float, nullable=True, default=6.0)
    market_size_score = db.Column(db.Float, nullable=True, default=6.0)
    competition_level_score = db.Column(db.Float, nullable=True, default=6.0)
    problem_severity_score = db.Column(db.Float, nullable=True, default=6.0)
    founder_fit_score = db.Column(db.Float, nullable=True, default=6.0)
    total_score = db.Column(db.Float, nullable=True, default=6.0)
    
    # Enhanced scoring system (stored as JSON)
    scores = db.Column(db.Text, nullable=True)  # JSON string with detailed scores
    
    # Validation evidence (stored as JSON)
    validation_evidence = db.Column(db.Text, nullable=True)  # JSON string
    
    # Enhanced data fields
    enhanced_data = db.Column(db.Text, nullable=True)  # JSON string with market research, financial analysis, etc.
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)  # For compatibility
    is_featured = db.Column(db.Boolean, default=False)
    niche = db.Column(db.String(100), nullable=True)
    tags = db.Column(db.String(500), nullable=True)  # comma-separated
    
    def __repr__(self):
        return f'<BusinessIdea {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'summary': self.summary,
            'target_audience': self.target_audience,
            'problem_solved': self.problem_solved,
            'ai_solution': self.ai_solution,
            'implementation': self.implementation,
            'revenue_model': self.revenue_model,
            'launch_cost': self.launch_cost,
            'revenue_1_year': self.revenue_1_year,
            'revenue_5_year': self.revenue_5_year,
            'scores': self.get_scores(),
            'validation_evidence': self.get_validation_evidence(),
            'enhanced_data': self.get_enhanced_data(),
            'created_at': self.created_at.isoformat(),
            'generated_at': self.generated_at.isoformat() if self.generated_at else self.created_at.isoformat(),
            'is_featured': self.is_featured,
            'niche': self.niche,
            'tags': self.tags.split(',') if self.tags else []
        }
    
    def get_scores(self):
        """Get scores as a dictionary (enhanced or legacy)"""
        if self.scores:
            try:
                return json.loads(self.scores)
            except:
                pass
        
        # Fallback to legacy scores
        return {
            'cost_to_build': self.cost_to_build_score or 6.0,
            'ease_of_implementation': self.ease_of_implementation_score or 6.0,
            'market_size': self.market_size_score or 6.0,
            'competition_level': self.competition_level_score or 6.0,
            'problem_severity': self.problem_severity_score or 6.0,
            'founder_fit': self.founder_fit_score or 6.0,
            'total': self.total_score or 6.0
        }
    
    def set_scores(self, scores_dict):
        """Set scores from a dictionary"""
        self.scores = json.dumps(scores_dict)
        
        # Also update legacy fields for compatibility
        if isinstance(scores_dict, dict):
            self.cost_to_build_score = scores_dict.get('cost_to_build', 6.0)
            self.ease_of_implementation_score = scores_dict.get('ease_of_implementation', 6.0)
            self.market_size_score = scores_dict.get('market_size', 6.0)
            self.competition_level_score = scores_dict.get('competition_level', 6.0)
            self.problem_severity_score = scores_dict.get('problem_severity', 6.0)
            self.founder_fit_score = scores_dict.get('founder_fit', 6.0)
            self.total_score = scores_dict.get('total', 6.0)
    
    def set_validation_evidence(self, evidence_dict):
        """Set validation evidence from a dictionary"""
        self.validation_evidence = json.dumps(evidence_dict)
    
    def get_validation_evidence(self):
        """Get validation evidence as a dictionary"""
        try:
            return json.loads(self.validation_evidence) if self.validation_evidence else {}
        except:
            return {}
    
    def set_enhanced_data(self, enhanced_dict):
        """Set enhanced data from a dictionary"""
        self.enhanced_data = json.dumps(enhanced_dict)
    
    def get_enhanced_data(self):
        """Get enhanced data as a dictionary"""
        try:
            return json.loads(self.enhanced_data) if self.enhanced_data else {}
        except:
            return {}

class ValidationRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    business_idea_id = db.Column(db.Integer, db.ForeignKey('business_idea.id'), nullable=False)
    user_email = db.Column(db.String(120), nullable=True)
    requested_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(50), default='pending')  # pending, processing, completed, failed
    detailed_validation = db.Column(db.Text, nullable=True)  # JSON string
    
    business_idea = db.relationship('BusinessIdea', backref=db.backref('validation_requests', lazy=True))
    
    def __repr__(self):
        return f'<ValidationRequest {self.id} for idea {self.business_idea_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'business_idea_id': self.business_idea_id,
            'user_email': self.user_email,
            'requested_at': self.requested_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'detailed_validation': json.loads(self.detailed_validation) if self.detailed_validation else None
        }
    
    def set_detailed_validation(self, validation_dict):
        """Set detailed validation from a dictionary"""
        self.detailed_validation = json.dumps(validation_dict)
    
    def get_detailed_validation(self):
        """Get detailed validation as a dictionary"""
        return json.loads(self.detailed_validation) if self.detailed_validation else {}

class DailyStats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, unique=True, nullable=False)
    ideas_generated = db.Column(db.Integer, default=0)
    emails_sent = db.Column(db.Integer, default=0)
    validation_requests = db.Column(db.Integer, default=0)
    website_visits = db.Column(db.Integer, default=0)
    
    def __repr__(self):
        return f'<DailyStats {self.date}>'
    
    def to_dict(self):
        return {
            'date': self.date.isoformat(),
            'ideas_generated': self.ideas_generated,
            'emails_sent': self.emails_sent,
            'validation_requests': self.validation_requests,
            'website_visits': self.website_visits
        }

