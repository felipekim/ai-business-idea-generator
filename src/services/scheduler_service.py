import os
import sys
import time
import threading
import schedule
from datetime import datetime, date
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.services.ai_service import AIService
from src.services.email_service import EmailService
from src.models.business_idea import db, BusinessIdea, DailyStats
from flask import Flask

class SchedulerService:
    def __init__(self, app: Flask = None):
        self.app = app
        self.ai_service = AIService()
        self.email_service = EmailService()
        self.running = False
        self.scheduler_thread = None
        
    def generate_and_send_daily_ideas(self):
        """Generate 5 new business ideas and send them via email"""
        try:
            print(f"[{datetime.now()}] Starting daily idea generation...")
            
            if self.app:
                with self.app.app_context():
                    # Generate 5 new ideas
                    raw_ideas = self.ai_service.generate_business_ideas(5)
                    
                    saved_ideas = []
                    for raw_idea in raw_ideas:
                        # Score the idea
                        scores = self.ai_service.score_business_idea(raw_idea)
                        
                        # Generate validation evidence
                        validation_evidence = self.ai_service.generate_validation_evidence(raw_idea)
                        
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
                            tags=''
                        )
                        
                        idea.set_validation_evidence(validation_evidence)
                        
                        db.session.add(idea)
                        saved_ideas.append(idea)
                    
                    db.session.commit()
                    
                    # Update daily stats
                    today = date.today()
                    stats = DailyStats.query.filter_by(date=today).first()
                    if not stats:
                        stats = DailyStats(date=today, ideas_generated=5, emails_sent=0)
                        db.session.add(stats)
                    else:
                        stats.ideas_generated += 5
                    
                    # Convert ideas to dict format for email
                    ideas_data = []
                    for idea in saved_ideas:
                        idea_dict = idea.to_dict()
                        # Add scores in the expected format
                        idea_dict['scores'] = {
                            'cost_to_build': idea.cost_to_build_score,
                            'ease_of_implementation': idea.ease_of_implementation_score,
                            'market_size': idea.market_size_score,
                            'competition_level': idea.competition_level_score,
                            'problem_severity': idea.problem_severity_score,
                            'founder_fit': idea.founder_fit_score,
                            'total': idea.total_score
                        }
                        ideas_data.append(idea_dict)
                    
                    # Email delivery disabled per user request
                    # email_sent = self.email_service.send_daily_ideas_email(
                    #     ideas_data, 
                    #     'ideasaibusiness@gmail.com'
                    # )
                    email_sent = False  # Email delivery disabled
                    
                    print(f"[{datetime.now()}] Email delivery disabled - ideas generated and stored in database")
                    
                    if email_sent:
                        stats.emails_sent += 1
                    
                    db.session.commit()
                    
                    print(f"[{datetime.now()}] Successfully generated {len(saved_ideas)} ideas and sent email: {email_sent}")
                    
            else:
                print("No Flask app context available for database operations")
                
        except Exception as e:
            print(f"[{datetime.now()}] Error in daily idea generation: {e}")
            import traceback
            traceback.print_exc()
    
    def process_validation_requests(self):
        """Process pending validation requests"""
        try:
            print(f"[{datetime.now()}] Processing validation requests...")
            
            if self.app:
                with self.app.app_context():
                    from src.models.business_idea import ValidationRequest
                    
                    # Get pending validation requests
                    pending_requests = ValidationRequest.query.filter_by(status='pending').all()
                    
                    for request in pending_requests:
                        try:
                            # Get the business idea
                            idea = BusinessIdea.query.get(request.business_idea_id)
                            if not idea:
                                continue
                            
                            # Generate deeper validation
                            validation_results = self.ai_service.generate_deeper_validation(idea.to_dict())
                            
                            # Send validation email
                            if request.user_email:
                                self.email_service.send_validation_results_email(
                                    idea.to_dict(),
                                    validation_results,
                                    request.user_email
                                )
                            
                            # Update request status
                            request.status = 'completed'
                            request.completed_at = datetime.utcnow()
                            
                        except Exception as e:
                            print(f"Error processing validation request {request.id}: {e}")
                            request.status = 'failed'
                    
                    db.session.commit()
                    print(f"[{datetime.now()}] Processed {len(pending_requests)} validation requests")
            
        except Exception as e:
            print(f"[{datetime.now()}] Error processing validation requests: {e}")
    
    def start_scheduler(self):
        """Start the background scheduler"""
        if self.running:
            print("Scheduler is already running")
            return
        
        self.running = True
        
        # Schedule daily idea generation at 9:00 AM UTC
        schedule.every().day.at("09:00").do(self.generate_and_send_daily_ideas)
        
        # Schedule validation processing every hour
        schedule.every().hour.do(self.process_validation_requests)
        
        # For testing: also allow manual trigger every 5 minutes (can be removed in production)
        # schedule.every(5).minutes.do(self.generate_and_send_daily_ideas)
        
        def run_scheduler():
            print(f"[{datetime.now()}] Scheduler started")
            print("Daily ideas scheduled for 9:00 AM UTC")
            print("Validation processing scheduled every hour")
            
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        print("Background scheduler started successfully")
    
    def stop_scheduler(self):
        """Stop the background scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        schedule.clear()
        print("Scheduler stopped")
    
    def trigger_daily_generation(self):
        """Manually trigger daily idea generation (for testing)"""
        print("Manually triggering daily idea generation...")
        self.generate_and_send_daily_ideas()
    
    def get_scheduler_status(self):
        """Get current scheduler status"""
        return {
            'running': self.running,
            'scheduled_jobs': len(schedule.jobs),
            'next_run': str(schedule.next_run()) if schedule.jobs else None,
            'jobs': [str(job) for job in schedule.jobs]
        }

