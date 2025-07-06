from flask import Blueprint, request, jsonify
from datetime import datetime
from src.services.scheduler_service import SchedulerService

admin_bp = Blueprint('admin', __name__)

# Global scheduler instance (will be initialized in main.py)
scheduler_service = None

def init_scheduler(app):
    """Initialize the scheduler service with Flask app context"""
    global scheduler_service
    scheduler_service = SchedulerService(app)
    return scheduler_service

@admin_bp.route('/scheduler/status', methods=['GET'])
def get_scheduler_status():
    """Get current scheduler status"""
    try:
        if not scheduler_service:
            return jsonify({'error': 'Scheduler not initialized'}), 500
        
        status = scheduler_service.get_scheduler_status()
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/scheduler/start', methods=['POST'])
def start_scheduler():
    """Start the scheduler"""
    try:
        if not scheduler_service:
            return jsonify({'error': 'Scheduler not initialized'}), 500
        
        scheduler_service.start_scheduler()
        return jsonify({'message': 'Scheduler started successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/scheduler/stop', methods=['POST'])
def stop_scheduler():
    """Stop the scheduler"""
    try:
        if not scheduler_service:
            return jsonify({'error': 'Scheduler not initialized'}), 500
        
        scheduler_service.stop_scheduler()
        return jsonify({'message': 'Scheduler stopped successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/scheduler/trigger', methods=['POST'])
def trigger_daily_generation():
    """Manually trigger daily idea generation"""
    try:
        if not scheduler_service:
            return jsonify({'error': 'Scheduler not initialized'}), 500
        
        scheduler_service.trigger_daily_generation()
        return jsonify({'message': 'Daily idea generation triggered successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/email/test', methods=['POST'])
def test_email():
    """Test email functionality"""
    try:
        data = request.get_json() or {}
        recipient = data.get('email', 'ideasaibusiness@gmail.com')
        
        # Create test ideas
        test_ideas = [
            {
                'name': 'Test AI Business Idea',
                'summary': 'This is a test email to verify the email system is working correctly.',
                'target_audience': 'Test users',
                'problem_solved': 'Testing email delivery',
                'ai_solution': 'Automated email system',
                'implementation': 'Flask backend with email service',
                'revenue_model': 'Test model',
                'launch_cost': 5000,
                'revenue_1_year': 50000,
                'revenue_5_year': 250000,
                'niche': 'Testing',
                'scores': {
                    'cost_to_build': 8.0,
                    'ease_of_implementation': 9.0,
                    'market_size': 7.0,
                    'competition_level': 6.0,
                    'problem_severity': 8.0,
                    'founder_fit': 9.0,
                    'total': 7.8
                }
            }
        ]
        
        if scheduler_service:
            email_sent = scheduler_service.email_service.send_daily_ideas_email(test_ideas, recipient)
            
            if email_sent:
                return jsonify({'message': f'Test email sent successfully to {recipient}'})
            else:
                return jsonify({'message': f'Email system configured but not sent (check email credentials)'})
        else:
            return jsonify({'error': 'Scheduler not initialized'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/system/info', methods=['GET'])
def get_system_info():
    """Get system information"""
    try:
        import os
        from datetime import datetime
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'environment': {
                'openai_api_configured': bool(os.getenv('OPENAI_API_KEY')),
                'email_configured': bool(os.getenv('EMAIL_PASSWORD')),
                'debug_mode': os.getenv('DEBUG', 'False').lower() == 'true'
            },
            'scheduler': scheduler_service.get_scheduler_status() if scheduler_service else {'running': False},
            'version': '1.0.0'
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

