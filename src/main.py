import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory, request, session, redirect, url_for, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv
from src.models.user import db
from src.models.business_idea import BusinessIdea, ValidationRequest, DailyStats
from src.routes.user import user_bp
from src.routes.ideas import ideas_bp
from src.routes.admin import admin_bp, init_scheduler
from src.auth import login_required, check_auth, ADMIN_PASSWORD, LOGIN_TEMPLATE

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'asdf#FGSgvasgf$5$WGT')

# Enable CORS for all routes
CORS(app, origins="*")

# Register blueprints with login protection
app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(ideas_bp, url_prefix='/api')
app.register_blueprint(admin_bp, url_prefix='/api/admin')

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Create all database tables
with app.app_context():
    db.create_all()

# Initialize scheduler
scheduler_service = init_scheduler(app)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        password = request.form.get('password')
        if password == ADMIN_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('serve', path=''))
        else:
            return render_template_string(LOGIN_TEMPLATE, error="Invalid password. Please try again.")
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.pop('authenticated', None)
    return redirect(url_for('login'))

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
@login_required
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404

# Protect API routes
@app.before_request
def require_login():
    """Require login for all routes except login page"""
    if request.endpoint and request.endpoint != 'login':
        if not check_auth():
            if request.path.startswith('/api/'):
                return {'error': 'Authentication required'}, 401
            return redirect(url_for('login'))

if __name__ == '__main__':
    # Start the scheduler when the app starts (but disable email)
    if os.getenv('START_SCHEDULER', 'True').lower() == 'true':
        scheduler_service.start_scheduler()
        print("Scheduler started automatically (email delivery disabled)")
    
    app.run(host='0.0.0.0', port=5000, debug=os.getenv('DEBUG', 'False').lower() == 'true')
