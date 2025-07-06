from functools import wraps
from flask import request, session, redirect, url_for, render_template_string, flash
import os

# Simple password from environment or default
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'Welcome2081!')

def login_required(f):
    """Decorator to require login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def check_auth():
    """Check if user is authenticated"""
    return session.get('authenticated', False)

# Simple login page template
LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Business Ideas - Login</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-container {
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            width: 100%;
            max-width: 400px;
            text-align: center;
        }
        .logo {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
            text-align: left;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #333;
        }
        input[type="password"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 6px;
            font-size: 16px;
            transition: border-color 0.3s;
            box-sizing: border-box;
        }
        input[type="password"]:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            width: 100%;
            padding: 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background: #5a6fd8;
        }
        .error {
            color: #e53e3e;
            margin-top: 10px;
            font-size: 14px;
        }
        .features {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e1e5e9;
            text-align: left;
        }
        .feature {
            margin-bottom: 8px;
            color: #666;
            font-size: 14px;
        }
        .feature::before {
            content: "✓ ";
            color: #38a169;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">🧠 AI Business Ideas</div>
        <div class="subtitle">Daily AI-Powered Business Concepts</div>
        
        <form method="POST">
            <div class="form-group">
                <label for="password">Access Password</label>
                <input type="password" id="password" name="password" required 
                       placeholder="Enter access password">
            </div>
            <button type="submit" class="btn">Access Dashboard</button>
        </form>
        
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        
        <div class="features">
            <div class="feature">5 new AI business ideas daily</div>
            <div class="feature">Comprehensive scoring system</div>
            <div class="feature">Market validation evidence</div>
            <div class="feature">Under $10K startup costs</div>
            <div class="feature">Solo founder friendly</div>
        </div>
    </div>
</body>
</html>
'''

