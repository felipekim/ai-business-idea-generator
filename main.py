import os
from flask import Flask, render_template_string, request, session, redirect, jsonify
import openai
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')
openai.api_key = os.getenv('OPENAI_API_KEY')

ideas = []

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('password') == os.getenv('ADMIN_PASSWORD', 'Welcome2081!'):
            session['auth'] = True
            return redirect('/')
    return '''<form method="post" style="max-width:400px;margin:50px auto;padding:20px;border:1px solid #ddd;">
    <h2>🤖 AI Business Ideas</h2>
    <input type="password" name="password" placeholder="Enter password" style="width:100%;padding:10px;margin:10px 0;">
    <button type="submit" style="width:100%;padding:10px;background:#007bff;color:white;border:none;">Login</button>
    </form>'''

@app.route('/')
def home():
    if not session.get('auth'):
        return redirect('/login')
    return f'''<div style="max-width:800px;margin:20px auto;padding:20px;">
    <h1>🤖 AI Business Ideas ({len(ideas)} total)</h1>
    <a href="/generate" style="background:#007bff;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;">Generate New Ideas</a>
    <div style="margin-top:20px;">
    {"".join([f"<div style='border:1px solid #ddd;margin:10px 0;padding:15px;'><h3>{i['title']}</h3><p>{i['summary']}</p></div>" for i in ideas[-5:]])}
    </div></div>'''

@app.route('/generate')
def generate():
    if not session.get('auth'):
        return redirect('/login')
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Generate a unique AI business idea with title and summary in JSON: {\"title\": \"...\", \"summary\": \"...\"}"}],
            max_tokens=200
        )
        idea = json.loads(response.choices[0].message.content)
        ideas.append(idea)
    except:
        ideas.append({"title": "AI Content Creator", "summary": "AI tool for automated content generation"})
    return redirect('/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
