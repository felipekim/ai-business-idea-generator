import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, date
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class EmailService:
    def __init__(self):
        self.smtp_host = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('EMAIL_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER', 'ideasaibusiness@gmail.com')
        self.email_password = os.getenv('EMAIL_PASSWORD', '')
        
    def create_daily_ideas_email(self, ideas: List[Dict[str, Any]], date_str: str) -> str:
        """Create HTML email content for daily business ideas"""
        
        # Email header
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Daily AI Business Ideas - {date_str}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8fafc;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 12px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: bold;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                    font-size: 16px;
                }}
                .idea-card {{
                    background: white;
                    border-radius: 12px;
                    padding: 25px;
                    margin-bottom: 25px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    border-left: 4px solid #667eea;
                }}
                .idea-title {{
                    font-size: 22px;
                    font-weight: bold;
                    color: #1a202c;
                    margin-bottom: 10px;
                }}
                .idea-summary {{
                    font-size: 16px;
                    color: #4a5568;
                    margin-bottom: 20px;
                    font-style: italic;
                }}
                .score-section {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                    padding: 15px;
                    background: #f7fafc;
                    border-radius: 8px;
                }}
                .total-score {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #667eea;
                    margin-right: 15px;
                }}
                .score-breakdown {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 10px;
                    font-size: 12px;
                }}
                .score-item {{
                    text-align: center;
                }}
                .score-value {{
                    font-weight: bold;
                    color: #2d3748;
                }}
                .details-section {{
                    margin-top: 20px;
                }}
                .detail-item {{
                    margin-bottom: 15px;
                }}
                .detail-label {{
                    font-weight: bold;
                    color: #2d3748;
                    margin-bottom: 5px;
                }}
                .detail-content {{
                    color: #4a5568;
                    line-height: 1.5;
                }}
                .financial-info {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                    padding: 15px;
                    background: #edf2f7;
                    border-radius: 8px;
                }}
                .financial-item {{
                    text-align: center;
                }}
                .financial-value {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #38a169;
                }}
                .financial-label {{
                    font-size: 12px;
                    color: #718096;
                    margin-top: 5px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    background: white;
                    border-radius: 12px;
                    color: #718096;
                    font-size: 14px;
                }}
                .cta-button {{
                    display: inline-block;
                    background: #667eea;
                    color: white;
                    padding: 12px 24px;
                    text-decoration: none;
                    border-radius: 6px;
                    font-weight: bold;
                    margin: 10px;
                }}
                @media (max-width: 600px) {{
                    body {{ padding: 10px; }}
                    .score-breakdown {{ grid-template-columns: repeat(2, 1fr); }}
                    .financial-info {{ grid-template-columns: 1fr; }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 Daily AI Business Ideas</h1>
                <p>{date_str} • 5 New Validated Ideas • Ready to Launch</p>
            </div>
        """
        
        # Add each idea
        for i, idea in enumerate(ideas, 1):
            scores = idea.get('scores', {})
            
            # Format currency
            def format_currency(amount):
                if amount >= 1000000:
                    return f"${amount/1000000:.1f}M"
                elif amount >= 1000:
                    return f"${amount/1000:.0f}K"
                else:
                    return f"${amount:,}"
            
            # Get score color
            def get_score_color(score):
                if score >= 8:
                    return "#38a169"  # green
                elif score >= 6:
                    return "#d69e2e"  # yellow
                else:
                    return "#e53e3e"  # red
            
            total_score = scores.get('total', 0)
            score_color = get_score_color(total_score)
            
            html_content += f"""
            <div class="idea-card">
                <div class="idea-title">#{i}. {idea.get('name', 'Untitled Idea')}</div>
                <div class="idea-summary">{idea.get('summary', '')}</div>
                
                <div class="score-section">
                    <div class="total-score" style="color: {score_color};">
                        {total_score:.1f}/10
                    </div>
                    <div class="score-breakdown">
                        <div class="score-item">
                            <div class="score-value">{scores.get('cost_to_build', 0):.1f}</div>
                            <div>Cost to Build</div>
                        </div>
                        <div class="score-item">
                            <div class="score-value">{scores.get('ease_of_implementation', 0):.1f}</div>
                            <div>Implementation</div>
                        </div>
                        <div class="score-item">
                            <div class="score-value">{scores.get('market_size', 0):.1f}</div>
                            <div>Market Size</div>
                        </div>
                        <div class="score-item">
                            <div class="score-value">{scores.get('competition_level', 0):.1f}</div>
                            <div>Competition</div>
                        </div>
                        <div class="score-item">
                            <div class="score-value">{scores.get('problem_severity', 0):.1f}</div>
                            <div>Problem Severity</div>
                        </div>
                        <div class="score-item">
                            <div class="score-value">{scores.get('founder_fit', 0):.1f}</div>
                            <div>Founder Fit</div>
                        </div>
                    </div>
                </div>
                
                <div class="financial-info">
                    <div class="financial-item">
                        <div class="financial-value">{format_currency(idea.get('launch_cost', 0))}</div>
                        <div class="financial-label">Launch Cost</div>
                    </div>
                    <div class="financial-item">
                        <div class="financial-value">{format_currency(idea.get('revenue_1_year', 0))}</div>
                        <div class="financial-label">1-Year Revenue</div>
                    </div>
                    <div class="financial-item">
                        <div class="financial-value">{format_currency(idea.get('revenue_5_year', 0))}</div>
                        <div class="financial-label">5-Year Revenue</div>
                    </div>
                </div>
                
                <div class="details-section">
                    <div class="detail-item">
                        <div class="detail-label">🎯 Target Audience</div>
                        <div class="detail-content">{idea.get('target_audience', '')}</div>
                    </div>
                    
                    <div class="detail-item">
                        <div class="detail-label">❗ Problem Solved</div>
                        <div class="detail-content">{idea.get('problem_solved', '')}</div>
                    </div>
                    
                    <div class="detail-item">
                        <div class="detail-label">🤖 AI Solution</div>
                        <div class="detail-content">{idea.get('ai_solution', '')}</div>
                    </div>
                    
                    <div class="detail-item">
                        <div class="detail-label">🛠️ Implementation</div>
                        <div class="detail-content">{idea.get('implementation', '')}</div>
                    </div>
                    
                    <div class="detail-item">
                        <div class="detail-label">💰 Revenue Model</div>
                        <div class="detail-content">{idea.get('revenue_model', '')}</div>
                    </div>
                </div>
            </div>
            """
        
        # Email footer
        html_content += f"""
            <div class="footer">
                <p><strong>AI Business Idea Generator</strong></p>
                <p>Automated daily delivery of validated AI business ideas</p>
                <p>Each idea is scored across 6 dimensions and validated with market evidence</p>
                <p>Perfect for solo, non-technical founders • Under $10K startup cost</p>
                
                <div style="margin-top: 20px;">
                    <a href="#" class="cta-button">View All Ideas</a>
                    <a href="#" class="cta-button">Request Validation</a>
                </div>
                
                <p style="margin-top: 20px; font-size: 12px; color: #a0aec0;">
                    Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M UTC')}<br>
                    This email was automatically generated by AI Business Idea Generator
                </p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def send_daily_ideas_email(self, ideas: List[Dict[str, Any]], recipient_email: str = None) -> bool:
        """Send daily business ideas email"""
        try:
            if not recipient_email:
                recipient_email = 'ideasaibusiness@gmail.com'
            
            # Create email content
            today = date.today().strftime('%B %d, %Y')
            html_content = self.create_daily_ideas_email(ideas, today)
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🧠 Daily AI Business Ideas - {today} ({len(ideas)} New Ideas)"
            msg['From'] = self.email_user
            msg['To'] = recipient_email
            
            # Create plain text version
            text_content = f"""
Daily AI Business Ideas - {today}

{len(ideas)} new validated AI business ideas perfect for solo, non-technical founders:

"""
            
            for i, idea in enumerate(ideas, 1):
                scores = idea.get('scores', {})
                text_content += f"""
{i}. {idea.get('name', 'Untitled')}
   Summary: {idea.get('summary', '')}
   Score: {scores.get('total', 0):.1f}/10
   Launch Cost: ${idea.get('launch_cost', 0):,}
   1-Year Revenue: ${idea.get('revenue_1_year', 0):,}
   
   Problem: {idea.get('problem_solved', '')}
   Solution: {idea.get('ai_solution', '')}
   
   ---
"""
            
            text_content += """
Generated by AI Business Idea Generator
Each idea is validated and scored across 6 dimensions
Perfect for solo founders with under $10K startup cost
"""
            
            # Attach both versions
            part1 = MIMEText(text_content, 'plain')
            part2 = MIMEText(html_content, 'html')
            
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            if self.email_password:  # Only send if password is configured
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.email_user, self.email_password)
                    server.send_message(msg)
                
                print(f"Daily ideas email sent successfully to {recipient_email}")
                return True
            else:
                print("Email password not configured, skipping email send")
                print(f"Would send email to {recipient_email} with {len(ideas)} ideas")
                return False
                
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def send_validation_results_email(self, idea: Dict[str, Any], validation_results: Dict[str, Any], recipient_email: str) -> bool:
        """Send deeper validation results email"""
        try:
            # Create email content for validation results
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Validation Results - {idea.get('name', 'Business Idea')}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background: #667eea; color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                    .content {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .section {{ margin-bottom: 20px; }}
                    .label {{ font-weight: bold; color: #2d3748; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🔍 Deeper Validation Results</h1>
                    <p>Comprehensive analysis for: {idea.get('name', 'Your Business Idea')}</p>
                </div>
                
                <div class="content">
                    <div class="section">
                        <div class="label">Business Idea:</div>
                        <p>{idea.get('summary', '')}</p>
                    </div>
                    
                    <div class="section">
                        <div class="label">Validation Score:</div>
                        <p>{validation_results.get('score', 'N/A')}/10</p>
                    </div>
                    
                    <div class="section">
                        <div class="label">Market Analysis:</div>
                        <p>{validation_results.get('market_analysis', 'Analysis pending...')}</p>
                    </div>
                    
                    <div class="section">
                        <div class="label">Recommendations:</div>
                        <p>{validation_results.get('recommendations', 'Recommendations will be provided...')}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Create and send email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🔍 Validation Results: {idea.get('name', 'Business Idea')}"
            msg['From'] = self.email_user
            msg['To'] = recipient_email
            
            msg.attach(MIMEText(html_content, 'html'))
            
            if self.email_password:
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.email_user, self.email_password)
                    server.send_message(msg)
                
                print(f"Validation results email sent to {recipient_email}")
                return True
            else:
                print("Email password not configured, skipping validation email")
                return False
                
        except Exception as e:
            print(f"Error sending validation email: {e}")
            return False

