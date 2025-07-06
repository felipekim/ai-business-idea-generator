# AI Business Idea Generator & Validator

A fully automated AI-powered web application that generates and validates 5 new AI business ideas daily, displays them on a website, and emails them automatically.

## 🚀 Features

### Core Functionality
- **Daily AI Idea Generation**: Automatically generates 5 new AI business ideas every day at 9:00 AM UTC
- **Comprehensive Scoring**: Each idea scored across 6 dimensions (Cost to Build, Implementation Ease, Market Size, Competition, Problem Severity, Founder Fit)
- **Automated Email Delivery**: Beautiful HTML emails sent daily to ideasaibusiness@gmail.com
- **Web Interface**: Modern, responsive website for browsing and filtering ideas
- **Validation System**: Request deeper validation for up to 3 ideas per day

### AI-Powered Features
- **OpenAI GPT-4 Integration**: High-quality idea generation and validation
- **Market Trend Analysis**: Real-time validation using web search and market data
- **Scoring Algorithm**: Intelligent scoring based on VC evaluation criteria
- **Evidence Collection**: Automated gathering of validation evidence

### User Interface
- **Modern Design**: Professional gradient design with responsive layout
- **Advanced Filtering**: Search, filter by niche, score, and sort options
- **Idea Cards**: Expandable cards with complete business details
- **Statistics Dashboard**: Track total ideas, validations, and performance
- **Mobile Responsive**: Works perfectly on all devices

## 🏗️ Architecture

### Backend (Flask)
- **API Endpoints**: RESTful API for ideas, validation, and admin functions
- **Database**: SQLite with SQLAlchemy ORM
- **AI Services**: OpenAI API integration with custom prompts
- **Email System**: SMTP-based email automation with HTML templates
- **Scheduler**: Background task scheduler for daily automation

### Frontend (React)
- **Modern Stack**: React + Vite + Tailwind CSS + shadcn/ui
- **Components**: Reusable UI components with professional design
- **State Management**: React hooks for API integration
- **Responsive Design**: Mobile-first design approach

### Automation
- **Daily Scheduling**: Automated idea generation at 9:00 AM UTC
- **Email Templates**: Beautiful HTML emails with complete idea details
- **Background Processing**: Validation requests processed hourly
- **Admin Controls**: Manual triggers and system monitoring

## 📋 Requirements

### System Requirements
- Python 3.11+
- Node.js 20+
- Git

### API Keys Required
- **OpenAI API Key**: For AI idea generation and validation
- **Email Credentials**: Gmail SMTP for email delivery (optional for testing)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd ai-business-idea-generator
```

### 2. Backend Setup
```bash
# Install Python dependencies
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key and email credentials
```

### 3. Frontend Setup
```bash
# Install Node.js dependencies
cd frontend
npm install

# Build for production
npm run build

# Copy built files to Flask static directory
cp -r dist/* ../src/static/
```

### 4. Run Application
```bash
# Start the Flask backend (includes frontend)
cd backend
source venv/bin/activate
python src/main.py
```

The application will be available at `http://localhost:5000`

## 🔧 Configuration

### Environment Variables (.env)
```env
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Email Configuration
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Flask Configuration
FLASK_SECRET_KEY=your_secret_key
DEBUG=False
START_SCHEDULER=True
```

### Email Setup (Optional)
To enable email delivery:
1. Create a Gmail App Password
2. Add email credentials to `.env`
3. Restart the application

## 📡 API Endpoints

### Ideas API
- `GET /api/ideas` - Get business ideas with filtering
- `GET /api/ideas/{id}` - Get specific idea
- `POST /api/ideas/generate` - Generate new ideas
- `POST /api/ideas/{id}/validate` - Request deeper validation
- `GET /api/ideas/today` - Get today's featured ideas
- `GET /api/ideas/niches` - Get available niches
- `GET /api/stats` - Get platform statistics

### Admin API
- `GET /api/admin/scheduler/status` - Check scheduler status
- `POST /api/admin/scheduler/start` - Start scheduler
- `POST /api/admin/scheduler/stop` - Stop scheduler
- `POST /api/admin/scheduler/trigger` - Manually trigger idea generation
- `POST /api/admin/email/test` - Test email functionality
- `GET /api/admin/system/info` - Get system information

## 🎯 Business Idea Criteria

Each generated idea meets these criteria:
- **AI-Powered Solution**: Uses artificial intelligence as core technology
- **Low Startup Cost**: Under $10,000 total launch cost
- **Solo Founder Friendly**: Can be built by non-technical founders
- **Market Validated**: Backed by real market evidence and trends
- **Clear Problem**: Solves a specific, validated problem
- **Revenue Model**: Defined path to profitability

## 📊 Scoring System

Ideas are scored 1-10 across 6 dimensions:

1. **Cost to Build** (10 = very low cost)
2. **Ease of Implementation** (10 = very easy)
3. **Market Size** (10 = very large market)
4. **Competition Level** (10 = low competition)
5. **Problem Severity** (10 = critical problem)
6. **Founder Fit** (10 = perfect for solo founders)

**Total Score**: Weighted average of all dimensions

## 🔄 Daily Automation

The system automatically:
1. **9:00 AM UTC**: Generates 5 new business ideas
2. **Every Hour**: Processes validation requests
3. **Daily Email**: Sends formatted ideas to configured email
4. **Database Storage**: Saves all ideas with full details
5. **Statistics Update**: Tracks generation and email metrics

## 🛠️ Development

### Project Structure
```
ai-business-idea-generator/
├── src/                    # Flask backend
│   ├── models/            # Database models
│   ├── routes/            # API endpoints
│   ├── services/          # Business logic
│   ├── static/            # Built frontend files
│   └── main.py           # Application entry point
├── frontend/              # React frontend source
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── hooks/         # Custom hooks
│   │   └── App.jsx       # Main app component
│   └── package.json
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── README.md
```

### Running in Development Mode

**Backend:**
```bash
cd backend
source venv/bin/activate
export DEBUG=True
python src/main.py
```

**Frontend (separate terminal):**
```bash
cd frontend
npm run dev
```

### Testing

**Test Idea Generation:**
```bash
curl -X POST http://localhost:5000/api/ideas/generate \
  -H "Content-Type: application/json" \
  -d '{"count": 2}'
```

**Test Email System:**
```bash
curl -X POST http://localhost:5000/api/admin/email/test \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com"}'
```

## 🚀 Deployment

### Production Deployment

1. **Prepare Environment:**
   ```bash
   export DEBUG=False
   export START_SCHEDULER=True
   ```

2. **Build Frontend:**
   ```bash
   cd frontend
   npm run build
   cp -r dist/* ../src/static/
   ```

3. **Deploy Backend:**
   ```bash
   cd backend
   python src/main.py
   ```

### Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "src/main.py"]
```

## 📈 Monitoring

### Scheduler Status
Check scheduler status via API:
```bash
curl http://localhost:5000/api/admin/scheduler/status
```

### System Information
Get system info:
```bash
curl http://localhost:5000/api/admin/system/info
```

### Logs
Monitor application logs for:
- Daily idea generation
- Email delivery status
- Validation processing
- Error tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section below
2. Review API documentation
3. Check application logs
4. Create an issue in the repository

## 🔧 Troubleshooting

### Common Issues

**Email not sending:**
- Verify EMAIL_PASSWORD is set correctly
- Check Gmail App Password configuration
- Ensure SMTP settings are correct

**Ideas not generating:**
- Verify OPENAI_API_KEY is valid
- Check API rate limits
- Review application logs

**Frontend not loading:**
- Ensure frontend is built: `npm run build`
- Check static files are copied to Flask directory
- Verify Flask is serving static files

**Scheduler not running:**
- Check START_SCHEDULER=True in .env
- Verify scheduler status via API
- Review scheduler logs

### Debug Mode
Enable debug mode for detailed logging:
```bash
export DEBUG=True
python src/main.py
```

## 🎉 Success Metrics

The application successfully:
- ✅ Generates 5 AI business ideas daily
- ✅ Scores ideas across 6 dimensions
- ✅ Sends automated daily emails
- ✅ Provides web interface for browsing
- ✅ Handles validation requests
- ✅ Maintains idea archive
- ✅ Runs fully automated

---

**Built with ❤️ using AI-powered development**

