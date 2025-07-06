# Railway Deployment Guide

## 🚀 Deploy AI Business Idea Generator to Railway

### Prerequisites
- GitHub account
- Railway account (free tier available)
- This repository code

### Step 1: Prepare GitHub Repository

1. **Create new repository on GitHub:**
   - Go to github.com and create a new repository
   - Name it: `ai-business-idea-generator`
   - Make it public or private (your choice)
   - Don't initialize with README (we have our own)

2. **Push code to GitHub:**
   ```bash
   cd ai-business-idea-generator
   git remote add origin https://github.com/YOUR_USERNAME/ai-business-idea-generator.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Railway

1. **Sign up/Login to Railway:**
   - Go to https://railway.app
   - Sign up with GitHub account (recommended)

2. **Create New Project:**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your `ai-business-idea-generator` repository

3. **Configure Environment Variables:**
   Go to your project → Variables tab and add:
   ```
   OPENAI_API_KEY=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
   ADMIN_PASSWORD=Welcome2081!
   FLASK_SECRET_KEY=your-super-secret-key-here-change-this
   DEBUG=False
   START_SCHEDULER=True
   PORT=5000
   ```

4. **Deploy:**
   - Railway will automatically detect the Python app
   - It will use the `nixpacks.toml` configuration
   - Build process will install dependencies from `requirements.txt`
   - App will start with `python src/main.py`

### Step 3: Access Your Application

1. **Get Railway URL:**
   - Go to your project dashboard
   - Click on "Deployments" tab
   - Copy the generated Railway URL (e.g., `https://your-app.railway.app`)

2. **Test Login:**
   - Visit your Railway URL
   - You should see the login page
   - Enter password: `Welcome2081!`
   - Access the AI Business Ideas dashboard

### Step 4: Custom Domain (Optional)

1. **Add Custom Domain:**
   - Go to project → Settings → Domains
   - Add your custom domain
   - Configure DNS records as shown

### Environment Variables Explained

- **OPENAI_API_KEY**: Your OpenAI API key for AI idea generation
- **ADMIN_PASSWORD**: Password for accessing the application (Welcome2081!)
- **FLASK_SECRET_KEY**: Secret key for Flask sessions (generate a secure one)
- **DEBUG**: Set to False for production
- **START_SCHEDULER**: Enable daily idea generation
- **PORT**: Railway will set this automatically, but we specify 5000

### Troubleshooting

**Build Fails:**
- Check that all files are committed to GitHub
- Verify requirements.txt has all dependencies
- Check Railway build logs for specific errors

**App Won't Start:**
- Verify environment variables are set correctly
- Check that OPENAI_API_KEY is valid
- Review Railway deployment logs

**Can't Access Login Page:**
- Ensure PORT environment variable is set
- Check that Railway URL is correct
- Verify app is running in Railway dashboard

### Features After Deployment

✅ **Password Protected**: Login required with Welcome2081!
✅ **Daily AI Ideas**: 5 new business ideas generated daily at 9:00 AM UTC
✅ **Web Interface**: Modern, responsive dashboard
✅ **Scoring System**: 6-dimension scoring for each idea
✅ **No Email**: Email delivery disabled as requested
✅ **Persistent Storage**: SQLite database for idea storage
✅ **Admin Controls**: Manual idea generation and system monitoring

### Security Notes

- Change FLASK_SECRET_KEY to a secure random string
- Consider rotating ADMIN_PASSWORD periodically
- Monitor OpenAI API usage and costs
- Railway provides HTTPS automatically

### Support

If you encounter issues:
1. Check Railway deployment logs
2. Verify all environment variables are set
3. Ensure GitHub repository is up to date
4. Test locally first with same environment variables

---

**Your AI Business Idea Generator will be live and generating ideas daily!**

