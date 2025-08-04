#!/bin/bash

# Gold AI Dashboard - Quick Deploy Script
# This script automates the entire deployment process

echo "🚀 Gold AI Dashboard - Quick Deploy"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Run the deployment automation
echo "📦 Running deployment automation..."
python3 deploy.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment files created successfully!"
    echo ""
    echo "🔥 Quick Deploy Options:"
    echo ""
    echo "1️⃣  Deploy to Streamlit Cloud:"
    echo "   - Go to https://share.streamlit.io"
    echo "   - Connect your GitHub account"
    echo "   - Create new app from the generated repository"
    echo ""
    echo "2️⃣  Deploy to Heroku:"
    echo "   cd gold-ai-dashboard-deployment"
    echo "   heroku create your-app-name"
    echo "   git push heroku main"
    echo ""
    echo "3️⃣  Deploy to Railway:"
    echo "   - Go to https://railway.app"
    echo "   - Connect GitHub and select the repository"
    echo "   - Deploy automatically"
    echo ""
    echo "4️⃣  Test locally with Docker:"
    echo "   cd gold-ai-dashboard-deployment"
    echo "   docker-compose up"
    echo ""
    echo "📁 All files are ready in: gold-ai-dashboard-deployment/"
    echo "📖 See DEPLOYMENT.md for detailed instructions"
else
    echo "❌ Deployment automation failed. Check the logs above."
    exit 1
fi
