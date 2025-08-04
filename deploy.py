#!/usr/bin/env python3
"""
Gold AI Dashboard - Automated Deployment Script
This script automates the entire process of setting up and deploying the dashboard to git and cloud platforms.
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime

class GoldDashboardDeployer:
    def __init__(self, project_name="gold-ai-dashboard"):
        self.project_name = project_name
        self.current_dir = Path(__file__).parent
        self.deployment_dir = self.current_dir.parent / f"{project_name}-deployment"
        
        # Essential files for deployment
        self.essential_files = [
            'app.py',
            'config.yaml',
            'requirements.txt',
            'data_loader.py',
            'feature_engineer.py',
            'model_trainer.py',
            'predictor.py',
            'updater.py',
            'macro_factors.py',
            'strategy_analyzer.py',
            'README.md'
        ]
        
        # Directories to copy
        self.essential_dirs = [
            'data',
            'saved_models'
        ]
    
    def log(self, message, level="INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def run_command(self, command, cwd=None):
        """Run shell command and return result"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=cwd or self.deployment_dir,
                capture_output=True, 
                text=True, 
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {command}", "ERROR")
            self.log(f"Error: {e.stderr}", "ERROR")
            return None
    
    def create_deployment_directory(self):
        """Create clean deployment directory"""
        self.log("Creating deployment directory...")
        
        if self.deployment_dir.exists():
            shutil.rmtree(self.deployment_dir)
        
        self.deployment_dir.mkdir(parents=True)
        self.log(f"Created: {self.deployment_dir}")
    
    def copy_essential_files(self):
        """Copy all essential files to deployment directory"""
        self.log("Copying essential files...")
        
        # Copy files
        for file_name in self.essential_files:
            src = self.current_dir / file_name
            dst = self.deployment_dir / file_name
            
            if src.exists():
                shutil.copy2(src, dst)
                self.log(f"Copied: {file_name}")
            else:
                self.log(f"Warning: {file_name} not found", "WARN")
        
        # Copy directories
        for dir_name in self.essential_dirs:
            src = self.current_dir / dir_name
            dst = self.deployment_dir / dir_name
            
            if src.exists():
                shutil.copytree(src, dst)
                self.log(f"Copied directory: {dir_name}")
            else:
                self.log(f"Warning: {dir_name} directory not found", "WARN")
    
    def create_cloud_configs(self):
        """Create cloud deployment configurations"""
        self.log("Creating cloud deployment configurations...")
        
        # Streamlit Cloud config
        streamlit_config = """
[general]
email = ""

[server]
headless = true
enableCORS = false
port = $PORT

[theme]
primaryColor = "#FFD700"
backgroundColor = "#1E1E1E"
secondaryBackgroundColor = "#2D2D2D"
textColor = "#FFFFFF"
"""
        
        # Create .streamlit directory and config
        streamlit_dir = self.deployment_dir / ".streamlit"
        streamlit_dir.mkdir(exist_ok=True)
        
        with open(streamlit_dir / "config.toml", "w") as f:
            f.write(streamlit_config)
        
        # Heroku Procfile
        procfile_content = "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
        with open(self.deployment_dir / "Procfile", "w") as f:
            f.write(procfile_content)
        
        # Railway deployment config
        railway_config = {
            "build": {
                "builder": "NIXPACKS"
            },
            "deploy": {
                "startCommand": "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0",
                "restartPolicyType": "ON_FAILURE",
                "restartPolicyMaxRetries": 10
            }
        }
        
        with open(self.deployment_dir / "railway.json", "w") as f:
            json.dump(railway_config, f, indent=2)
        
        # Docker configuration
        dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p data saved_models

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
        
        with open(self.deployment_dir / "Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        # Docker Compose for local testing
        docker_compose_content = """version: '3.8'

services:
  gold-dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./data:/app/data
      - ./saved_models:/app/saved_models
    restart: unless-stopped
"""
        
        with open(self.deployment_dir / "docker-compose.yml", "w") as f:
            f.write(docker_compose_content)
        
        self.log("Created cloud deployment configurations")
    
    def create_github_actions(self):
        """Create GitHub Actions workflow for CI/CD"""
        self.log("Creating GitHub Actions workflow...")
        
        github_dir = self.deployment_dir / ".github" / "workflows"
        github_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_content = """name: Deploy Gold AI Dashboard

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Test imports
      run: |
        python -c "import streamlit; import pandas; import numpy; import xgboost; print('All imports successful')"
    
    - name: Check app syntax
      run: |
        python -m py_compile app.py
        python -m py_compile data_loader.py
        python -m py_compile predictor.py

  deploy-streamlit:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to Streamlit Cloud
      run: |
        echo "Deployment to Streamlit Cloud triggered"
        # Add your Streamlit Cloud deployment commands here
"""
        
        with open(github_dir / "deploy.yml", "w") as f:
            f.write(workflow_content)
        
        self.log("Created GitHub Actions workflow")
    
    def create_deployment_readme(self):
        """Create comprehensive deployment README"""
        self.log("Creating deployment README...")
        
        readme_content = f"""# Gold AI Dashboard - Cloud Deployment

This repository contains the automated deployment setup for the Gold AI Trading Dashboard.

## 🚀 Quick Deploy Options

### Option 1: Streamlit Cloud (Recommended)
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from this repository
5. Set main file as `app.py`

### Option 2: Heroku
```bash
# Install Heroku CLI first
heroku create {self.project_name}
git push heroku main
```

### Option 3: Railway
1. Go to [railway.app](https://railway.app)
2. Connect GitHub repository
3. Deploy automatically with railway.json config

### Option 4: Docker (Any Cloud Provider)
```bash
# Build and run locally
docker build -t gold-dashboard .
docker run -p 8501:8501 gold-dashboard

# Or use docker-compose
docker-compose up
```

## 📋 Environment Variables

Set these environment variables in your cloud platform:

```
PYTHONUNBUFFERED=1
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🔧 Local Development

```bash
# Clone repository
git clone <your-repo-url>
cd {self.project_name}

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## 📁 Project Structure

```
{self.project_name}/
├── app.py                 # Main Streamlit application
├── config.yaml           # Configuration settings
├── requirements.txt       # Python dependencies
├── data_loader.py        # Data fetching and processing
├── feature_engineer.py   # Technical indicators
├── model_trainer.py      # ML model training
├── predictor.py          # Prediction engine
├── macro_factors.py      # Macroeconomic factors
├── strategy_analyzer.py  # Trading strategies
├── updater.py            # Automated updates
├── data/                 # Data storage
├── saved_models/         # Trained models
├── .streamlit/           # Streamlit configuration
├── .github/workflows/    # CI/CD workflows
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── Procfile              # Heroku configuration
└── railway.json          # Railway configuration
```

## 🔄 Automated Updates

The dashboard includes automated daily updates:
- Fetches latest gold price data
- Retrains ML models
- Updates predictions
- Logs all activities

## 🛠️ Troubleshooting

### Common Issues:
1. **Memory Issues**: Reduce model complexity in `config.yaml`
2. **API Limits**: Check data source rate limits
3. **Dependencies**: Ensure all packages in `requirements.txt` are compatible

### Logs:
Check application logs in your cloud platform for debugging.

## 📊 Features

- ✅ Real-time gold price predictions
- ✅ 32+ technical indicators
- ✅ Macroeconomic factor analysis
- ✅ Interactive charts and visualizations
- ✅ Model performance monitoring
- ✅ Automated daily updates
- ✅ Mobile-responsive design

## 🔐 Security Notes

- No API keys are hardcoded
- All sensitive data should be set as environment variables
- Regular security updates included

## 📞 Support

For issues or questions:
1. Check the logs in your deployment platform
2. Review the troubleshooting section above
3. Create an issue in this repository

---

**Generated on**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Deployment Script Version**: 1.0
"""
        
        with open(self.deployment_dir / "DEPLOYMENT.md", "w") as f:
            f.write(readme_content)
        
        self.log("Created deployment README")
    
    def initialize_git_repository(self):
        """Initialize git repository and make initial commit"""
        self.log("Initializing git repository...")
        
        # Initialize git
        self.run_command("git init")
        
        # Create .gitignore
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
.env

# Data files
*.csv
*.json
*.pkl
*.joblib

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp

# Model files (keep structure, ignore large files)
saved_models/*.pkl
saved_models/*.joblib
saved_models/*.model

# Keep directory structure
!saved_models/.gitkeep
!data/.gitkeep
"""
        
        with open(self.deployment_dir / ".gitignore", "w") as f:
            f.write(gitignore_content)
        
        # Create .gitkeep files for empty directories
        (self.deployment_dir / "data" / ".gitkeep").touch()
        (self.deployment_dir / "saved_models" / ".gitkeep").touch()
        
        # Add all files
        self.run_command("git add .")
        
        # Make initial commit
        commit_message = f"Initial deployment setup for Gold AI Dashboard - {datetime.now().strftime('%Y-%m-%d')}"
        self.run_command(f'git commit -m "{commit_message}"')
        
        self.log("Git repository initialized with initial commit")
    
    def create_setup_script(self):
        """Create automated setup script for cloud platforms"""
        self.log("Creating setup script...")
        
        setup_script = """#!/bin/bash

# Gold AI Dashboard - Automated Setup Script
echo "🚀 Setting up Gold AI Dashboard..."

# Create necessary directories
mkdir -p data saved_models logs

# Set permissions
chmod +x *.py

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Initialize data if needed
echo "📊 Initializing data..."
python -c "
import os
from data_loader import GoldDataLoader
from model_trainer import GoldModelTrainer

# Create initial data and model if they don't exist
if not os.path.exists('data/gold_data.csv'):
    print('Fetching initial data...')
    loader = GoldDataLoader()
    loader.fetch_and_save_data()

if not os.path.exists('saved_models/gold_model.pkl'):
    print('Training initial model...')
    trainer = GoldModelTrainer()
    trainer.train_and_save_model()

print('✅ Setup complete!')
"

echo "✅ Gold AI Dashboard setup completed!"
echo "🌐 Run: streamlit run app.py"
"""
        
        with open(self.deployment_dir / "setup.sh", "w") as f:
            f.write(setup_script)
        
        # Make script executable
        os.chmod(self.deployment_dir / "setup.sh", 0o755)
        
        self.log("Created setup script")
    
    def deploy(self):
        """Main deployment function"""
        try:
            self.log("🚀 Starting Gold AI Dashboard deployment automation...")
            
            # Step 1: Create deployment directory
            self.create_deployment_directory()
            
            # Step 2: Copy essential files
            self.copy_essential_files()
            
            # Step 3: Create cloud configurations
            self.create_cloud_configs()
            
            # Step 4: Create GitHub Actions
            self.create_github_actions()
            
            # Step 5: Create deployment documentation
            self.create_deployment_readme()
            
            # Step 6: Create setup script
            self.create_setup_script()
            
            # Step 7: Initialize git repository
            self.initialize_git_repository()
            
            self.log("✅ Deployment automation completed successfully!")
            self.log(f"📁 Deployment files created in: {self.deployment_dir}")
            self.log("\n🔥 Next Steps:")
            self.log("1. cd into the deployment directory")
            self.log("2. Create a new GitHub repository")
            self.log("3. Add remote: git remote add origin <your-repo-url>")
            self.log("4. Push: git push -u origin main")
            self.log("5. Deploy to your preferred cloud platform")
            self.log("\n📖 See DEPLOYMENT.md for detailed instructions")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Deployment failed: {str(e)}", "ERROR")
            return False

if __name__ == "__main__":
    deployer = GoldDashboardDeployer()
    success = deployer.deploy()
    sys.exit(0 if success else 1)
