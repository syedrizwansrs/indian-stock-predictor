#!/bin/bash

# Quick push script - run after creating GitHub repository
# Usage: ./push_to_github.sh YOUR_USERNAME

if [ $# -eq 0 ]; then
    echo "Usage: ./push_to_github.sh YOUR_GITHUB_USERNAME"
    echo "Example: ./push_to_github.sh johndoe"
    exit 1
fi

USERNAME=$1
REPO_URL="https://github.com/$USERNAME/indian-stock-predictor.git"

echo "🚀 Pushing to GitHub repository: $REPO_URL"
echo ""

# Add remote origin
git remote add origin $REPO_URL

# Rename branch to main (modern GitHub default)
git branch -M main

# Push to GitHub
git push -u origin main

echo ""
echo "✅ Successfully pushed to GitHub!"
echo "🔗 Your repository: https://github.com/$USERNAME/indian-stock-predictor"
echo ""
echo "🎉 Your Indian Stock Market Analysis & Prediction System is now live on GitHub!"
