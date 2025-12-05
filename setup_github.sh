#!/bin/bash

# Script to create GitHub repository and push code
# Usage: ./setup_github.sh <repository-name> [description]

REPO_NAME=${1:-"fso-beam-recovery-ao"}
DESCRIPTION=${2:-"FSO Beam Recovery with Adaptive Optics - Python simulation framework for OAM communication systems"}

echo "Creating GitHub repository: $REPO_NAME"
echo "Description: $DESCRIPTION"
echo ""

# Check if user is logged in to GitHub
if ! git config --get user.name > /dev/null 2>&1; then
    echo "Please configure git user:"
    echo "  git config --global user.name 'Your Name'"
    echo "  git config --global user.email 'your.email@example.com'"
    exit 1
fi

echo "Step 1: Create a new repository on GitHub:"
echo "  1. Go to https://github.com/new"
echo "  2. Repository name: $REPO_NAME"
echo "  3. Description: $DESCRIPTION"
echo "  4. Visibility: Public"
echo "  5. DO NOT initialize with README, .gitignore, or license"
echo "  6. Click 'Create repository'"
echo ""
read -p "Press Enter after you've created the repository on GitHub..."

echo ""
echo "Step 2: Adding remote and pushing..."
echo "Please enter your GitHub username:"
read GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "Error: GitHub username is required"
    exit 1
fi

# Add remote
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "Done! Repository is now available at:"
echo "https://github.com/$GITHUB_USERNAME/$REPO_NAME"

