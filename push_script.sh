#!/bin/bash
set -e

# Get the GitHub token from environment variable
TOKEN="${GITHUB_TOKEN}"

# Setup repository
REPO_OWNER="stealthg0dd"
REPO_NAME="neufin"
REPO_URL="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}"

# Check if repository exists
echo "Checking if repository exists..."
REPO_EXISTS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: token ${TOKEN}" "${REPO_URL}")

# If repository doesn't exist, create it
if [ "${REPO_EXISTS}" != "200" ]; then
  echo "Repository doesn't exist. Creating it..."
  curl -s -H "Authorization: token ${TOKEN}" \
       -d '{"name":"'${REPO_NAME}'", "description":"Neufin - Advanced Market Sentiment Analysis Platform", "private":false}' \
       https://api.github.com/user/repos
fi

# Configure Git
git config --global user.name "Replit User"
git config --global user.email "replit@example.com"

# Clean previous remotes
git remote remove origin 2>/dev/null || true

# Add the new remote with token
git remote add origin "https://${TOKEN}@github.com/${REPO_OWNER}/${REPO_NAME}.git"

# Push to GitHub
echo "Pushing code to GitHub..."
git push -u origin main
