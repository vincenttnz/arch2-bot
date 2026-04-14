cd F:\arch2

# 1. Wipe any existing Git tracking to prevent accidental massive commits
Remove-Item -Recurse -Force .git -ErrorAction SilentlyContinue

# 2. Generate a tailored .gitignore based on your repository structure
@"
# Massive Data & Image Folders
data/
datasets/
images/
labels/

# Virtual Environments & Caches
.venv/
venv/
SenseNova-SI/.venv/
__pycache__/
*.py[cod]

# AI Training Runs & Model Weights
runs/
models/
*.pt
*.pth
*.weights
*.onnx

# Logs & OS generated files
logs/
Thumbs.db
desktop.ini
"@ | Out-File -FilePath .gitignore -Encoding utf8

echo "Cleaned old Git and created custom .gitignore for F:\arch2!"

# 3. Initialize, stage, and commit
git init
git add .
git commit -m "Moved to F: drive - Clean push ignoring 35GB of data/venvs"
git branch -M main

# 4. Link to GitHub and force the push
git remote add origin https://github.com/vincenttnz/arch2-bot.git
git push -u origin main --force

echo "Push complete! Check https://github.com/vincenttnz/arch2-bot"