param(
    [string]$VenvPath = ".venv"
)

python -m venv $VenvPath
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvPath\Scripts\pip.exe" install -r requirements.txt
Copy-Item .env.example .env -ErrorAction SilentlyContinue
& "$VenvPath\Scripts\python.exe" main.py --init-db
