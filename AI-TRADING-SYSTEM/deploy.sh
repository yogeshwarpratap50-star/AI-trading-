#!/usr/bin/env bash
# ============================================================
# VPS Deployment Script — Phase 10B
# Supports: Ubuntu/Debian (Linux), any VPS provider
# ============================================================
set -e

APP_DIR="/opt/ai-trading-system"
SERVICE_NAME="ai-trading"

echo "=== AI Trading System — VPS Deployment ==="

# 1. System packages
echo ">>> Installing system packages..."
apt-get update -qq
apt-get install -y python3.11 python3.11-venv python3-pip git curl sqlite3 supervisor

# 2. Clone or update repository
if [ -d "$APP_DIR/.git" ]; then
  echo ">>> Updating existing repo..."
  cd "$APP_DIR" && git pull
else
  echo ">>> Cloning repo..."
  git clone . "$APP_DIR"
  cd "$APP_DIR"
fi

# 3. Virtual environment
echo ">>> Setting up Python environment..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Environment file
if [ ! -f ".env" ]; then
  cp .env.example .env 2>/dev/null || cat > .env << 'EOF'
SIMULATION_MODE=true
BROKER_NAME=paper
STARTING_CAPITAL=100000
LOG_LEVEL=INFO
EOF
fi

# 5. Directory setup
mkdir -p data/historical data/live logs reports models backups

# 6. Supervisor config (process management + auto restart)
cat > /etc/supervisor/conf.d/${SERVICE_NAME}.conf << EOF
[program:${SERVICE_NAME}-dashboard]
command=${APP_DIR}/venv/bin/python -m streamlit run dashboard/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
directory=${APP_DIR}
user=root
autostart=true
autorestart=true
stderr_logfile=${APP_DIR}/logs/dashboard.err.log
stdout_logfile=${APP_DIR}/logs/dashboard.out.log
environment=PYTHONPATH="${APP_DIR}",SIMULATION_MODE="true"

[program:${SERVICE_NAME}-engine]
command=${APP_DIR}/venv/bin/python main.py --paper-trade --symbol RELIANCE.NS
directory=${APP_DIR}
user=root
autostart=true
autorestart=true
startsecs=10
stderr_logfile=${APP_DIR}/logs/engine.err.log
stdout_logfile=${APP_DIR}/logs/engine.out.log
environment=PYTHONPATH="${APP_DIR}",SIMULATION_MODE="true"
EOF

supervisorctl reread
supervisorctl update
supervisorctl restart all

echo ""
echo "=== Deployment Complete ==="
echo "Dashboard: http://$(curl -s ifconfig.me):8501"
echo "Logs:      $APP_DIR/logs/"
echo "Status:    supervisorctl status"
