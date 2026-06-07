FROM python:3.11-slim

LABEL maintainer="AI Trading System"
LABEL description="NSE/NIFTY50 AI Trading System"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create runtime directories
RUN mkdir -p data/historical data/live logs reports models backups

# Environment defaults (override at runtime)
ENV PYTHONPATH=/app \
    LOG_LEVEL=INFO \
    SIMULATION_MODE=true \
    BROKER_NAME=paper \
    STARTING_CAPITAL=100000

# Health check — verifies the settings module is importable
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from ai_trading_system.config.settings import get_settings; get_settings()" || exit 1

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "dashboard/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
