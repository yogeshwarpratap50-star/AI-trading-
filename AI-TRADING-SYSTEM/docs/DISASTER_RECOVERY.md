# Disaster Recovery Runbook
**AI Trading System — Production**

---

## 1. Overview

The disaster recovery system provides automated backup and restore for all
stateful components: SQLite database, trained models, and trade journals.

| Component | Backup method | Recovery time |
|-----------|--------------|--------------|
| Database (SQLite) | Hot-backup via sqlite3 API + integrity check | < 30 s |
| ML Models | File copy of `.pkl` + `model_registry.json` | < 10 s |
| Trade Journal (CSV) | File copy + multi-backup deduplication merge | < 5 s |
| System logs | File copy | < 5 s |
| Config (`.env`, `requirements.txt`) | File copy | < 2 s |

---

## 2. Backup

### 2.1 Automatic (scheduled)

Add to crontab (Linux/VPS) — runs daily at 2 AM:

```bash
0 2 * * * cd /app && python -c "
from monitoring.disaster_recovery import DisasterRecovery
dr = DisasterRecovery()
m = dr.backup_all()
print('Backup OK:', m.backup_id)
" >> /var/log/ai-trading-backup.log 2>&1
```

### 2.2 Manual (CLI)

```bash
python main.py --backup
```

### 2.3 Python API

```python
from monitoring.disaster_recovery import DisasterRecovery

dr = DisasterRecovery(
    source_root=".",
    backup_root="backups",
    db_path="database/trading.db",
    model_dir="models/saved",
    trade_journal_path="reports/trade_journal.csv",
    log_dir="logs",
)

manifest = dr.backup_all()
print(f"Backup {manifest.backup_id} created with {len(manifest.files)} files")
```

Backup sets are stored in `backups/<YYYYMMDD_HHMMSS>/`:

```
backups/
  20250607_020000/
    manifest.json         ← index of all files + integrity flags
    trading.db            ← SQLite hot-backup
    registry.json         ← model registry
    models/               ← all .pkl model files
    logs/
      trade_journal.csv
      trading.log
```

---

## 3. Restore

### 3.1 Automatic (latest backup)

```python
from monitoring.disaster_recovery import DisasterRecovery
dr = DisasterRecovery()
result = dr.restore_latest()
print(result.to_dict())
```

### 3.2 Specific backup

```python
result = dr.restore("20250607_020000")
if not result.success:
    print("Errors:", result.errors)
```

### 3.3 Dry run (preview without writing)

```python
result = dr.restore_latest(dry_run=True)
print("Would restore:", result.restored)
```

### 3.4 Restore sequence (what happens)

1. Locate backup directory
2. Verify database backup integrity via `PRAGMA integrity_check`
3. Overwrite `database/trading.db`
4. Replace `models/saved/` with backup copy
5. Restore `models/model_registry.json`
6. Restore `reports/trade_journal.csv`
7. Re-run `DatabaseInitializer.initialize()` (idempotent — safe to repeat)

---

## 4. Trade Log Recovery

If the live trade journal is corrupted, recover by merging all backup copies:

```python
from monitoring.disaster_recovery import DisasterRecovery
dr = DisasterRecovery()
records = dr.recover_trade_logs()
print(f"Recovered {len(records)} unique trades")
# Written to: reports/trade_journal_recovered.csv
```

**What it does:**
- Scans every backup set for `logs/trade_journal.csv`
- Deduplicates on `trade_id` (live journal wins on conflict)
- Writes merged result to `reports/trade_journal_recovered.csv`

---

## 5. Server Recovery (Cold Start)

Run this after a full server crash or fresh VM deployment:

```python
from monitoring.disaster_recovery import DisasterRecovery
dr = DisasterRecovery()
report = dr.server_recovery()   # add dry_run=True to preview
print(report)
```

**Steps executed automatically:**

| Step | Action |
|------|--------|
| 1 | Find latest backup in `backups/` |
| 2 | Verify all backup file integrity |
| 3 | Restore database |
| 4 | Restore ML models + registry |
| 5 | Restore trade journal |
| 6 | Reconstruct trade log from all backups |
| 7 | Re-initialise DB schema (idempotent) |

---

## 6. Emergency Procedures

### 6.1 Daily loss limit triggered

The `SafetySystem` halts trading automatically. To resume:

```python
# In Python
engine.reset_safety()
engine.start()

# Via dashboard: Autonomous Trading → Reset Safety & Resume
```

### 6.2 Model confidence collapse

Triggered when rolling AI confidence drops below 45% for 10 consecutive predictions.

**Recovery:**
1. Check `reports/autonomous/` logs for model accuracy
2. Retrain: `python main.py --train-models --data-file data/historical/RELIANCE_NS.csv`
3. Reset safety and restart engine

### 6.3 Broker connection failure

Paper broker never truly disconnects. For live broker adapters:
1. Check `ZERODHA_API_KEY`, `ZERODHA_ACCESS_TOKEN` env vars
2. Refresh access token (Zerodha tokens expire daily)
3. Restart engine

### 6.4 Database corruption

```bash
# Verify integrity
python -c "
import sqlite3
with sqlite3.connect('database/trading.db') as conn:
    result = conn.execute('PRAGMA integrity_check').fetchone()
    print(result)  # 'ok' = healthy
"

# If corrupted — restore from backup
python -c "
from monitoring.disaster_recovery import DisasterRecovery
dr = DisasterRecovery()
result = dr.restore_latest()
print(result.to_dict())
"
```

---

## 7. Backup Retention Policy

Backups accumulate in `backups/`. Clean up old sets manually:

```python
from monitoring.disaster_recovery import DisasterRecovery
import shutil, datetime

dr = DisasterRecovery()
all_backups = dr.list_backups()   # newest first
keep = 7  # keep last 7 backups

for manifest in all_backups[keep:]:
    backup_dir = dr.backup_root / manifest.backup_id
    shutil.rmtree(backup_dir)
    print(f"Removed: {manifest.backup_id}")
```

---

## 8. CSV Historical Data

### Exact file path

```
data/historical/<SYMBOL_WITH_DOTS_AS_UNDERSCORES>.csv
```

**Examples:**

| Symbol | CSV filename |
|--------|-------------|
| `RELIANCE.NS` | `data/historical/RELIANCE_NS.csv` |
| `TCS.NS` | `data/historical/TCS_NS.csv` |
| `HDFCBANK.NS` | `data/historical/HDFCBANK_NS.csv` |

This path is used by:
- Dashboard → Stock Analysis page (`load_historical_csv()`)
- `StockScanner._load_ohlcv()` in autonomous mode
- `BacktestEngine` and all strategy tests

### Download RELIANCE.NS data

```bash
# Using the built-in collector (recommended)
python main.py --collect-history --symbol RELIANCE.NS --years 2

# The file will be saved to:
#   data/historical/RELIANCE_NS.csv
```

The collector uses `yfinance` under the hood via `YahooFinanceProvider` and
saves the validated OHLCV CSV automatically at the correct path.

---

## 9. Verify Recovery Worked

```bash
python -m pytest tests/test_e2e_simulation.py -q
```

All 57 tests must pass. The final test (`TestValidationReport`) prints:

```
============================================================
  AI TRADING SYSTEM -- PRODUCTION VALIDATION REPORT
============================================================
  Score     : 100.0%
  Readiness : PRODUCTION READY
  [PASS] Stage 1: Data Validation                 100%
  ...
  [PASS] Stage 7: Disaster Recovery               100%
============================================================
```
