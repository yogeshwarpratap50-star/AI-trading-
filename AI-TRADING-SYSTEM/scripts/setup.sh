#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${1:-.venv}"
python3.12 -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/pip" install -r requirements.txt
cp -n .env.example .env || true
"${VENV_PATH}/bin/python" main.py --init-db
