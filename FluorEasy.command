#!/bin/zsh
# FluorEasy — One-Click macOS Launcher for Batch Microscope Image Processing
APP_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$APP_DIR" || {
  echo "Error: Could not find FluorEasy directory at $APP_DIR"
  read -k 1 "?Press any key to exit..."
  exit 1
}

if [ ! -d "$APP_DIR/.venv" ]; then
  echo "Creating Python virtual environment and installing dependencies..."
  python3 -m venv "$APP_DIR/.venv"
  "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"
fi

source "$APP_DIR/.venv/bin/activate"

clear
echo "================================================================"
echo "  🔬 FluorEasy — Batch Microscope Image Studio"
echo "================================================================"
echo "  Starting FluorEasy web server..."
echo "  Your browser will open automatically at:"
echo "  http://localhost:8501"
echo ""
echo "  Keep this window open while using FluorEasy."
echo "  Press Control + C here when you are ready to quit."
echo "================================================================"
echo ""

exec "$APP_DIR/.venv/bin/streamlit" run "$APP_DIR/app.py"
