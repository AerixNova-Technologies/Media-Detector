#!/bin/bash

# AI CCTV Surveillance System - Install Script (Unix)
# Sets up virtual environment, installs dependencies, and seeds PostgreSQL database.

echo "--- Installing AI CCTV System ---"

# 1. Create venv if not existing
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtualenv already exists. Skipping."
fi

# 2. Activate venv
source venv/bin/activate

# 3. Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure DATABASE_URL if missing
if [ -z "$DATABASE_URL" ]; then
    echo "PostgreSQL connection setup:"
    read -r -p "DB name (default: cctv_logs): " DB_NAME
    read -r -p "DB user (default: postgres): " DB_USER
    read -r -p "DB host (default: localhost): " DB_HOST
    read -r -p "DB port (default: 5432): " DB_PORT
    read -r -s -p "DB password: " DB_PASS
    echo ""

    DB_NAME=${DB_NAME:-cctv_logs}
    DB_USER=${DB_USER:-postgres}
    DB_HOST=${DB_HOST:-localhost}
    DB_PORT=${DB_PORT:-5432}

    DATABASE_URL="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
    export DATABASE_URL

    echo "DATABASE_URL=${DATABASE_URL}" > .env
    echo "Saved DATABASE_URL to .env"
fi

# 5. Seed database
if [ -f "scripts/seed_admin.py" ]; then
    echo "Seeding default admin user..."
    python3 scripts/seed_admin.py
fi

echo "--- Installation Complete! ---"
echo "To run the system:"
echo "1. source venv/bin/activate"
echo "2. python3 app.py"
