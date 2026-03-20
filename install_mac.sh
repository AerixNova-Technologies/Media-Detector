#!/usr/bin/env bash
# macOS Installation Script for Media Detector

set -e

PROJECT_ROOT=$(pwd)
REQUIRE_PY="3.10"

echo -e "\033[1;36m--- Media Detector macOS Installation ---\033[0m"

# 1. Ensure Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo -e "\033[1;31mHomebrew is not installed. Please install it first from https://brew.sh/\033[0m"
    echo "Command: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# 2. Ensure Python 3.10 is installed
if ! command -v python3.10 &> /dev/null; then
    echo "Python 3.10 not found. Installing via Homebrew..."
    brew install python@3.10
fi

# Verify Python version
PY_VER=$(python3.10 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$PY_VER" != "3.10" ]; then
    echo -e "\033[1;31mFailed to install or locate Python 3.10. Found: $PY_VER\033[0m"
    exit 1
fi

# 3. Ensure PostgreSQL is installed and running
if ! command -v psql &> /dev/null; then
    export PATH="/opt/homebrew/opt/postgresql@16/bin:/usr/local/opt/postgresql@16/bin:$PATH"
fi

if ! command -v psql &> /dev/null; then
    echo "psql not found. Installing PostgreSQL 16 via Homebrew..."
    brew install postgresql@16
    export PATH="/opt/homebrew/opt/postgresql@16/bin:/usr/local/opt/postgresql@16/bin:$PATH"
fi

# Start the postgresql service just in case
echo "Ensuring PostgreSQL is running..."
brew services start postgresql@16 || true
sleep 2 # Give it a moment to start up

# 4. Create internal virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Virtual Environment (venv) with Python 3.10..."
    python3.10 -m venv venv
fi

# Activate venv
source "$PROJECT_ROOT/venv/bin/activate"

echo -e "\n\033[1;36mInstalling Dependencies (this may take a while)...\033[0m"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install psycopg2-binary

# 5. Database Configuration
echo -e "\n\033[1;33m--- Database Configuration ---\033[0m"

read -p "Enter Database Name (required) [default: mediadetect]: " DB_NAME
DB_NAME=${DB_NAME:-mediadetect}

MAC_USER=$(whoami)
read -p "Enter PostgreSQL Username [default: $MAC_USER]: " DB_USER
DB_USER=${DB_USER:-$MAC_USER}

read -p "Enter PostgreSQL Host [default: 127.0.0.1]: " DB_HOST
DB_HOST=${DB_HOST:-127.0.0.1}

read -p "Enter PostgreSQL Port [default: 5432]: " DB_PORT
DB_PORT=${DB_PORT:-5432}

# On Mac with Homebrew, the default superuser is often the mac username, without a password for local connections.
read -s -p "Enter Password for $DB_USER (leave blank if none): " DB_PASS
echo

if [ -z "$DB_PASS" ]; then
    DB_URL="postgresql://$DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"
else
    DB_URL="postgresql://$DB_USER:$DB_PASS@$DB_HOST:$DB_PORT/$DB_NAME"
fi

# Update .env
if [ ! -f .env ]; then
    touch .env
fi

if grep -q "^DATABASE_URL=" .env; then
    sed -i.bak "s|^DATABASE_URL=.*|DATABASE_URL=$DB_URL|" .env
    rm -f .env.bak
else
    echo "DATABASE_URL=$DB_URL" >> .env
fi
echo -e "\033[1;32mConnection string saved to .env\033[0m"

# Create Database if it doesn't exist
if [ -n "$DB_PASS" ]; then
    export PGPASSWORD="$DB_PASS"
fi

echo "Checking if database '$DB_NAME' exists..."
DB_EXISTS=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" 2>/dev/null || echo "0")

if [ "$DB_EXISTS" != "1" ]; then
    echo -e "\033[1;36mCreating database '$DB_NAME'...\033[0m"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE \"$DB_NAME\";" || {
        echo -e "\033[1;31mFailed to create database. Does the user '$DB_USER' exist with db creation permissions?\033[0m"
    }
fi

# Initialize Schema
echo -e "\033[1;36mInitializing database tables...\033[0m"
python -c "from app.db.session import init_db; init_db()" || true

# Migration support
if [ -f "migrate_to_pg.py" ]; then
    read -p "Do you want to migrate existing data from SQLite? (y/n) [default: n]: " MOVE
    if [[ "$MOVE" == "y" || "$MOVE" == "Y" ]]; then
        python migrate_to_pg.py
    fi
fi

unset PGPASSWORD

# Seed Admin
if [ -f "scripts/seed_admin.py" ]; then
    echo "Seeding default admin user..."
    python scripts/seed_admin.py || true
fi

echo -e "\n\033[1;32m--------------------------------------------------------\033[0m"
echo -e "\033[1;32mINSTALLATION SUCCESSFUL!\033[0m"
echo "1. Activate: source venv/bin/activate"
echo "2. Run: python run.py"
echo -e "\033[1;32m--------------------------------------------------------\033[0m"
