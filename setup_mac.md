# Project Setup Guide (`setup_mac.md`)

This guide helps a new user set up and run the project from scratch on macOS.

## 1) Prerequisites

- **Homebrew**: The standard package manager for macOS. You can install it from [brew.sh](https://brew.sh/).
- **Python 3.10.x**: Required for compatibility with AI packages like PyTorch (`brew install python@3.10`).
- **PostgreSQL 14+**: The primary database (`brew install postgresql@16`).

## 2) Clone the project

```bash
git clone <your-repository-url>
cd media
```

## 3) Create and activate virtual environment

Make sure you use Python 3.10.

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

## 4) Install dependencies

```bash
pip install -r requirements.txt
pip install psycopg2-binary
```

## 5) Configure PostgreSQL and environment variables

Make sure PostgreSQL is running (`brew services start postgresql@16`).
Create a new PostgreSQL database. 
*Note: Homebrew PostgreSQL uses your macOS username as the default root user without a password for local connections.*

Create/update `.env` in the project root:

```env
# Replace your_mac_username with your actual macOS username. Password can be left empty if not configured.
DATABASE_URL=postgresql://your_mac_username:@127.0.0.1:5432/mediadetect
PORT=5000
FLASK_SECRET_KEY=replace_with_a_long_random_secret
DETECTION_MODE=yolo_only
```

*Optional variables (Telegram, Imou Camera, etc.) can also be added here.*

## 6) Initialize database tables

The app auto-initializes tables at startup, but you can run it explicitly:

```bash
python -c "from app.db.session import init_db; init_db()"
```

## 7) (Optional) Seed an admin user

If the script exists:
```bash
python scripts/seed_admin.py
```

## 8) Run the application

Production-style (Waitress if installed):

```bash
python run.py
```

Development mode (Flask debug/reload):

```bash
python run.py --dev
```

Open: `http://localhost:5000`

## 9) One-command installer (alternative)

You can also run the interactive installer for macOS:

```bash
chmod +x install_mac.sh
./install_mac.sh
```

This script will automatically:
- Check for Homebrew
- Install Python 3.10 and PostgreSQL 16 via Homebrew if missing
- Create the virtual environment and install dependencies
- Update `.env` with `DATABASE_URL`
- Create the database and initialize tables
