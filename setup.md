# Project Setup Guide (`setup.md`)

This guide helps a new user set up and run the project from scratch on Windows.

## 1) Prerequisites

- Python `3.10.x` (recommended: `3.10.11`)
- PostgreSQL `14+` (recommended: `16`)
- Git
- (Recommended) Visual Studio C++ Build Tools (some AI packages may need compilation support)

## 2) Clone the project

```powershell
git clone <your-repository-url>
cd media
```

## 3) Create and activate virtual environment

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 4) Install dependencies

```powershell
pip install -r requirements.txt
```

## 5) Configure PostgreSQL and environment variables

Create a PostgreSQL database (example name: `mediadetect`) and then create/update `.env` in project root:

```env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5432/mediadetect
PORT=5000
FLASK_SECRET_KEY=replace_with_a_long_random_secret
DETECTION_MODE=yolo_only
```

Optional variables:

```env
# Optional Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Optional Imou camera/cloud
IMOU_APP_ID=your_imou_app_id
IMOU_APP_SECRET=your_imou_app_secret
IMOU_CAM1_URL=rtsp://username:password@camera-ip:554/cam/realmonitor?channel=1&subtype=0
IMOU_CAM2_URL=rtsp://username:password@camera-ip:554/cam/realmonitor?channel=1&subtype=0
```

## 6) Initialize database tables

The app auto-initializes tables at startup, but you can run it explicitly:

```powershell
python -c "from app.db.session import init_db; init_db()"
```

## 7) (Optional) Seed an admin user

```powershell
python scripts\init_db.py --email admin@example.com --name Admin --company MyCompany --password ChangeMe123!
```

## 8) Run the application

Production-style (Waitress if installed):

```powershell
python run.py
```

Development mode (Flask debug/reload):

```powershell
python run.py --dev
```

Open: `http://localhost:5000`

## 9) One-command installer (alternative)

You can also run the interactive installer:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_windows.ps1
```

This script can:
- Ensure Python/PostgreSQL are available
- Create virtual environment and install dependencies
- Update `.env` with `DATABASE_URL`
- Create database if missing
- Initialize tables

## 10) Quick verification checklist

- `.env` exists and has correct `DATABASE_URL`
- PostgreSQL service is running
- `python run.py` starts without DB errors
- Dashboard loads at `http://localhost:5000`

## 11) Common issues

- **`DATABASE_URL not found`**: Add `DATABASE_URL=...` to `.env` in project root.
- **`psycopg2-binary is required`**: Re-run `pip install -r requirements.txt`.
- **Port already in use**: Change `PORT` in `.env` (for example, `PORT=5050`).
- **PowerShell script blocked**: Run with `-ExecutionPolicy Bypass` as shown above.
