import argparse
import os

import psycopg2
from werkzeug.security import generate_password_hash


def load_dotenv(project_root: str) -> None:
    env_path = os.path.join(project_root, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def ensure_users_table(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            company TEXT NOT NULL,
            password_hash TEXT NOT NULL
        );
        """
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed default admin user.")
    parser.add_argument("--email", default="admin@growmax.com")
    parser.add_argument("--name", default="Aerix")
    parser.add_argument("--company", default="Aerixnova technologies")
    parser.add_argument("--password", default="Admin@123")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(project_root)

    db_url = args.db_url or os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/cctv_logs"
    )

    conn = psycopg2.connect(db_url)
    try:
        cur = conn.cursor()
        ensure_users_table(cur)

        cur.execute("SELECT 1 FROM users WHERE email = %s", (args.email,))
        if cur.fetchone():
            print(f"User already exists: {args.email}")
            conn.commit()
            return 0

        pw_hash = generate_password_hash(args.password)
        cur.execute(
            """
            INSERT INTO users (email, name, company, password_hash)
            VALUES (%s, %s, %s, %s)
            """,
            (args.email, args.name, args.company, pw_hash),
        )
        conn.commit()
        print(f"Seeded admin user: {args.email}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
