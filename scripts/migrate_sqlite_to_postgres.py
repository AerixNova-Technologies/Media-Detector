import argparse
import os
import sqlite3

import psycopg2


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
    parser = argparse.ArgumentParser(description="Migrate users.db to PostgreSQL.")
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(project_root)

    sqlite_path = args.sqlite_path or os.path.join(project_root, "users.db")
    if not os.path.exists(sqlite_path):
        print(f"SQLite file not found: {sqlite_path}")
        return 1

    db_url = args.db_url or os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/cctv_logs"
    )

    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_cur = sqlite_conn.cursor()

    sqlite_cur.execute(
        "SELECT email, name, company, password_hash FROM users"
    )
    rows = sqlite_cur.fetchall()
    sqlite_conn.close()

    if not rows:
        print("No users found in SQLite.")
        return 0

    pg_conn = psycopg2.connect(db_url)
    try:
        pg_cur = pg_conn.cursor()
        ensure_users_table(pg_cur)

        inserted = 0
        for email, name, company, password_hash in rows:
            pg_cur.execute(
                """
                INSERT INTO users (email, name, company, password_hash)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (email) DO NOTHING
                """,
                (email, name, company, password_hash),
            )
            if pg_cur.rowcount == 1:
                inserted += 1

        pg_conn.commit()
        print(f"Migrated {inserted} user(s) from SQLite to PostgreSQL.")
        return 0
    finally:
        pg_conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
