import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import json
import os

# Connect to the default 'postgres' database to create a new one
conn = psycopg2.connect("dbname=postgres user=postgres password=postgres host=localhost port=5432")
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()

db_name = "cctv_logs"

cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'cctv_logs'")
exists = cur.fetchone()

if not exists:
    print(f"Creating database {db_name}...")
    cur.execute(f"CREATE DATABASE {db_name};")
else:
    print(f"Database {db_name} already exists.")

cur.close()
conn.close()

# Connect to the new database
conn = psycopg2.connect(f"dbname={db_name} user=postgres password=postgres host=localhost port=5432")
cur = conn.cursor()

# Create table for Users
print("Creating users table...")
cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        name VARCHAR(255) NOT NULL,
        company VARCHAR(255) NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")
conn.commit()

# Push JSON data to the new database
json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
if os.path.exists(json_file):
    print("Migrating users.json to Postgres...")
    with open(json_file, 'r') as f:
        users = json.load(f)
        for email, data in users.items():
            try:
                cur.execute("""
                    INSERT INTO users (email, name, company, password_hash)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (email) DO NOTHING;
                """, (email, data.get("name"), data.get("company"), data.get("password")))
            except Exception as e:
                print(f"Error inserting user {email}: {e}")
                conn.rollback()
    
    conn.commit()
    print("Migration successful.")
else:
    print("No users.json found to migrate.")

cur.close()
conn.close()
print("Database setup complete.")
