import psycopg2
import os

from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.environ.get("DATABASE_URL"))
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS local_cameras (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    brand VARCHAR(100) NOT NULL,
    ip_address VARCHAR(100) NOT NULL,
    port INTEGER DEFAULT 554,
    username VARCHAR(255),
    password VARCHAR(255),
    stream_path TEXT,
    owner_email VARCHAR(255) REFERENCES users(email) ON DELETE CASCADE
);
""")

conn.commit()
cur.close()
conn.close()

print("Local camera table created successfully!")
