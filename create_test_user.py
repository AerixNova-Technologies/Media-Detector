import os
import werkzeug.security
import psycopg2
from dotenv import load_dotenv

load_dotenv()
url = os.environ.get("DATABASE_URL")
if url and "postgres" in url:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    conn = psycopg2.connect(url)
    cur  = conn.cursor()
    
    email = "admin@example.com"
    password = "admin"
    pw_hash = werkzeug.security.generate_password_hash(password)
    
    cur.execute("SELECT id FROM roles WHERE name = 'Administrator'")
    role_id = cur.fetchone()[0]
    
    cur.execute("""
        INSERT INTO users (email, name, company, password_hash, role_id) 
        VALUES (%s, %s, %s, %s, %s) 
        ON CONFLICT (email) DO UPDATE SET role_id = EXCLUDED.role_id, password_hash = EXCLUDED.password_hash
    """, (email, "Admin User", "Mission Control", pw_hash, role_id))
    
    conn.commit()
    print("Admin user created: email=admin@example.com, password=admin")
    conn.close()
