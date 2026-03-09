import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))
from db_manager import create_user, verify_user
import sqlite3

user = create_user("witness", "witness@nexus.ai", "admin123")
if user:
    print(f"Created witness user: {user}")
    conn = sqlite3.connect("data/database/nexus.db")
    conn.execute('UPDATE users SET is_admin=1 WHERE username="witness"')
    conn.commit()
    conn.close()
    print("Made witness an admin")
else:
    print("Witness user already exists")

user = verify_user("witness", "admin123")
if user:
    print("Verification OK:", user)
