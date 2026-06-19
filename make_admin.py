import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'data', 'database', 'nexus.db')
conn = sqlite3.connect(db_path)

conn.execute("UPDATE users SET is_admin = 1 WHERE username = 'syedanas'")
conn.commit()

row = conn.execute("SELECT id, username, is_admin FROM users WHERE username = 'syedanas'").fetchone()
if row:
    print(f"SUCCESS! User '{row[1]}' (id={row[0]}) => is_admin = {row[2]}")
else:
    print("ERROR: User 'syedanas' not found in database!")

all_users = conn.execute("SELECT id, username, is_admin FROM users").fetchall()
print("\nAll users in DB:")
for u in all_users:
    print(f"  id={u[0]}  username={u[1]}  is_admin={u[2]}")

conn.close()
