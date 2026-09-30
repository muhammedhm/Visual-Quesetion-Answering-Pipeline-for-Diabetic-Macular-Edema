import os
import sqlite3
import hashlib
import secrets

DB_PATH = "deepeye.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER NOT NULL DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    cols = {row[1] for row in c.execute("PRAGMA table_info(users)").fetchall()}
    if "is_admin" not in cols:
        c.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
    c.execute("""CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS rag_searches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        query_text TEXT,
        results_json TEXT NOT NULL,
        structured_json TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        admin_username = os.getenv("DEEPEYE_ADMIN_USERNAME", "admin")
        admin_password = os.getenv("DEEPEYE_ADMIN_PASSWORD", "admin123")
        c.execute(
            "INSERT INTO users (username,password,is_admin) VALUES (?,?,1)",
            (admin_username.strip(), _hash(admin_password)),
        )
    conn.commit()
    conn.close()

def _hash(p): return hashlib.sha256(p.encode()).hexdigest()

def register_user(username, password, is_admin=False):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT INTO users (username,password,is_admin) VALUES (?,?,?)",
                     (username.strip(), _hash(password), 1 if is_admin else 0))
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?",
              (username.strip(), _hash(password)))
    row = c.fetchone()
    if not row:
        conn.close()
        return False, "Invalid username or password.", -1
    token = secrets.token_hex(32)
    conn.execute("INSERT INTO sessions (token,user_id) VALUES (?,?)", (token, row[0]))
    conn.commit()
    conn.close()
    return True, token, row[0]

def validate_token(token):
    if not token: return False, -1, ""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT s.user_id,u.username FROM sessions s JOIN users u ON s.user_id=u.id WHERE s.token=?", (token,))
    row = c.fetchone()
    conn.close()
    return (True, row[0], row[1]) if row else (False, -1, "")

def validate_token_details(token):
    if not token: return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT s.user_id,u.username,u.is_admin FROM sessions s "
        "JOIN users u ON s.user_id=u.id WHERE s.token=?",
        (token,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row["user_id"], "username": row["username"], "is_admin": bool(row["is_admin"])}

def list_users():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, username, is_admin, created_at FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [
        {"id": r["id"], "username": r["username"], "is_admin": bool(r["is_admin"]), "created_at": r["created_at"]}
        for r in rows
    ]

def delete_user_by_username(username, requester_id=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    row = c.execute("SELECT id, username FROM users WHERE username=?", (username.strip(),)).fetchone()
    if not row:
        conn.close()
        return False, "User not found."
    if requester_id is not None and row["id"] == requester_id:
        conn.close()
        return False, "Admins cannot delete their own active account."
    c.execute("DELETE FROM sessions WHERE user_id=?", (row["id"],))
    c.execute("DELETE FROM queries WHERE user_id=?", (row["id"],))
    c.execute("DELETE FROM rag_searches WHERE user_id=?", (row["id"],))
    c.execute("DELETE FROM users WHERE id=?", (row["id"],))
    conn.commit()
    conn.close()
    return True, "User deleted successfully."

def logout_user(token):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()
    conn.close()

def save_query(user_id, image_path, question, answer):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO queries (user_id,image_path,question,answer) VALUES (?,?,?,?)",
              (user_id, image_path, question, answer))
    qid = c.lastrowid
    conn.commit(); conn.close()
    return qid

def get_user_queries(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM queries WHERE user_id=? ORDER BY created_at DESC", (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_query(qid, user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM queries WHERE id=? AND user_id=?", (qid, user_id))
    ok = c.rowcount > 0
    conn.commit(); conn.close()
    return ok

def save_rag_search(user_id, image_path, query_text, results_json, structured_json=""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO rag_searches (user_id,image_path,query_text,results_json,structured_json) VALUES (?,?,?,?,?)",
                 (user_id, image_path, query_text, results_json, structured_json))
    conn.commit(); conn.close()

def get_user_rag_searches(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM rag_searches WHERE user_id=? ORDER BY created_at DESC LIMIT 30", (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_rag_search(sid, user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM rag_searches WHERE id=? AND user_id=?", (sid, user_id))
    ok = c.rowcount > 0
    conn.commit(); conn.close()
    return ok
