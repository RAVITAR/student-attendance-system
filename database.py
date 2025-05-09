# database.py
import mysql.connector
from werkzeug.security import generate_password_hash
from config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB


# For NASA-level usage, these might come from environment variables
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "Abdulbasit12."
MYSQL_DB = "student_attendance"

def create_database():
    """
    Creates the MySQL database if it doesn't exist.
    In production, ensure you handle user privileges securely.
    """
    try:
        tmp_db = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD
        )
        tmp_cursor = tmp_db.cursor()
        tmp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB};")
        tmp_db.commit()
        tmp_db.close()
        print(f"✅ Database '{MYSQL_DB}' ready.")
    except mysql.connector.Error as err:
        print(f"❌ MySQL Error (Create DB): {err}")

def create_tables():
    """
    Creates or alters the necessary tables (students, attendance, staff).
    Ensures staff has a password_hash column and a default admin user.
    """
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        c = conn.cursor()

        # Students
        c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id VARCHAR(50) UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            rfid_card VARCHAR(50) UNIQUE,
            face_data VARCHAR(100),
            qr_code VARCHAR(255)
        );
        """)

        # Attendance
        c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id VARCHAR(50) NOT NULL,
            method VARCHAR(20),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Staff
        c.execute("""
        CREATE TABLE IF NOT EXISTS staff (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(100) UNIQUE NOT NULL
            -- We'll add password_hash if missing
        );
        """)

        # Check if 'password_hash' column exists
        c.execute("SHOW COLUMNS FROM staff LIKE 'password_hash'")
        column_result = c.fetchone()
        if not column_result:
            try:
                c.execute("ALTER TABLE staff ADD password_hash VARCHAR(200) NOT NULL")
                print("✅ Added 'password_hash' column to 'staff' table.")
            except mysql.connector.Error as alt_err:
                print(f"❌ Could not add 'password_hash' column: {alt_err}")

        # Insert default admin user if missing
        c.execute("SELECT * FROM staff WHERE email=%s", ("admin@example.com",))
        existing_admin = c.fetchone()
        if not existing_admin:
            admin_hash = generate_password_hash("admin123")
            c.execute("""
                INSERT INTO staff (email, password_hash)
                VALUES ('admin@example.com', %s)
            """, (admin_hash,))
            print("✅ Inserted default admin with hashed password.")

        conn.commit()
        conn.close()
        print("✅ Tables ready.")
    except mysql.connector.Error as err:
        print(f"❌ MySQL Error (Create Tables): {err}")

def get_db_connection():
    """
    Returns a new MySQL connection or None if error.
    For NASA-level usage, consider connection pooling.
    """
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        return conn
    except mysql.connector.Error as err:
        print(f"❌ DB Connection Error: {err}")
        return None
