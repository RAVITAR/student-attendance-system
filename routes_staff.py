# routes_staff.py
import os
import shutil
import time
import uuid
import threading
import qrcode
from flask import (
    Blueprint, render_template, request, redirect, url_for,
    session, flash, send_from_directory, jsonify
)
from werkzeug.security import check_password_hash
from database import get_db_connection
from model_adapter import train_model, capture_face_images

staff_bp = Blueprint('staff_bp', __name__)

# Global training status tracker
training_status = {
    "phase": 0,
    "percent": 0,
    "done": False,
    "message": "Not started",
    "accuracy": None
}

@staff_bp.route('/train_model_status')
def get_training_status():
    return jsonify(training_status)

def run_training():
    try:
        training_status.update({"phase": 1, "percent": 10, "message": "Loading Dataset", "done": False})
        time.sleep(1)

        training_status.update({"phase": 2, "percent": 30, "message": "Extracting Features"})
        time.sleep(2)

        training_status.update({"phase": 3, "percent": 60, "message": "Training Classifier"})
        time.sleep(3)

        training_status.update({"phase": 4, "percent": 90, "message": "Evaluating Model"})
        acc = train_model(dataset_dir="student_dataset")

        training_status.update({
            "phase": 4,
            "percent": 100,
            "message": f"Done. Accuracy: {acc * 100:.2f}%" if acc else "Training completed.",
            "done": True,
            "accuracy": acc * 100 if acc else None
        })
    except Exception as e:
        training_status.update({
            "phase": -1,
            "percent": 100,
            "message": f"Error: {str(e)}",
            "done": True
        })

@staff_bp.route('/start_training_async')
def start_training_async():
    if not session.get('staff_logged_in'):
        flash("Login required", "danger")
        return redirect(url_for('staff_bp.staff_login'))

    training_status.update({"phase": 0, "percent": 0, "message": "Starting...", "done": False})
    thread = threading.Thread(target=run_training)
    thread.start()

    return render_template("train_model_status.html")

@staff_bp.route('/staff_login')
def staff_login():
    return render_template("staff_login.html")

@staff_bp.route('/do_staff_login', methods=['POST'])
def do_staff_login():
    email = request.form.get("email")
    password = request.form.get("password")

    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT id, password_hash FROM staff WHERE email=%s", (email,))
        result = c.fetchone()
    except Exception as e:
        flash(f"DB Error: {e}", "danger")
        conn.close()
        return redirect(url_for('staff_bp.staff_login'))
    conn.close()

    if result:
        staff_id, password_hash = result
        if check_password_hash(password_hash, password):
            session['staff_logged_in'] = True
            session['staff_id'] = staff_id
            flash("Logged in successfully!", "success")
            return redirect(url_for('staff_bp.admin_panel'))
        else:
            flash("Wrong password!", "danger")
    else:
        flash("Staff not found!", "danger")
    return redirect(url_for('staff_bp.staff_login'))

@staff_bp.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('home'))

@staff_bp.route('/student_images/<student_id>/<filename>')
def student_images(student_id, filename):
    base_path = os.path.join("student_dataset", student_id)
    if not os.path.exists(base_path):
        flash("No images found for this student.", "warning")
        return redirect(url_for('staff_bp.admin_panel'))
    return send_from_directory(base_path, filename)

@staff_bp.route('/admin_panel')
def admin_panel():
    if not session.get('staff_logged_in'):
        flash("Please log in as staff first.", "warning")
        return redirect(url_for('staff_bp.staff_login'))

    conn = get_db_connection()
    c = conn.cursor()

    search_students = request.args.get('search_students', '').strip()
    search_logs = request.args.get('search_logs', '').strip()

    try:
        if search_students:
            like_q = f"%{search_students}%"
            c.execute("""
                SELECT student_id, name, rfid_card, qr_code 
                FROM students
                WHERE student_id LIKE %s OR name LIKE %s OR rfid_card LIKE %s
                ORDER BY student_id
            """, (like_q, like_q, like_q))
        else:
            c.execute("SELECT student_id, name, rfid_card, qr_code FROM students ORDER BY student_id")
        students = c.fetchall()
    except Exception as e:
        flash(f"DB Error fetching students: {e}", "danger")
        conn.close()
        return redirect(url_for('staff_bp.staff_login'))

    try:
        if search_logs:
            like_q = f"%{search_logs}%"
            c.execute("""
                SELECT a.id, a.student_id, s.name, s.rfid_card, a.method, a.timestamp
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                WHERE a.student_id LIKE %s OR s.name LIKE %s OR s.rfid_card LIKE %s OR a.method LIKE %s
                ORDER BY a.timestamp DESC
                LIMIT 50
            """, (like_q, like_q, like_q, like_q))
        else:
            c.execute("""
                SELECT a.id, a.student_id, s.name, s.rfid_card, a.method, a.timestamp
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                ORDER BY a.timestamp DESC
                LIMIT 50
            """)
        logs = c.fetchall()
    except Exception as e:
        flash(f"DB Error fetching attendance logs: {e}", "danger")
        logs = []
    conn.close()

    accuracy_info = training_status.get("accuracy")
    images_dict = {}
    base_dataset = "student_dataset"
    for s in students:
        sid = s[0]
        folder_path = os.path.join(base_dataset, sid)
        first_image = next((f for f in sorted(os.listdir(folder_path)) if os.path.isfile(os.path.join(folder_path, f))), None) if os.path.exists(folder_path) else None
        images_dict[sid] = first_image

    return render_template(
        "admin.html",
        students=students,
        accuracy_info=accuracy_info,
        images_dict=images_dict,
        logs=logs,
        search_students=search_students,
        search_logs=search_logs
    )

@staff_bp.route('/add_student', methods=['POST'])
def add_student():
    if not session.get('staff_logged_in'):
        flash("Please log in as staff first.", "warning")
        return redirect(url_for('staff_bp.staff_login'))

    student_id = request.form.get("student_id")
    name = request.form.get("name")
    rfid_card = request.form.get("rfid_card") or None

    try:
        unique_uuid = str(uuid.uuid4())
        qr_data = f"{student_id}|{unique_uuid}|{int(time.time())}"
        qr_img = qrcode.make(qr_data)
        qr_dir = os.path.join("static", "qr_codes")
        os.makedirs(qr_dir, exist_ok=True)
        qr_filename = f"{student_id}.png"
        qr_img.save(os.path.join(qr_dir, qr_filename))
    except Exception as e:
        flash(f"Error generating QR code: {e}", "danger")
        return redirect(url_for('staff_bp.admin_panel'))

    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO students (student_id, name, rfid_card, qr_code)
            VALUES (%s, %s, %s, %s)
        """, (student_id, name, rfid_card, qr_filename))
        conn.commit()
        flash("Student added successfully! Unique QR code generated.", "success")
    except Exception as e:
        flash(f"Error adding student: {e}", "danger")
    conn.close()

    return redirect(url_for('staff_bp.admin_panel'))

@staff_bp.route('/enroll_face/<student_id>')
def enroll_face(student_id):
    if not session.get('staff_logged_in'):
        flash("Please log in as staff first.", "warning")
        return redirect(url_for('staff_bp.staff_login'))

    success = capture_face_images(student_id, dataset_dir="student_dataset", num_images=5)
    if success:
        acc = train_model(dataset_dir="student_dataset")
        if acc is not None:
            training_status['accuracy'] = acc * 100
        flash(f"✅ Face enrollment completed for {student_id}. Model retrained.", "success")
    else:
        flash("❌ Face enrollment failed (no images captured or camera issue).", "danger")

    return redirect(url_for('staff_bp.admin_panel'))

@staff_bp.route('/edit_student/<student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    if not session.get('staff_logged_in'):
        flash("Please log in first.", "warning")
        return redirect(url_for('staff_bp.staff_login'))

    conn = get_db_connection()
    c = conn.cursor()
    try:
        if request.method == 'POST':
            new_name = request.form.get('name')
            new_rfid = request.form.get('rfid_card') or None
            c.execute("""
                UPDATE students 
                SET name=%s, rfid_card=%s 
                WHERE student_id=%s
            """, (new_name, new_rfid, student_id))
            conn.commit()
            flash(f"Student '{student_id}' updated successfully!", "success")
            conn.close()
            return redirect(url_for('staff_bp.admin_panel'))
        else:
            c.execute("SELECT student_id, name, rfid_card FROM students WHERE student_id=%s", (student_id,))
            student = c.fetchone()
            conn.close()
            if not student:
                flash("Student not found!", "danger")
                return redirect(url_for('staff_bp.admin_panel'))
            return render_template("edit_student.html", student=student)
    except Exception as e:
        flash(f"Error editing student: {e}", "danger")
        conn.close()
        return redirect(url_for('staff_bp.admin_panel'))

@staff_bp.route('/delete_student/<student_id>')
def delete_student(student_id):
    if not session.get('staff_logged_in'):
        flash("Please log in first.", "warning")
        return redirect(url_for('staff_bp.staff_login'))

    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("DELETE FROM students WHERE student_id=%s", (student_id,))
        conn.commit()
        flash(f"Student '{student_id}' deleted successfully!", "success")
    except Exception as e:
        flash(f"Error deleting student: {e}", "danger")
    conn.close()

    folder_path = os.path.join("student_dataset", student_id)
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            flash(f"Error removing images: {e}", "warning")

    return redirect(url_for('staff_bp.admin_panel'))

@staff_bp.route('/delete_attendance/<int:log_id>')
def delete_attendance(log_id):
    if not session.get('staff_logged_in'):
        flash("Please log in first.", "warning")
        return redirect(url_for('staff_bp.staff_login'))

    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("DELETE FROM attendance WHERE id=%s", (log_id,))
        conn.commit()
        flash(f"Attendance log #{log_id} deleted successfully!", "success")
    except Exception as e:
        flash(f"Error deleting attendance log: {e}", "danger")
    conn.close()
    return redirect(url_for('staff_bp.admin_panel'))
