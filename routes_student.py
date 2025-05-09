import time
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify, current_app
from database import get_db_connection
from model_adapter import aggregator_recognize_face, capture_face_images
from extensions import csrf
from datetime import datetime

student_bp = Blueprint('student_bp', __name__)

# ---------------------------------------------------------------------------
# Route: /student
# Renders the student login page (choose QR or RFID).
# ---------------------------------------------------------------------------
@student_bp.route('/student')
def student():
    return render_template("student_login.html")

# ---------------------------------------------------------------------------
# Route: /student_auth
# Handles student login submission.
# For QR, it expects a string in the format: student_id|unique_uuid|timestamp.
# For RFID, it now looks up the student record and stores the student ID.
# ---------------------------------------------------------------------------
@student_bp.route('/student_auth', methods=['POST'])
def student_auth():
    method = request.form.get("method")
    session['method'] = method

    if method == "QR":
        qr_data = request.form.get("qr_data")
        if not qr_data:
            flash("No QR data provided!", "danger")
            return redirect(url_for('student_bp.student'))
        try:
            parts = qr_data.split("|")
            if len(parts) != 3:
                flash("Invalid QR data format!", "danger")
                return redirect(url_for('student_bp.student'))
            sid, unique_uuid, ts = parts
            session['qr_student_id'] = sid
        except Exception:
            flash("Invalid QR data format!", "danger")
            return redirect(url_for('student_bp.student'))
        return redirect(url_for('student_bp.face_2fa'))

    elif method == "RFID":
        card_id = request.form.get("rfid_input")
        if card_id:
            # Look up the student id from the database using the RFID card value.
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT student_id FROM students WHERE rfid_card=%s", (card_id,))
            row = c.fetchone()
            conn.close()
            if row:
                session['rfid_id'] = card_id
                session['rfid_student_id'] = row[0]
                return redirect(url_for('student_bp.face_2fa'))
            else:
                flash("RFID not recognized in our system!", "danger")
                return redirect(url_for('student_bp.student'))
        else:
            flash("No RFID data scanned!", "danger")
            return redirect(url_for('student_bp.student'))
    else:
        flash("Invalid method!", "danger")
        return redirect(url_for('student_bp.student'))

# ---------------------------------------------------------------------------
# Route: /face_2fa
# Renders the face 2FA page (the page’s JS will call the JSON endpoint).
# ---------------------------------------------------------------------------
@student_bp.route('/face_2fa')
def face_2fa():
    return render_template("face_2fa.html")

# ---------------------------------------------------------------------------
# Route: /do_face_2fa_json (AJAX Endpoint)
# CSRF-exempt so that AJAX POST requests are not blocked.
# This endpoint runs the face recognition aggregator and logs attendance.
# ---------------------------------------------------------------------------
# Modified parts of student_bp.py to fix CSRF issues and redirect to login_success
@csrf.exempt
@student_bp.route('/do_face_2fa_json', methods=['POST'])
def do_face_2fa_json():
    # Comprehensive logging for debugging
    current_app.logger.info(f"Session data: {dict(session)}")

    # Validate session data before proceeding
    method = session.get('method')
    if not method:
        current_app.logger.error("No authentication method in session")
        return jsonify({
            "status": "error", 
            "message": "Authentication method not found. Please restart the login process."
        }), 400

    # Robust handling of aggregator result
    try:
        result = aggregator_recognize_face(
            max_duration=20,
            conf_min=0.05,
            aggregator_size=15,
            aggregator_sum_threshold=2.0
        )
    except Exception as e:
        current_app.logger.error(f"Face recognition error: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Face recognition system error: {str(e)}"
        }), 500

    # Validate result structure
    if not isinstance(result, dict):
        current_app.logger.error(f"Invalid result structure: {result}")
        return jsonify({
            "status": "error", 
            "message": "Unexpected response from face recognition system"
        }), 400

    # Check status and label
    if result.get("status") != "ok":
        current_app.logger.warning(f"Face recognition failed: {result.get('reason', 'Unknown reason')}")
        return jsonify({
            "status": "error", 
            "message": result.get("reason", "Face recognition failed")
        }), 400

    recognized_id = result.get("label")
    if not recognized_id:
        current_app.logger.warning("No label recognized in face recognition result")
        return jsonify({
            "status": "error", 
            "message": "Unable to recognize face"
        }), 400

    # More robust session and database checks
    conn = get_db_connection()
    try:
        c = conn.cursor()

        # QR Method Verification
        if method == "QR":
            qr_sid = session.get("qr_student_id")
            if not qr_sid:
                current_app.logger.error("No QR student ID in session")
                return jsonify({
                    "status": "error", 
                    "message": "QR authentication data missing"
                }), 400

            if recognized_id != qr_sid:
                current_app.logger.warning(f"Face mismatch: {recognized_id} != {qr_sid}")
                return jsonify({
                    "status": "error", 
                    "message": "Face does not match QR code"
                }), 400

            c.execute("INSERT INTO attendance (student_id, method) VALUES (%s, %s)", (recognized_id, "QR+Face"))

        # RFID Method Verification
        elif method == "RFID":
            rfid_student_id = session.get("rfid_student_id")
            if not rfid_student_id:
                # Attempt fallback lookup
                rfid_id = session.get("rfid_id")
                if not rfid_id:
                    current_app.logger.error("No RFID data in session")
                    return jsonify({
                        "status": "error", 
                        "message": "RFID authentication data missing"
                    }), 400

                c.execute("SELECT student_id FROM students WHERE rfid_card=%s", (rfid_id,))
                row = c.fetchone()
                if not row:
                    current_app.logger.error(f"No student found for RFID: {rfid_id}")
                    return jsonify({
                        "status": "error", 
                        "message": "RFID not recognized"
                    }), 400
                rfid_student_id = row[0]

            if rfid_student_id != recognized_id:
                current_app.logger.warning(f"Face mismatch: {recognized_id} != {rfid_student_id}")
                return jsonify({
                    "status": "error", 
                    "message": "Face does not match RFID"
                }), 400

            c.execute("INSERT INTO attendance (student_id, method) VALUES (%s, %s)", (recognized_id, "RFID+Face"))

        else:
            current_app.logger.error(f"Invalid method in session: {method}")
            return jsonify({
                "status": "error", 
                "message": "Invalid authentication method"
            }), 400

        conn.commit()
        session["final_student_id"] = recognized_id

        return jsonify({
            "status": "ok", 
            "redirect": url_for('student_bp.login_success')
        })

    except Exception as e:
        current_app.logger.error(f"Database error during 2FA: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": "Internal authentication error"
        }), 500
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Route: /do_face_2fa (Blocking Version)
# A classic blocking approach that uses flash messages.
# ---------------------------------------------------------------------------
@student_bp.route('/do_face_2fa', methods=['POST'])
def do_face_2fa():
    result = aggregator_recognize_face(
        max_duration=20,
        conf_min=0.05,
        aggregator_size=15,
        aggregator_sum_threshold=2.0
    )
    if result.get("status") != "ok":
        flash(f"Face not recognized or camera issue! Reason: {result.get('reason','unknown')}", "danger")
        return redirect(url_for('student_bp.face_2fa'))
    
    recognized_id = result.get("label")
    if not recognized_id:
        flash("No recognized label. Possibly aggregator error.", "danger")
        return redirect(url_for('student_bp.face_2fa'))
    
    conn = get_db_connection()
    c = conn.cursor()
    method = session.get("method")
    
    if method == "QR":
        qr_sid = session.get("qr_student_id")
        if recognized_id != qr_sid:
            flash("Face mismatch with QR student ID!", "danger")
            conn.close()
            return redirect(url_for('student_bp.face_2fa'))
        c.execute("INSERT INTO attendance (student_id, method) VALUES (%s, %s)", (recognized_id, "QR+Face"))
        conn.commit()
        conn.close()
        session["final_student_id"] = recognized_id
        # Redirect to login_success page instead of student page
        return redirect(url_for('student_bp.login_success'))
    
    elif method == "RFID":
        rfid_student_id = session.get("rfid_student_id")
        if not rfid_student_id:
            flash("RFID data missing.", "danger")
            conn.close()
            return redirect(url_for('student_bp.face_2fa'))
        if rfid_student_id != recognized_id:
            flash("Face mismatch with RFID student ID!", "danger")
            conn.close()
            return redirect(url_for('student_bp.face_2fa'))
        c.execute("INSERT INTO attendance (student_id, method) VALUES (%s, %s)", (recognized_id, "RFID+Face"))
        conn.commit()
        conn.close()
        session["final_student_id"] = recognized_id
        # Redirect to login_success page instead of student page
        return redirect(url_for('student_bp.login_success'))
    else:
        conn.close()
        flash("Invalid method in session!", "danger")
        return redirect(url_for('student_bp.student'))

@student_bp.route('/login_success')
def login_success():
    # Get student info from session or database
    student_id = session.get('final_student_id')
    
    # If we don't have a student ID in the session, redirect to login
    if not student_id:
        flash("Session expired or invalid access.", "danger")
        return redirect(url_for('student_bp.student'))
    
    # Get student information from database
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT name FROM students WHERE student_id=%s", (student_id,))
    row = c.fetchone()
    conn.close()
    
    student_name = row[0] if row else "Unknown Student"
    
    return render_template(
        "login_success.html",
        student_id=student_id,
        student_name=student_name,
        auth_method=session.get('method', '') + "+Face",
        attendance_date=datetime.now().strftime('%B %d, %Y'),
        attendance_time=datetime.now().strftime('%I:%M %p'),
        confidence_score="98"  # From your face recognition system
    )