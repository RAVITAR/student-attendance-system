# Student Attendance System (SAS)

A secure, real-time hybrid student attendance system developed as a final year project at the University of Bedfordshire. This system uses two-factor authentication by combining **RFID or QR code scanning** with **facial recognition** using deep learning to eliminate impersonation and attendance fraud.

---

## 👨‍🎓 Project by:
**Mustapha Olayiwola Sanni**  
BSc (Hons) Information Technology with FY  
Supervised by **Professor Vladan Velisavljevic**

---

## 🔐 Features

- Two-Factor Authentication (2FA): QR/RFID + Face
- Real-time student login with dashboard monitoring
- Face recognition using MobileFaceNet and MediaPipe
- Admin login and student enrollment
- Attendance logs stored in MySQL database
- Training pipeline for updating face recognition model
- Works on both Windows and Raspberry Pi (with MFRC522 RFID module)

---

## 🚀 Technologies Used

- Python, Flask
- OpenCV, Torch, MobileFaceNet
- MediaPipe / Dlib (fallback)
- MySQL + Connector
- HTML, CSS, JavaScript (Flask templates)
- MFRC522 (RFID hardware module)

---

## 📦 Folder Structure

```
Student-Attendance-System/
│
├── app.py                  # Flask main app
├── config.py               # Environment-secured config
├── database.py             # DB creation and connection
├── routes_student.py       # QR/RFID + face auth logic
├── routes_staff.py         # Admin routes
│
├── templates/              # HTML templates
├── static/                 # CSS, JS, QR code assets
├── models/                 # Saved face models
├── student_dataset/        # Sample face images
├── logs/                   # Recognition logs
├── requirements.txt
├── .env                    # 🔒 Hidden via .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/student-attendance-system.git
cd student-attendance-system
```

### 2. Create `.env` file
```env
FLASK_SECRET_KEY=your-secret-key
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your-db-password
MYSQL_DB=student_attendance
EMBEDDING_SECRET_KEY=your-32-byte-key
```

### 3. Create virtual environment and install requirements
```bash
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

---

## 📊 Performance

| Metric                    | Result      |
|--------------------------|-------------|
| QR Code Scan             | 100%        |
| RFID Scan                | 100%        |
| Face Recognition (1st try)| 87.5%       |
| 2FA Authentication       | 96%         |

---


---

## 📈 Future Improvements

- NFC mobile tap-in integration
- Kiosk-based terminal check-in
- Blink/liveness detection
- Cloud hosting (AWS/Azure)
- Mobile app extension

---

## 🛡️ Security Notes

- Uses `.env` to secure DB and secret keys
- Future enhancement: AES encryption for embeddings
- Admin passwords hashed using Werkzeug

---

## 📄 License

This project is for educational and demonstration purposes. For reuse or deployment, please credit the author.
