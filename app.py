import os
import sys
import logging
from datetime import datetime
from flask import Flask, render_template, session, flash, redirect, url_for
from flask_wtf.csrf import CSRFProtect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-fallback-secret")

# Import your custom modules
from database import create_database, create_tables
from routes_student import student_bp
from routes_staff import staff_bp
from model_adapter import MODEL_PATH, load_model, train_model
from extensions import csrf

# Attach CSRF protection
csrf.init_app(app)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AttendanceSystem")

# Raspberry Pi Setup (optional)
IS_RPI = False
rfid_reader = None
if sys.platform.startswith("linux"):
    try:
        from mfrc522 import SimpleMFRC522
        rfid_reader = SimpleMFRC522()
        IS_RPI = True
        logger.info("Running on Raspberry Pi, MFRC522 ready.")
    except ImportError:
        logger.warning("MFRC522 not installed, skipping Pi RFID logic.")
else:
    logger.info("Not on Raspberry Pi, skipping Pi-specific RFID logic.")

# ----------------------------------------
# DATABASE + MODEL INITIALIZATION
# ----------------------------------------

def initialize_model():
    """Trains or loads face recognition model"""
    if not os.path.exists(MODEL_PATH):
        logger.info("Model not found. Training...")
        acc = train_model(dataset_dir="student_dataset", do_augment=True)
        app.config["MODEL_ACCURACY"] = f"{acc*100:.2f}"
    else:
        logger.info("Loading pre-trained model...")
        load_model()
        if "MODEL_ACCURACY" not in app.config:
            app.config["MODEL_ACCURACY"] = "N/A"

create_database()
create_tables()
initialize_model()

# ----------------------------------------
# ROUTES & CONTEXT
# ----------------------------------------

app.register_blueprint(student_bp)
app.register_blueprint(staff_bp)

@app.context_processor
def inject_context():
    return {
        'current_year': datetime.now().year,
        'model_accuracy': app.config.get("MODEL_ACCURACY", "N/A")
    }

@app.route('/')
def home():
    return render_template("index.html")

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

# ----------------------------------------
# START APP
# ----------------------------------------

if __name__ == '__main__':
    logger.info("Starting Flask server in debug mode...")
    app.run(debug=True)
