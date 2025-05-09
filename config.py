# config.py
import os
from dotenv import load_dotenv
import torch

# Load .env variables
load_dotenv()

# Torch config
device = torch.device('cpu')  # or 'cuda' if using GPU
emb_size = 512
num_classes = 1000

# MySQL config (securely loaded)
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "student_attendance")

# Flask secret key
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

# AES or embedding key (optional, for future use)
EMBEDDING_SECRET_KEY = os.getenv("EMBEDDING_SECRET_KEY", "changeme")