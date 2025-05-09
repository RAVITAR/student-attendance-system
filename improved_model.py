"""
Advanced Face Recognition System with Deep Learning and Optimized Pipeline
"""
import os
import cv2
import time
import numpy as np
import logging
import joblib
import platform
import threading
import random
import yaml
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
from mobilefacenet import MobileFaceNet


# Deep learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from tqdm import tqdm

# Optional imports with fallbacks
try:
    import mediapipe as mp
    USE_MEDIAPIPE = True
except ImportError:
    import dlib
    USE_MEDIAPIPE = False

try:
    if platform.system().lower() == "windows":
        import winsound
        CAN_BEEP = True
    else:
        import subprocess
        CAN_BEEP = True
except ImportError:
    CAN_BEEP = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("face_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FaceRecognition")

#######################################################################
#                    Configuration Management                         #
#######################################################################

@dataclass
class FaceRecognitionConfig:
    """Centralized configuration for the face recognition system"""
    # Paths
    model_path: str = "models/face_recognition_model.pkl"
    face_detector_path: str = "shape_predictor_68_face_landmarks.dat"
    dataset_dir: str = "student_dataset"
    log_dir: str = "logs"
    
    # Model parameters
    use_deep_learning: bool = True
    embedding_size: int = 512
    use_gpu: bool = torch.cuda.is_available()
    pca_variance: float = 0.95
    
    # Recognition parameters
    confidence_threshold: float = 0.7
    aggregator_threshold: float = 2.0
    recognition_timeout: int = 15
    min_face_size: int = 60
    
    # Enrollment parameters
    enrollment_images: int = 10
    
    # Augmentation parameters
    do_augmentation: bool = True
    max_augmentations: int = 5
    
    # UI and feedback
    use_feedback_sound: bool = CAN_BEEP
    show_debug_ui: bool = True
    
    # Security
    encrypt_embeddings: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'FaceRecognitionConfig':
        """Load configuration from YAML file"""
        if not os.path.exists(yaml_path):
            logger.warning(f"Config file {yaml_path} not found, using defaults")
            return cls()
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

# Global configuration instance
CONFIG = FaceRecognitionConfig()

#######################################################################
#                    Face Detection Components                        #
#######################################################################

class FaceDetector:
    """Abstract base class for face detection"""
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in the image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face information dictionaries containing:
            - 'bbox': (x, y, w, h)
            - 'landmarks': Dict of facial landmarks
            - 'confidence': Detection confidence
        """
        raise NotImplementedError
    
    def align_face(self, image: np.ndarray, face_info: Dict[str, Any]) -> np.ndarray:
        """
        Align face to canonical position
        
        Args:
            image: Input image
            face_info: Face information from detect_faces
            
        Returns:
            Aligned face image (100x100)
        """
        raise NotImplementedError


class MediaPipeFaceDetector(FaceDetector):
    """Face detector using MediaPipe Face Mesh"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return []
        
        faces = []
        for face_landmarks in results.multi_face_landmarks:
            # Extract face bounding box
            h, w, _ = image.shape
            landmarks = []
            
            for i, landmark in enumerate(face_landmarks.landmark):
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmarks.append((x, y))
            
            landmarks = np.array(landmarks)
            
            # Calculate bounding box
            x1, y1 = np.min(landmarks, axis=0)
            x2, y2 = np.max(landmarks, axis=0)
            
            # Add margin to the bounding box
            margin = int(0.1 * max(x2-x1, y2-y1))
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            # Create face info dictionary
            faces.append({
                'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                'landmarks': {
                    'left_eye': np.mean(landmarks[33:36], axis=0),
                    'right_eye': np.mean(landmarks[362:365], axis=0),
                    'all': landmarks
                },
                'confidence': 1.0  # MediaPipe doesn't provide confidence
            })
        
        return faces
    
    def align_face(self, image: np.ndarray, face_info: Dict[str, Any]) -> np.ndarray:
        # Get landmarks
        left_eye = face_info['landmarks']['left_eye']
        right_eye = face_info['landmarks']['right_eye']
        
        # Calculate angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate scale
        eye_distance = np.sqrt(dx*dx + dy*dy)
        desired_distance = 40.0
        scale = desired_distance / (eye_distance + 1e-8)
        
        # Center point
        center_x = (left_eye[0] + right_eye[0]) / 2.0
        center_y = (left_eye[1] + right_eye[1]) / 2.0
        
        # Get transformation matrix
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
        M[0, 2] += (50 - center_x)
        M[1, 2] += (50 - center_y)
        
        # Apply transformation
        aligned = cv2.warpAffine(image, M, (100, 100), flags=cv2.INTER_CUBIC)
        return aligned


class DlibFaceDetector(FaceDetector):
    """Face detector using dlib (fallback when MediaPipe is not available)"""
    
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(CONFIG.face_detector_path)
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray, 0)
        
        if not dets:
            return []
        
        faces = []
        for det in dets:
            shape = self.predictor(gray, det)
            landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            # Calculate bounding box
            x, y = det.left(), det.top()
            w, h = det.right() - x, det.bottom() - y
            
            # Create face info dictionary
            faces.append({
                'bbox': (x, y, w, h),
                'landmarks': {
                    'left_eye': np.mean(landmarks[36:42], axis=0),
                    'right_eye': np.mean(landmarks[42:48], axis=0),
                    'all': landmarks
                },
                'confidence': 1.0  # dlib doesn't provide confidence
            })
        
        return faces
    
    def align_face(self, image: np.ndarray, face_info: Dict[str, Any]) -> np.ndarray:
        # Get landmarks
        left_eye = face_info['landmarks']['left_eye']
        right_eye = face_info['landmarks']['right_eye']
        
        # Calculate angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        if np.sqrt(dx*dx + dy*dy) < 2.0:
            # Eyes too close, fallback to crop
            x, y, w, h = face_info['bbox']
            face = image[y:y+h, x:x+w]
            return cv2.resize(face, (100, 100))
        
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate scale
        eye_distance = np.sqrt(dx*dx + dy*dy)
        desired_distance = 40.0
        scale = desired_distance / (eye_distance + 1e-8)
        
        # Center point
        center_x = (left_eye[0] + right_eye[0]) / 2.0
        center_y = (left_eye[1] + right_eye[1]) / 2.0
        
        # Get transformation matrix
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
        M[0, 2] += (50 - center_x)
        M[1, 2] += (50 - center_y)
        
        # Apply transformation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        aligned = cv2.warpAffine(gray, M, (100, 100), flags=cv2.INTER_CUBIC)
        return aligned

#######################################################################
#                    Feature Extraction & Models                      #
#######################################################################

class FaceEmbedder:
    """Abstract base class for face embedding extraction"""
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        # Ensure face image is RGB (3 channels)
        if len(face_image.shape) == 2 or face_image.shape[2] == 1:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        elif face_image.shape[2] == 4:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2BGR)

        # Resize to expected input size if needed (100x100)
        if face_image.shape[:2] != (100, 100):
            face_image = cv2.resize(face_image, (100, 100))

        # Normalize and convert to tensor
        face_tensor = self.transform(face_image).unsqueeze(0).to(self.device)  # Shape: [1, 3, 100, 100]

        with torch.no_grad():
            embedding = self.model(face_tensor)

        return embedding.cpu().numpy().flatten()



class DeepFaceEmbedder(FaceEmbedder):
    """Face embedder using a deep learning model (MobileFaceNet architecture)"""
    
    def __init__(self):
        self.device = torch.device('cuda' if CONFIG.use_gpu else 'cpu')
        self.model = self._build_model().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _build_model(self) -> nn.Module:
        model = MobileFaceNet()
        state_dict = torch.load("models/face_embedder.pth", map_location=self.device) # No weights_only
        model.load_state_dict(state_dict)
        model.eval()
        return model


    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from face image with robust shape handling
        
        Args:
            face_image: Input face image
        
        Returns:
            Numpy array of face embedding
        """
        logger.debug(f"Input image shape: {face_image.shape}")
        
        try:
            # Handle image shape variations
            if face_image is None:
                logger.error("Received None image")
                return None
            
            # Normalize image dimensions
            if len(face_image.shape) == 2:
                # Grayscale image - add channel dimension
                logger.debug("Converting 2D grayscale to 3D")
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif len(face_image.shape) == 3:
                # Check number of channels
                if face_image.shape[2] == 1:
                    # Single channel image
                    logger.debug("Converting single channel to RGB")
                    face_image = cv2.cvtColor(face_image.squeeze(), cv2.COLOR_GRAY2RGB)
                elif face_image.shape[2] == 4:
                    # RGBA image
                    logger.debug("Converting RGBA to RGB")
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
                elif face_image.shape[2] == 3:
                    # Ensure color order is correct (OpenCV uses BGR)
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                else:
                    logger.error(f"Unexpected image shape: {face_image.shape}")
                    return None
            else:
                logger.error(f"Invalid image shape: {face_image.shape}")
                return None
            
            # Resize to expected input size
            if face_image.shape[:2] != (100, 100):
                logger.debug(f"Resizing image from {face_image.shape[:2]} to (100, 100)")
                face_image = cv2.resize(face_image, (100, 100), interpolation=cv2.INTER_CUBIC)
            
            # Normalize and convert to tensor
            face_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(face_tensor)
            
            return embedding.cpu().numpy().flatten()
        
        except Exception as e:
            logger.error(f"Unexpected error in embedding extraction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


class HOGFaceEmbedder(FaceEmbedder):
    """Face embedder using HOG features (fallback for non-deep-learning approach)"""
    
    def __init__(self):
        # HOG parameters
        self.cell_size = (8, 8)
        self.block_size = (2, 2)
        self.nbins = 9
    
    def safe_extract_embedding(aligned_face):
        """Safely extract embedding with multiple fallback attempts"""
        attempts = 0
        while attempts < 3:
            try:
                embedding = safe_extract_embedding(aligned_face)
                if embedding is None:
                    continue  # Skip this frame and continue recognition
                if embedding is not None:
                    return embedding
            except Exception as e:
                logger.warning(f"Embedding extraction attempt {attempts+1} failed: {str(e)}")
            
            # Try alternative preprocessing
            if attempts == 0:
                # Try converting to grayscale
                aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
            elif attempts == 1:
                # Try resizing
                aligned_face = cv2.resize(aligned_face, (100, 100))
            
            attempts += 1
        
        logger.error("Failed to extract embedding after multiple attempts")
        return None

#######################################################################
#                    Data Augmentation Pipeline                       #
#######################################################################

class FaceAugmenter:
    """Data augmentation for face images"""
    
    def __init__(self, max_augmentations: int = 5):
        self.max_augmentations = max_augmentations
    
    def augment(self, face_image: np.ndarray) -> List[np.ndarray]:
        """
        Apply random augmentations to a face image
        
        Args:
            face_image: Input face image
            
        Returns:
            List of augmented face images including the original
        """
        results = [face_image]
        
        # Determine how many augmentations to create
        n_augs = random.randint(1, self.max_augmentations)
        
        # Define augmentation types and their probabilities
        aug_types = [
            ('flip', 0.5),
            ('brightness', 0.7),
            ('rotation', 0.4),
            ('noise', 0.3),
            ('blur', 0.3),
            ('perspective', 0.2),
            ('occlusion', 0.2)
        ]
        
        # Apply random augmentations
        for _ in range(n_augs):
            # Select an augmentation based on probabilities
            aug_type = random.choices(
                [aug[0] for aug in aug_types],
                weights=[aug[1] for aug in aug_types],
                k=1
            )[0]
            
            # Apply the selected augmentation
            aug_method = getattr(self, f"_augment_{aug_type}")
            augmented = aug_method(face_image.copy())
            results.append(augmented)
        
        return results
    
    def _augment_flip(self, image: np.ndarray) -> np.ndarray:
        """Horizontal flip"""
        return cv2.flip(image, 1)
    
    def _augment_brightness(self, image: np.ndarray) -> np.ndarray:
        """Random brightness adjustment"""
        shift = random.randint(-40, 40)
        return np.clip(image.astype(np.int32) + shift, 0, 255).astype(np.uint8)
    
    def _augment_rotation(self, image: np.ndarray) -> np.ndarray:
        """Random rotation"""
        h, w = image.shape[:2]
        angle = random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), 
                             flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_REFLECT_101)
    
    def _augment_noise(self, image: np.ndarray) -> np.ndarray:
        """Add random noise"""
        noise = np.random.normal(0, 10, image.shape).astype(np.int32)
        return np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    
    def _augment_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur"""
        ksize = random.choice([3, 5])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    
    def _augment_perspective(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective transformation"""
        h, w = image.shape[:2]
        
        # Define the 4 source points
        src_pts = np.array([
            [0, 0],
            [w-1, 0],
            [w-1, h-1],
            [0, h-1]
        ], dtype=np.float32)
        
        # Define the 4 destination points with random perturbation
        max_shift = 0.05 * min(h, w)
        dst_pts = np.array([
            [0 + random.uniform(0, max_shift), 0 + random.uniform(0, max_shift)],
            [w-1 - random.uniform(0, max_shift), 0 + random.uniform(0, max_shift)],
            [w-1 - random.uniform(0, max_shift), h-1 - random.uniform(0, max_shift)],
            [0 + random.uniform(0, max_shift), h-1 - random.uniform(0, max_shift)]
        ], dtype=np.float32)
        
        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply perspective transformation
        return cv2.warpPerspective(image, M, (w, h), 
                                  flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_REFLECT_101)
    
    def _augment_occlusion(self, image: np.ndarray) -> np.ndarray:
        """Add random occlusion rectangles"""
        h, w = image.shape[:2]
        occ = image.copy()
        
        # Number of occlusion rectangles
        n_rects = random.randint(1, 3)
        
        for _ in range(n_rects):
            occ_w = random.randint(5, int(w * 0.3))
            occ_h = random.randint(5, int(h * 0.3))
            x0 = random.randint(0, w - occ_w)
            y0 = random.randint(0, h - occ_h)
            
            # Fill with random color or black
            if random.random() < 0.5:
                color = random.randint(0, 255)
            else:
                color = 0
                
            cv2.rectangle(occ, (x0, y0), (x0+occ_w, y0+occ_h), color, -1)
            
        return occ

#######################################################################
#                    Recognition System                               #
#######################################################################

class FaceRecognitionSystem:
    """Main class for face recognition system"""
    
    def __init__(self, config: FaceRecognitionConfig = None):
        """Initialize the face recognition system"""
        global CONFIG
        if config is not None:
            CONFIG = config
        
        # Initialize components
        self.detector = MediaPipeFaceDetector() if USE_MEDIAPIPE else DlibFaceDetector()
        self.embedder = DeepFaceEmbedder() if CONFIG.use_deep_learning else HOGFaceEmbedder()
        self.augmenter = FaceAugmenter(max_augmentations=CONFIG.max_augmentations)
        
        # Initialize model variables
        self.model = None
        self.scaler = None
        self.pca = None
        self.label_map = {}
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load the recognition model if it exists"""
        if not os.path.exists(CONFIG.model_path):
            logger.warning(f"No model found at {CONFIG.model_path}")
            return False
        
        try:
            model_data = joblib.load(CONFIG.model_path)
            
            if isinstance(model_data, tuple) and len(model_data) == 4:
                self.scaler, self.pca, self.model, self.label_map = model_data
                logger.info(f"Loaded recognition model from {CONFIG.model_path}")
                logger.info(f"Loaded {len(self.label_map)} identities: {list(self.label_map.values())}")
                return True
            else:
                logger.error(f"Invalid model format in {CONFIG.model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _save_model(self) -> bool:
        """Save the recognition model"""
        try:
            os.makedirs(os.path.dirname(CONFIG.model_path), exist_ok=True)
            joblib.dump((self.scaler, self.pca, self.model, self.label_map), CONFIG.model_path)
            logger.info(f"Saved recognition model to {CONFIG.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def enroll_student(self, student_id: str, num_images: int = None) -> bool:
        """
        Enroll a new student with face images
        
        Args:
            student_id: Student ID to enroll
            num_images: Number of images to capture (default: from config)
            
        Returns:
            Success status
        """
        if num_images is None:
            num_images = CONFIG.enrollment_images
        
        # Create dataset directory if it doesn't exist
        dataset_dir = CONFIG.dataset_dir
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create student directory
        student_dir = os.path.join(dataset_dir, student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        # Open camera
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            logger.error("Camera not accessible")
            return False
        
        # Setup progress indicators
        count = 0
        logger.info(f"Enrolling {student_id}, capturing {num_images} images...")
        
        try:
            while count < num_images:
                # Read frame
                ret, frame = cam.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Create overlay for countdown
                overlay = frame.copy()
                
                # Detect face
                faces = self.detector.detect_faces(frame)
                if not faces:
                    cv2.putText(overlay, "No face detected", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Face Enrollment", overlay)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                    continue
                
                # Get largest face
                face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
                x, y, w, h = face['bbox']
                
                # Draw rectangle
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Countdown
                for i in range(3, 0, -1):
                    countdown = overlay.copy()
                    cv2.putText(countdown, f"Capturing in {i}...", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Face Enrollment", countdown)
                    
                    if CONFIG.use_feedback_sound:
                        self._beep()
                    
                    if cv2.waitKey(800) & 0xFF == ord('q'):
                        break
                
                # Align face
                aligned_face = self.detector.align_face(frame, face)
                
                # Save aligned face
                path = os.path.join(student_dir, f"{count}.jpg")
                cv2.imwrite(path, aligned_face)
                logger.info(f"Saved {path}")
                count += 1
                
                # Display success message
                success = overlay.copy()
                cv2.putText(success, f"Captured {count}/{num_images}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Face Enrollment", success)
                cv2.waitKey(500)
                
        except Exception as e:
            logger.error(f"Error during enrollment: {str(e)}")
            return False
        finally:
            cam.release()
            cv2.destroyAllWindows()
        
        return count > 0
    
    def _beep(self):
        """Make a beep sound for feedback"""
        if not CONFIG.use_feedback_sound:
            return
        
        try:
            if platform.system().lower() == "windows":
                winsound.Beep(440, 120)
            else:
                # Use subprocess to play a sound on Linux/Mac
                subprocess.call(["aplay", "-q", "beep.wav"], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
        except Exception:
            pass
    
    def train_model(self, dataset_dir: str = None) -> float:
        """
        Train the face recognition model
        
        Args:
            dataset_dir: Directory containing student face images
            
        Returns:
            Accuracy score on test set
        """
        if dataset_dir is None:
            dataset_dir = CONFIG.dataset_dir
        
        logger.info(f"Loading dataset from {dataset_dir}")
        
        # Load dataset
        X, y, label_map = self._load_dataset(dataset_dir)
        if not X or len(X) == 0:
            logger.error("No data => training aborted")
            return 0.0
        
        # Convert to NumPy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        self.label_map = label_map
        
        # Check for minimum requirements
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.error("Need at least 2 classes to train")
            return 0.0
        
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {len(unique_classes)} classes")
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA if needed (for high-dimensional features)
        self.pca = None
        if X_train.shape[1] > 100 and not CONFIG.use_deep_learning:
            self.pca = PCA(n_components=CONFIG.pca_variance, whiten=True)
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)
            logger.info(f"Applied PCA: reduced dimensions from {X_train.shape[1]} to {X_train_scaled.shape[1]}")
        
        # Train model
        if CONFIG.use_deep_learning:
            from sklearn.neighbors import KNeighborsClassifier
            
            # Use KNN for embeddings from deep learning model
            self.model = KNeighborsClassifier(
                n_neighbors=min(5, len(X_train_scaled) // len(unique_classes)),
                weights='distance',
                metric='cosine'
            )
            self.model.fit(X_train_scaled, y_train)
            logger.info("Trained KNN classifier for deep learning embeddings")
        else:
            from sklearn.svm import SVC
            from sklearn.ensemble import VotingClassifier
            
            # Use ensemble of SVMs for traditional features
            estimators = []
            for i in range(3):  # Train 3 SVMs with different random states
                svm = SVC(
                    C=10.0, 
                    gamma='scale', 
                    kernel='rbf', 
                    probability=True, 
                    random_state=42+i
                )
                estimators.append((f'svm_{i}', svm))
            
            self.model = VotingClassifier(estimators=estimators, voting='soft')
            self.model.fit(X_train_scaled, y_train)
            logger.info("Trained SVM ensemble classifier")

        logger.info("Classifier training complete. Evaluating accuracy...")

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy*100:.2f}%")
        
        # Generate detailed report
        if logger.level <= logging.INFO:
            report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            logger.info(f"Classification report:\n{report}")
            logger.info(f"Confusion matrix:\n{conf_matrix}")
        
        # Save model
        self._save_model()
        
        return accuracy
    
    def _load_dataset(self, dataset_dir: str) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
        """
        Load face images dataset from directory structure
        
        Args:
            dataset_dir: Directory containing subdirectories of student images
            
        Returns:
            Tuple of (feature_list, label_list, label_map)
        """
        X = []  # Features
        y = []  # Labels
        label_map = {}  # Numeric label -> student ID
        
        # List subdirectories (each is a student)
        subfolders = sorted([f for f in os.listdir(dataset_dir) 
                            if os.path.isdir(os.path.join(dataset_dir, f))])
        
        if not subfolders:
            logger.warning(f"No student subdirectories found in {dataset_dir}")
            return [], [], {}
        
        # Process each student folder
        for idx, folder in enumerate(subfolders):
            student_id = folder
            student_path = os.path.join(dataset_dir, folder)
            label_map[idx] = student_id
            
            # Get image files
            image_files = sorted([f for f in os.listdir(student_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            if not image_files:
                logger.warning(f"No images found for student {student_id}")
                continue
            
            logger.info(f"Processing {len(image_files)} images for {student_id}")
            
            # Process each image
            # Process each image with tqdm progress bar
            for img_file in tqdm(image_files, desc=f"{student_id} images"):
                img_path = os.path.join(student_path, img_file)
                try:
                    # Load image
                    img = cv2.imread(img_path)
                    if img is None:
                        logger.warning(f"Failed to load image: {img_path}")
                        continue

                    if CONFIG.use_deep_learning:
                        embedding = self.embedder.extract_embedding(img)
                        X.append(embedding)
                        y.append(idx)
                    else:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        if CONFIG.do_augmentation:
                            augmented_imgs = self.augmenter.augment(gray)
                            for aug_img in augmented_imgs:
                                features = self.embedder.extract_embedding(aug_img)
                                X.append(features)
                                y.append(idx)
                        else:
                            features = self.embedder.extract_embedding(gray)
                            X.append(features)
                            y.append(idx)
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {str(e)}")

            logger.info(f"Finished student {student_id} ({len(image_files)} images)")

            
            logger.info(f"Added {len(image_files)} images for student {student_id}")
        
        logger.info(f"Loaded dataset with {len(X)} samples from {len(label_map)} students")
        return X, y, label_map
    
    def recognize_face(self, max_duration: int = None, confidence_threshold: float = None) -> Dict[str, Any]:
        """
        Perform face recognition with advanced aggregation
        
        Args:
            max_duration: Maximum duration in seconds (default: from config)
            confidence_threshold: Confidence threshold (default: from config)
            
        Returns:
            Recognition result dictionary
        """
        if max_duration is None:
            max_duration = CONFIG.recognition_timeout
        
        if confidence_threshold is None:
            confidence_threshold = CONFIG.confidence_threshold
        
        # Check if model is loaded
        if self.model is None:
            if not self._load_model():
                return {
                    "status": "error",
                    "label": None,
                    "reason": "No recognition model loaded"
                }
        
        # Open camera
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return {
                "status": "error",
                "label": None,
                "reason": "Camera not accessible"
            }
        
        # Setup aggregator
        aggregator = {}  # label -> cumulative confidence
        recognized_label = None
        start_time = time.time()
        
        # UI elements
        spinner = ["◐", "◓", "◑", "◒"]  # Unicode spinner
        spinner_idx = 0
        
        try:
            while True:
                # Read frame
                ret, frame = cam.read()
                if not ret:
                    return {
                        "status": "error",
                        "label": None,
                        "reason": "Failed to read frame"
                    }
                
                # Create overlay
                overlay = frame.copy()
                
                # Update spinner
                cv2.putText(overlay, f"Recognition {spinner[spinner_idx]}", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                spinner_idx = (spinner_idx + 1) % len(spinner)
                
                # Detect faces
                faces = self.detector.detect_faces(frame)
                
                if not faces:
                    cv2.putText(overlay, "No face detected", (20, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # Get largest face
                    face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
                    x, y, w, h = face['bbox']
                    
                    # Draw rectangle
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Align and extract embedding
                    aligned_face = self.detector.align_face(frame, face)
                    embedding = self.embedder.extract_embedding(aligned_face)
                    
                    # Scale and transform embedding
                    embedding_scaled = self.scaler.transform([embedding])
                    if self.pca is not None:
                        embedding_scaled = self.pca.transform(embedding_scaled)
                    
                    # Get prediction and confidence
                    probs = self.model.predict_proba(embedding_scaled)[0]
                    pred_label = np.argmax(probs)
                    confidence = probs[pred_label]
                    
                    # Display confidence
                    if confidence >= confidence_threshold:
                        cv2.putText(overlay, f"Conf: {confidence:.2f}", (20, 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Update aggregator
                        aggregator[pred_label] = aggregator.get(pred_label, 0.0) + confidence
                    else:
                        cv2.putText(overlay, f"Low conf: {confidence:.2f}", (20, 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Display aggregator status
                    y_offset = 150
                    for label_idx, sum_val in sorted(aggregator.items(), 
                                                  key=lambda x: x[1], 
                                                  reverse=True)[:3]:  # Show top 3
                        student_id = self.label_map.get(label_idx, f"Unknown-{label_idx}")
                        status_text = f"{student_id}: {sum_val:.2f}"
                        cv2.putText(overlay, status_text, (20, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        y_offset += 30
                        
                        # Check if we have a winner
                        if sum_val >= CONFIG.aggregator_threshold:
                            recognized_label = label_idx
                            if CONFIG.use_feedback_sound:
                                self._beep()
                            break
                
                # Show elapsed time
                elapsed = time.time() - start_time
                time_color = (0, 255, 0) if elapsed < max_duration * 0.7 else (0, 165, 255)
                if elapsed > max_duration * 0.9:
                    time_color = (0, 0, 255)
                
                cv2.putText(overlay, f"Time: {elapsed:.1f}s / {max_duration}s", (20, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, time_color, 2)
                
                # Show frame
                cv2.imshow("Face Recognition", overlay)
                
                # Check for exit keys
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    return {
                        "status": "error",
                        "label": None,
                        "reason": "User cancelled recognition"
                    }
                
                # Check if we have a winner
                if recognized_label is not None:
                    break
                
                # Check timeout
                if elapsed > max_duration:
                    return {
                        "status": "error",
                        "label": None,
                        "reason": "Recognition timeout - insufficient confidence"
                    }
        
        finally:
            # Clean up
            cam.release()
            cv2.destroyAllWindows()
        
        # Return recognition result
        if recognized_label not in self.label_map:
            return {
                "status": "error",
                "label": None,
                "reason": f"Unknown label index: {recognized_label}"
            }
        
        student_id = self.label_map[recognized_label]
        return {
            "status": "ok",
            "label": student_id,
            "reason": "Recognition successful",
            "confidence": aggregator[recognized_label]
        }

#######################################################################
#                          Main CLI Interface                         #
#######################################################################

def main():
    """Command-line interface for the face recognition system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Face Recognition System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Enroll command
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a new student")
    enroll_parser.add_argument("student_id", type=str, help="Student ID to enroll")
    enroll_parser.add_argument("--images", type=int, default=None, 
                              help="Number of images to capture")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the recognition model")
    train_parser.add_argument("--dataset", type=str, default=None,
                             help="Dataset directory path")
    
    # Recognize command
    recognize_parser = subparsers.add_parser("recognize", help="Perform face recognition")
    recognize_parser.add_argument("--timeout", type=int, default=None,
                                help="Recognition timeout in seconds")
    recognize_parser.add_argument("--threshold", type=float, default=None,
                                help="Confidence threshold")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("--save", type=str, help="Save configuration to file")
    config_parser.add_argument("--load", type=str, help="Load configuration from file")
    config_parser.add_argument("--show", action="store_true", help="Show current configuration")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize system
    system = FaceRecognitionSystem()
    
    # Execute command
    if args.command == "enroll":
        logger.info(f"Enrolling student {args.student_id}")
        result = system.enroll_student(args.student_id, args.images)
        if result:
            logger.info(f"Successfully enrolled {args.student_id}")
        else:
            logger.error(f"Failed to enroll {args.student_id}")
    
    elif args.command == "train":
        logger.info("Training recognition model")
        accuracy = system.train_model(args.dataset)
        logger.info(f"Training completed with accuracy: {accuracy*100:.2f}%")
    
    elif args.command == "recognize":
        logger.info("Starting face recognition")
        result = system.recognize_face(args.timeout, args.threshold)
        if result["status"] == "ok":
            logger.info(f"Recognized: {result['label']} with confidence {result['confidence']:.2f}")
        else:
            logger.error(f"Recognition failed: {result['reason']}")
    
    elif args.command == "config":
        if args.save:
            CONFIG.to_yaml(args.save)
            logger.info(f"Configuration saved to {args.save}")
        
        if args.load:
            new_config = FaceRecognitionConfig.from_yaml(args.load)
            system = FaceRecognitionSystem(new_config)
            logger.info(f"Configuration loaded from {args.load}")
        
        if args.show:
            import pprint
            logger.info("Current configuration:")
            pprint.pprint(CONFIG.__dict__)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()