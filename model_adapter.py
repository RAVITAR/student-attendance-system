# model_adapter.py
"""
Adapter module to maintain backward compatibility with the original model.py functions
while using the improved face recognition system.
"""

import os
import time
import logging
from pathlib import Path
from improved_model import (
    FaceRecognitionSystem, 
    FaceRecognitionConfig, 
    CONFIG
)

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ModelAdapter")

# Create face recognition system
logger.info(" Initializing FaceRecognitionSystem")
face_system = FaceRecognitionSystem()

# Compatibility constants
MODEL_PATH = CONFIG.model_path
PREDICTOR_PATH = CONFIG.face_detector_path
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def load_model():
    """
    Original load_model equivalent using the improved system.
    """
    logger.info(f" Loading model from {MODEL_PATH}")
    success = face_system._load_model()
    if success:
        logger.info(" Model loaded successfully")
    else:
        logger.warning(" Model not loaded — possibly missing or invalid")


def train_model(dataset_dir="student_dataset", do_augment=True, n_estimators=3, param_grid=None):
    """
    Original train_model/train_ensemble equivalent using the improved system.
    
    Args:
        dataset_dir: Directory containing student images
        do_augment: Whether to use data augmentation
        n_estimators: Number of estimators (ignored, kept for compatibility)
        param_grid: Parameter grid (ignored, kept for compatibility)
        
    Returns:
        Accuracy score
    """
    import time
    from datetime import datetime

    logger.info("Training session started")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Data augmentation enabled: {do_augment}")

    CONFIG.dataset_dir = dataset_dir
    CONFIG.do_augmentation = do_augment

    start_time = time.time()

    # Train the model and get accuracy
    accuracy = face_system.train_model(dataset_dir)

    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"Training duration: {elapsed:.2f} seconds")

    if os.path.exists(CONFIG.model_path):
        logger.info(f"Model saved to: {CONFIG.model_path}")
    else:
        logger.warning(f"Model path not found after training: {CONFIG.model_path}")

    logger.info(f"Training accuracy: {accuracy*100:.2f}%")
    logger.info("Training session completed\n")

    return accuracy



def capture_face_images(student_id, dataset_dir="student_dataset", num_images=5):
    """
    Capture and save images for enrollment.
    """
    logger.info(f" Starting enrollment for student ID: {student_id}")
    logger.info(f" Capturing {num_images} images")
    logger.info(f" Saving to directory: {dataset_dir}")

    CONFIG.dataset_dir = dataset_dir
    CONFIG.enrollment_images = num_images

    success = face_system.enroll_student(student_id, num_images)

    if success:
        logger.info(f" Enrollment complete for {student_id}")
    else:
        logger.error(f" Enrollment failed for {student_id}")

    return success


def aggregator_recognize_face(
    max_duration=20,
    aggregator_size=15,
    aggregator_sum_threshold=1.5,
    conf_min=0.02
):
    """
    Face recognition with real-time aggregation.
    """
    import traceback
    import threading
    import queue

    logger.info(" Starting face recognition")
    logger.info(f" Timeout: {max_duration}s")
    logger.info(f" Aggregator threshold: {aggregator_sum_threshold}")
    logger.info(f" Confidence minimum: {conf_min}")

    try:
        CONFIG.recognition_timeout = max_duration
        CONFIG.aggregator_threshold = aggregator_sum_threshold
        CONFIG.confidence_threshold = conf_min

        # Queue to store the result
        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def recognize_with_result():
            try:
                # Capture raw recognition result
                raw_result = face_system.recognize_face()
                
                # Log raw result for debugging
                logger.debug(f"Raw recognition result: {raw_result}")

                # Validate result structure
                if not isinstance(raw_result, dict):
                    raise ValueError(f"Invalid result type: {type(raw_result)}")

                # Check for required keys
                required_keys = ["status", "label", "reason"]
                for key in required_keys:
                    if key not in raw_result:
                        raise KeyError(f"Missing key in result: {key}")

                # Additional validation for successful recognition
                if raw_result["status"] == "ok":
                    # Ensure label exists and is not None
                    if not raw_result.get("label"):
                        raise ValueError("Recognition successful but no label provided")

                result_queue.put(raw_result)

            except Exception as e:
                logger.error(f"Recognition processing error: {str(e)}")
                exception_details = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                logger.error(f"Error details: {exception_details}")
                exception_queue.put(exception_details)

        # Start recognition in a separate thread
        recognition_thread = threading.Thread(target=recognize_with_result)
        recognition_thread.daemon = True
        recognition_thread.start()

        # Wait for the thread to complete or timeout
        recognition_thread.join(timeout=max_duration)

        # Check for exceptions
        if not exception_queue.empty():
            exception = exception_queue.get()
            logger.error(f"Exception in face recognition: {exception}")
            
            return {
                "status": "error",
                "label": None,
                "reason": f"Recognition processing error: {exception.get('error_message', 'Unknown error')}",
                "error_details": exception
            }

        # Check if recognition completed
        if recognition_thread.is_alive():
            logger.error("Face recognition timed out")
            return {
                "status": "error",
                "label": None,
                "reason": "Face recognition timed out"
            }

        # Get the result
        result = result_queue.get() if not result_queue.empty() else None

        # Final validation
        if not result:
            logger.error("No recognition result obtained")
            return {
                "status": "error",
                "label": None,
                "reason": "No recognition result"
            }

        # Log and return the result
        if result["status"] == "ok":
            logger.info(f"Recognition successful — Label: {result['label']} | Confidence: {result.get('confidence', 'N/A')}")
            # Log detailed confidence data if available
            if "confidence_details" in result:
                logger.debug(f"Confidence Details: {result['confidence_details']}")
        else:
            logger.warning(f"Recognition failed — Reason: {result['reason']}")

        return result

    except Exception as e:
        logger.error(f"Unexpected error in face recognition: {str(e)}")
        return {
            "status": "error",
            "label": None,
            "reason": f"Unexpected error: {str(e)}",
            "error_details": {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
        }

if __name__ == "__main__":
    logger.info("Running training process manually from model_adapter.py")
    acc = train_model()
    logger.info(f"Training completed. Accuracy: {acc * 100:.2f}%")


# Make sure the models directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)