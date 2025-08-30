import os
import cv2
import spacy
import numpy as np
import pytesseract
import re
from pathlib import Path

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configuration
IMAGES_DIR = "images"
CENSORED_DIR = "censored"

# Create directories if they don't exist
Path(IMAGES_DIR).mkdir(exist_ok=True)
Path(CENSORED_DIR).mkdir(exist_ok=True)

# ------------------------------------
# Image Processing Functions
# ------------------------------------
def load_image_to_bgr(file_bytes: bytes) -> np.ndarray:
    """Load image from bytes and convert to BGR format."""
    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid JPG/PNG.")
    return img

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB format."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def apply_mosaic(roi: np.ndarray, pixel_size: int = 20) -> np.ndarray:
    """Apply mosaic/pixelation effect to a region of interest."""
    h, w = roi.shape[:2]
    down_w = max(1, w // pixel_size)
    down_h = max(1, h // pixel_size)
    small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_gaussian(roi: np.ndarray, ksize: int = 31) -> np.ndarray:
    """Apply Gaussian blur to a region of interest."""
    # ksize must be odd
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(roi, (ksize, ksize), 0)

def censor_region(img_bgr: np.ndarray, x: int, y: int, w: int, h: int, mode: str, strength: int) -> None:
    """Apply censoring (mosaic or blur) to a specific region of the image."""
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)
    x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
    roi = img_bgr[y:y2, x:x2]
    if roi.size == 0:
        return
    if mode == "Mosaic":
        censored = apply_mosaic(roi, pixel_size=max(4, strength))
    else:
        censored = apply_gaussian(roi, ksize=max(5, strength | 1))  # ensure odd
    img_bgr[y:y2, x:x2] = censored

# ------------------------------------
# Detection Functions
# ------------------------------------
def get_cascades():
    """Load OpenCV cascade classifiers for face and license plate detection."""
    # Try to find cascade files in the OpenCV data folder
    cascade_dir = cv2.data.haarcascades
    face_xml = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
    # Try common plate cascade file names
    plate_xml_candidates = [
        "haarcascade_russian_plate_number.xml",
        "haarcascade_licence_plate_rus_16stages.xml",
    ]

    face_cascade = cv2.CascadeClassifier(face_xml) if os.path.exists(face_xml) else None

    plate_cascade = None
    for name in plate_xml_candidates:
        path = os.path.join(cascade_dir, name)
        if os.path.exists(path):
            plate_cascade = cv2.CascadeClassifier(path)
            break

    return face_cascade, plate_cascade, face_xml, [os.path.join(cv2.data.haarcascades, n) for n in plate_xml_candidates]

def detect_faces_and_plates(img_bgr: np.ndarray, want_faces: bool, want_plates: bool):
    """Detect faces and license plates in an image using OpenCV cascades."""
    face_cascade, plate_cascade, face_xml_path, plate_xml_paths = get_cascades()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = []
    plates = []
    warnings = []

    if want_faces:
        if face_cascade is not None and not face_cascade.empty():
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
        else:
            warnings.append("Face detector (haarcascade_frontalface_default.xml) not found in your OpenCV installation.")

    if want_plates:
        if plate_cascade is not None and not plate_cascade.empty():
            # License plates are often wider than tall, so a slightly larger minSize helps
            plates = plate_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 20)
            )
        else:
            warnings.append("License plate cascade not found (looked for Russian plate cascades).")
    
    return faces, plates, warnings

def detect_pii_spacy(img_bgr):
    """Detect PII using spaCy NER and regex patterns."""
    # Load spaCy English model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Return empty list if spaCy model is not available
        return []
    
    # Get word-level OCR data
    try:
        ocr_data = pytesseract.image_to_data(img_bgr, output_type=pytesseract.Output.DICT)
        words = ocr_data['text']
        left = ocr_data['left']
        top = ocr_data['top']
        width = ocr_data['width']
        height = ocr_data['height']
    except:
        # Return empty list if OCR fails
        return []
    
    # Reconstruct full text for spaCy
    full_text = ' '.join(words)
    
    pii_boxes = []
    
    # Use spaCy to detect entities
    doc = nlp(full_text)
    
    # Regex for bank account numbers (8-16 digits)
    bank_pattern = r'^\d{8,16}$'
    for i, word in enumerate(words):
        if re.fullmatch(bank_pattern, word):
            box = (left[i], top[i], left[i] + width[i], top[i] + height[i])
            pii_boxes.append(box)

    # spaCy NER for names, locations, organizations, etc.
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG", "LOC", "DATE", "TIME", "MONEY", "CARDINAL", "FAC"]:
            ent_text = ent.text.strip()
            # Find matching word(s) and their bounding boxes
            for i, word in enumerate(words):
                if word and ent_text in word:
                    box = (left[i], top[i], left[i] + width[i], top[i] + height[i])
                    pii_boxes.append(box)
    
    return pii_boxes

def process_image(img_bgr: np.ndarray, want_faces: bool, want_plates: bool, want_pii: bool, mode: str, strength: int):
    """Main processing function that applies all detections and censoring."""
    result = img_bgr.copy()
    warnings = []
    
    # Detect faces and plates
    faces, plates, detection_warnings = detect_faces_and_plates(img_bgr, want_faces, want_plates)
    warnings.extend(detection_warnings)
    
    # Detect PII
    pii = []
    if want_pii:
        try:
            pii = detect_pii_spacy(img_bgr)
        except Exception as e:
            warnings.append(f"PII detection failed: {str(e)}")
    
    # Apply censorship
    if want_faces:
        for (x, y, w, h) in faces:
            censor_region(result, x, y, w, h, mode, strength)

    if want_plates:
        for (x, y, w, h) in plates:
            censor_region(result, x, y, w, h, mode, strength)

    if want_pii:
        for (x, y, w, h) in pii:
            censor_region(result, x, y, w, h, mode, strength)
    
    return result, faces, plates, pii, warnings

def process_all_images(mode, strength, want_faces, want_plates, want_pii):
    """Process all images in the images directory"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(IMAGES_DIR) 
                  if f.lower().endswith(image_extensions)]
    
    for filename in image_files:
        image_path = os.path.join(IMAGES_DIR, filename)
        img_bgr = cv2.imread(image_path)
        if img_bgr is not None:
            result, faces, plates, pii, warnings = process_image(
                img_bgr, want_faces, want_plates, want_pii, mode, strength
            )
            censored_path = os.path.join(CENSORED_DIR, filename)
            cv2.imwrite(censored_path, result)

def get_censored_images():
    """Get list of censored images"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return [f for f in os.listdir(CENSORED_DIR) 
            if f.lower().endswith(image_extensions)]

def get_original_image(filename):
    """Load original image from images directory"""
    original_path = os.path.join(IMAGES_DIR, filename)
    return cv2.imread(original_path)