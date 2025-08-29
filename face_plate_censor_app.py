import os
import io
import cv2
import spacy
import numpy as np
import streamlit as st
import pytesseract
import re
from presidio_analyzer import AnalyzerEngine

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="Face & Plate Censor", page_icon="🕶️", layout="centered")

st.title("🕶️ Face & License Plate Censor")
st.write("Upload a photo and I'll automatically blur or pixelate faces and car plate numbers. "
         "Works offline using OpenCV Haar cascades.")

# ------------------------------------
# Helpers
# ------------------------------------
def load_image_to_bgr(file_bytes: bytes) -> np.ndarray:
    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid JPG/PNG.")
    return img

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def apply_mosaic(roi: np.ndarray, pixel_size: int = 20) -> np.ndarray:
    h, w = roi.shape[:2]
    down_w = max(1, w // pixel_size)
    down_h = max(1, h // pixel_size)
    small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_gaussian(roi: np.ndarray, ksize: int = 31) -> np.ndarray:
    # ksize must be odd
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(roi, (ksize, ksize), 0)

def censor_region(img_bgr: np.ndarray, x: int, y: int, w: int, h: int, mode: str, strength: int) -> None:
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

@st.cache_resource
def get_cascades():
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
    face_cascade, plate_cascade, face_xml_path, plate_xml_paths = get_cascades()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = []
    plates = []

    if want_faces:
        if face_cascade is not None and not face_cascade.empty():
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
        else:
            st.warning("Face detector (haarcascade_frontalface_default.xml) not found in your OpenCV installation. "
                       "Install OpenCV with data files or place the XML in your environment.")

    if want_plates:
        if plate_cascade is not None and not plate_cascade.empty():
            # License plates are often wider than tall, so a slightly larger minSize helps
            plates = plate_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 20)
            )
        else:
            st.info("License plate cascade not found (looked for Russian plate cascades). "
                    "Plate detection may be unreliable without it. "
                    "Consider downloading 'haarcascade_russian_plate_number.xml' and placing it in OpenCV's data folder.")
    return faces, plates

def detect_pii_presidio(img_bgr):
    # Get word-level OCR data
    ocr_data = pytesseract.image_to_data(img_bgr, output_type=pytesseract.Output.DICT)
    words = ocr_data['text']
    left = ocr_data['left']
    top = ocr_data['top']
    width = ocr_data['width']
    height = ocr_data['height']

    # Reconstruct full text for Presidio
    full_text = ' '.join(words)

    # Initialize Presidio Analyzer
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=full_text, entities=[], language='en')

    # Find coordinates for PII entities
    pii_boxes = []
    for result in results:
        pii_text = full_text[result.start:result.end]
        # Find matching word(s) and their bounding boxes
        for i, word in enumerate(words):
            if word and pii_text.strip() in word:
                box = (left[i], top[i], left[i] + width[i], top[i] + height[i])
                pii_boxes.append(box)

    return pii_boxes 


def detect_pii_spacy(img_bgr):
    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")

    
    # Get word-level OCR data
    ocr_data = pytesseract.image_to_data(img_bgr, output_type=pytesseract.Output.DICT)
    words = ocr_data['text']
    left = ocr_data['left']
    top = ocr_data['top']
    width = ocr_data['width']
    height = ocr_data['height']
    
    # Reconstruct full text for spaCy
    full_text = ' '.join(words)
    
    
    pii_boxes = []
    
    # Use spaCy to detect entities
    doc = nlp(full_text)
    bank_pattern = r'^\d{8,16}$'
    for i, word in enumerate(words):
        if re.fullmatch(bank_pattern, word):
            box = (left[i], top[i], left[i] + width[i], top[i] + height[i])
            pii_boxes.append(box)

    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG", "LOC", "DATE", "TIME", "MONEY", "CARDINAL", "FAC"]:  # Add more labels as needed
            ent_text = ent.text.strip()
            # Find matching word(s) and their bounding boxes
            for i, word in enumerate(words):
                if word and ent_text in word:
                    box = (left[i], top[i], left[i] + width[i], top[i] + height[i])
                    pii_boxes.append(box)
    return pii_boxes

# ------------------------------------
# Sidebar controls
# ------------------------------------
st.sidebar.header("Censor Settings")
mode = st.sidebar.selectbox("Blur style", ["Mosaic", "Gaussian Blur"])
strength = st.sidebar.slider("Strength (larger = stronger)", min_value=5, max_value=60, value=25, step=1)
want_faces = st.sidebar.checkbox("Censor faces", value=True)
want_plates = st.sidebar.checkbox("Censor license plates")
want_pii = st.sidebar.checkbox("Censor PII", value=True)
show_boxes = st.sidebar.checkbox("Show detection boxes (debug)", value=False)

uploaded = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        img_bgr = load_image_to_bgr(uploaded.read())
    except Exception as e:
        st.error(str(e))
        st.stop()

    faces, plates = detect_faces_and_plates(img_bgr, want_faces, want_plates)
    pii = detect_pii_spacy(img_bgr)

    result = img_bgr.copy()
    # Apply censorship
    if want_faces:
        for (x, y, w, h) in faces:
            censor_region(result, x, y, w, h, mode, strength)

    if want_plates:
        for (x, y, w, h) in plates:
            censor_region(result, x, y, w, h, mode, strength)

    for (x, y, w, h) in pii:
        censor_region(result, x, y, w, h, mode, strength)

    # Optional: draw boxes for debug on a copy
    if show_boxes:
        dbg = result.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y, w, h) in plates:
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (255, 0, 0), 2)
        st.subheader("Detections (debug view)")
        st.image(bgr_to_rgb(dbg), caption="Detected regions", use_column_width=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Original")
        st.image(bgr_to_rgb(img_bgr), use_column_width=True)
    with col2:
        st.subheader("Censored")
        st.image(bgr_to_rgb(result), use_column_width=True)

    # Download censored image
    ok, png_bytes = cv2.imencode(".png", result)
    if ok:
        st.download_button(
            label="⬇️ Download censored image (PNG)",
            data=io.BytesIO(png_bytes.tobytes()),
            file_name="censored.png",
            mime="image/png",
        )

# Footer notes
with st.expander("Notes & Tips"):
    st.markdown("""
- Face detection uses `haarcascade_frontalface_default.xml`. This is usually bundled with OpenCV.
- Plate detection uses a **Russian license plate** cascade ('haarcascade_russian_plate_number.xml') if present. It can still work on other regions, but accuracy varies.
- For better plate performance in Singapore or other countries, consider swapping in a more suitable detector (e.g., YOLO models trained for license plates).
- All processing is done locally in your browser session (no uploads to external servers by this app).
    """)
