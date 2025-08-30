import io
import cv2
import streamlit as st
from backend import (
    load_image_to_bgr, 
    bgr_to_rgb, 
    process_image
)

# Configure Streamlit page
st.set_page_config(page_title="Face & Plate Censor", page_icon="🕶️", layout="centered")

st.title("🕶️ Face & License Plate Censor")
st.write("Upload a photo and I'll automatically blur or pixelate faces and car plate numbers. "
         "Works offline using OpenCV Haar cascades.")

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

# ------------------------------------
# Main Interface
# ------------------------------------
uploaded = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        img_bgr = load_image_to_bgr(uploaded.read())
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Process the image
    with st.spinner("Processing image..."):
        result, faces, plates, pii, warnings = process_image(
            img_bgr, want_faces, want_plates, want_pii, mode, strength
        )
    
    # Display any warnings
    for warning in warnings:
        if "Face detector" in warning:
            st.warning(warning + " Install OpenCV with data files or place the XML in your environment.")
        elif "License plate cascade" in warning:
            st.info(warning + " Plate detection may be unreliable without it. "
                   "Consider downloading 'haarcascade_russian_plate_number.xml' and placing it in OpenCV's data folder.")
        else:
            st.warning(warning)

    # Optional: draw boxes for debug on a copy
    if show_boxes:
        dbg = result.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y, w, h) in plates:
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (x, y, w, h) in pii:
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (255, 255, 0), 2)
        st.subheader("Detections (debug view)")
        st.image(bgr_to_rgb(dbg), caption="Detected regions - Green: Faces, Blue: Plates, Yellow: PII", use_column_width=True)

    # Display original and censored images side by side
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Original")
        st.image(bgr_to_rgb(img_bgr), use_column_width=True)
    with col2:
        st.subheader("Censored")
        st.image(bgr_to_rgb(result), use_column_width=True)

    # Show detection statistics
    st.subheader("Detection Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Faces Detected", len(faces))
    with col2:
        st.metric("License Plates Detected", len(plates))
    with col3:
        st.metric("PII Detected", len(pii))

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
- **Face detection** uses `haarcascade_frontalface_default.xml`. This is usually bundled with OpenCV.
- **Plate detection** uses a **Russian license plate** cascade ('haarcascade_russian_plate_number.xml') if present. It can still work on other regions, but accuracy varies.
- **PII detection** uses spaCy NER to detect names, organizations, locations, and regex patterns for bank account numbers.
- For better plate performance in Singapore or other countries, consider swapping in a more suitable detector (e.g., YOLO models trained for license plates).
- All processing is done locally in your browser session (no uploads to external servers by this app).
    """)
