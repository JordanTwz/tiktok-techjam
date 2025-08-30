import io
import cv2
import streamlit as st
import os
import hashlib
import json
from pathlib import Path
from backend import *

# Configuration
IMAGES_DIR = "images"
CENSORED_DIR = "censored"
CONFIG_FILE = "config.json"

# Create directories if they don't exist
Path(IMAGES_DIR).mkdir(exist_ok=True)
Path(CENSORED_DIR).mkdir(exist_ok=True)

# Configure Streamlit page
st.set_page_config(page_title="Face & Plate Censor", page_icon="🕶️", layout="centered")

# Password Management
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"password_hash": None}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password, password_hash):
    return hash_password(password) == password_hash

def setup_password():
    config = load_config()
    if config.get("password_hash") is None:
        st.sidebar.subheader("First-time Setup")
        password = st.sidebar.text_input("Set a password", type="password")
        confirm_password = st.sidebar.text_input("Confirm password", type="password")
        
        if st.sidebar.button("Set Password") and password:
            if password == confirm_password:
                config["password_hash"] = hash_password(password)
                save_config(config)
                st.sidebar.success("Password set successfully!")
                return True
            else:
                st.sidebar.error("Passwords do not match")
                return False
    return True

def authenticate():
    config = load_config()
    if config.get("password_hash") is None:
        return False
    
    st.sidebar.subheader("Authentication")
    password = st.sidebar.text_input("Enter password", type="password")
    if st.sidebar.button("Authenticate"):
        if check_password(password, config["password_hash"]):
            st.session_state.authenticated = True
            st.sidebar.success("Authenticated!")
        else:
            st.sidebar.error("Incorrect password")
    
    return st.session_state.get("authenticated", False)

# Main Application
st.title("🕶️ Face & License Plate Censor")
st.write("Automatically censor images in the 'images' folder and view them with password protection.")

# Password setup and authentication
if not setup_password():
    st.stop()

authenticated = authenticate()

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

# Process images button
if st.sidebar.button("Process All Images"):
    with st.spinner("Processing all images..."):
        process_all_images(mode, strength, want_faces, want_plates, want_pii)
    st.success("All images processed successfully!")

# Display censored images
st.header("Censored Images")
censored_files = get_censored_images()

if not censored_files:
    st.info("No censored images found. Process images first.")
else:
    # Display images in a grid
    cols = st.columns(3)
    for i, filename in enumerate(censored_files):
        col = cols[i % 3]
        
        # Display censored image
        censored_path = os.path.join(CENSORED_DIR, filename)
        col.image(censored_path, caption=filename, use_column_width=True)
        
        # Show original button if authenticated
        if authenticated:
            original_path = os.path.join(IMAGES_DIR, filename)
            if os.path.exists(original_path):
                if col.button(f"View Original", key=f"orig_{filename}"):
                    original_img = get_original_image(filename)
                    col.image(bgr_to_rgb(original_img), caption=f"Original: {filename}", use_column_width=True)
            else:
                col.warning("Original not found")
        else:
            col.info("Authenticate to view original")

# Upload new images section
st.header("Upload New Images")
uploaded = st.file_uploader("Upload a JPG or PNG image to censor", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded:
    for file in uploaded:
        try:
            img_bgr = load_image_to_bgr(file.read())
            filename = file.name
            
            # Save original image
            original_path = os.path.join(IMAGES_DIR, filename)
            cv2.imwrite(original_path, img_bgr)
            
            # Process and save censored image
            with st.spinner(f"Processing {filename}..."):
                result, faces, plates, pii, warnings = process_image(
                    img_bgr, want_faces, want_plates, want_pii, mode, strength
                )
                
                censored_path = os.path.join(CENSORED_DIR, filename)
                cv2.imwrite(censored_path, result)
            
            st.success(f"Processed and saved {filename}")
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")

# Footer notes
with st.expander("Notes & Tips"):
    st.markdown("""
- Place images to be censored in the images folder
- Censored images are saved in the censored folder
- Face detection uses haarcascade_frontalface_default.xml
- Plate detection uses a Russian license plate cascade if available
- PII detection uses spaCy NER to detect names, organizations, locations, and regex patterns
- Set a password on first run to protect original images
- All processing is done locally
    """)