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
st.set_page_config(page_title="Secure Photo Gallery", layout="centered")

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'refresh_needed' not in st.session_state:
    st.session_state.refresh_needed = False

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

def isNewUser():
    config = load_config()
    return (config.get("password_hash") is None)

def setup_password():
    st.write("Welcome! Set up your authentication password first to start using this app.")
    password = st.text_input("Set a password", type="password", autocomplete="new-password", key="setup_pw")
    confirm_password = st.text_input("Confirm password", type="password", autocomplete="new-password", key="setup_pw_confirm")
        
    if st.button("Set Password") and password:
        if password == confirm_password:
            config = load_config()
            config["password_hash"] = hash_password(password)
            save_config(config)
            st.success("Password set successfully! You can now authenticate.")
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Passwords do not match")

def authenticate():
    config = load_config()
    if config.get("password_hash") is None:
        return False
    
    password = st.text_input("Enter password", type="password", autocomplete="current-password", key="auth_pw")
    if st.button("Authenticate"):
        if check_password(password, config["password_hash"]):
            st.session_state.authenticated = True
            st.success("Authentication successful!")
            st.rerun()
        else:
            st.error("Incorrect password")
    
    return st.session_state.authenticated

# Function to display the image gallery
def display_image_gallery():
    censored_files = get_censored_images()
    
    if not censored_files:
        st.info("No censored images found. Upload images to get started.")
    else:
        # Display images in a grid
        cols = st.columns(3)
        for i, filename in enumerate(censored_files):
            col = cols[i % 3]
            
            # Display censored image
            censored_path = os.path.join(CENSORED_DIR, filename)
            col.image(censored_path, caption=filename, use_column_width=True)
            
            # Show original button if authenticated
            if st.session_state.authenticated:
                original_path = os.path.join(IMAGES_DIR, filename)
                if os.path.exists(original_path):
                    if col.button(f"View Original", key=f"orig_{filename}"):
                        original_img = get_original_image(filename)
                        col.image(bgr_to_rgb(original_img), caption=f"Original: {filename}", use_column_width=True)
                else:
                    col.warning("Original not found")
            else:
                col.info("Authenticate to view original")

# Main Application
st.title("Secure Photo Gallery")

# Authentication flow
if isNewUser():
    setup_password()
elif not st.session_state.authenticated:
    st.write("Images with private or sensitive information blurred. Unlock them with password.")
    authenticate()
else:
    st.write("Images unlocked! View the original images below.")

# Only show the main content if authenticated or if there are images to show
if st.session_state.authenticated or get_censored_images():
    mode = "Mosaic"  # Can choose between Mosaic and Gaussian Blur
    strength = 45  # between 5 and 60
    
    # Display censored images
    st.header("Images")
    display_image_gallery()
    
    # Upload new images section
    st.header("Upload New Images")
    uploaded = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded:
        for file in uploaded:
            try:
                # Check if this file was already processed in this session
                if file.name in st.session_state.processed_images:
                    continue
                    
                img_bgr = load_image_to_bgr(file.read())
                filename = file.name
                
                # Save original image
                original_path = os.path.join(IMAGES_DIR, filename)
                cv2.imwrite(original_path, img_bgr)
                
                # Process and save censored image
                with st.spinner(f"Processing {filename}..."):
                    result, faces, plates, pii, warnings = process_image(
                        img_bgr, mode, strength
                    )
                    
                    censored_path = os.path.join(CENSORED_DIR, filename)
                    cv2.imwrite(censored_path, result)
                
                st.success(f"Processed and saved {filename}")
                st.session_state.processed_images.append(filename)
                st.session_state.refresh_needed = True
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
        
        # Force a refresh after processing
        if st.session_state.refresh_needed:
            st.session_state.refresh_needed = False
            st.rerun()