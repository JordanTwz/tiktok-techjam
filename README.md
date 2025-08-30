# Privasee

A lightweight, privacy-first application that automatically detects and censors sensitive information in images.

## Overview

It uses computer vision and NLP techniques to blur or pixelate:

* Human faces
* License plates
* Personally Identifiable Information (PII) such as names, dates, card/account numbers

All processing happens **locally on-device** — no images are ever uploaded externally. Original uncensored files are stored securely and protected with a password system, ensuring that only authenticated users can view them.

---

## Features

* Automatic face & license plate detection (OpenCV Haar cascades)
* PII detection with OCR (Tesseract) + NLP (spaCy) + regex patterns
* Optional Presidio Analyzer integration for enhanced PII recognition
* Choice of Mosaic pixelation or Gaussian blur, with adjustable strength
* Password-protected viewer for originals
* Batch processing of all images in the `images/` folder
* Local-first privacy – no cloud uploads

---

## Tech Stack

* **Language:** Python 3.10+
* **UI Framework:** Streamlit
* **Libraries & Tools:**

  * OpenCV – face & license plate detection
  * Tesseract OCR – text extraction
  * spaCy – Named Entity Recognition (NER)
  * Presidio Analyzer – advanced PII detection (optional)
  * NumPy – array manipulation
  * Hashlib & JSON – password management

---

## Installation & Usage

### 1) Clone the repo

```bash
git clone https://github.com/JordanTwz/tiktok-techjam.git
cd tiktok-techjam
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Install Tesseract OCR

**Windows:**
[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get install tesseract-ocr
```

**macOS (Homebrew):**

```bash
brew install tesseract
```

> **Note:** Update the path in `backend.py` and `face_plate_censor_app.py` if Tesseract is installed elsewhere.

### 4) Run the app

**Password-protected app:**

```bash
streamlit run frontend.py
```

**Lightweight demo (no password system):**

```bash
streamlit run face_plate_censor_app.py
```

### 5) Workflow

1. Set password on first use
2. Place images in the `images/` folder
3. Censored outputs are saved in the `censored/` folder
4. Authenticate with your password to view original images
