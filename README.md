[README.md](https://github.com/user-attachments/files/24722499/README.md)

# Medical Imaging Classifier

A Streamlit-based application for classifying chest X‑ray images using a custom convolutional neural network.  
The system predicts **Normal** vs **Pneumonia**, provides confidence scores, and generates a downloadable PDF report.

This project is an academic prototype focused on machine learning experimentation and user‑friendly model visualization.  
It is **not** a medical device and must not be used for diagnostic or clinical decisions.

---

## Features

- Streamlit web interface  
- Image upload and preprocessing  
- Pneumonia classification using a custom CNN  
- Real‑time prediction with probability score  
- Automatic PDF report generation  
- Device‑aware inference (CPU, CUDA, MPS)

---

## How It Works

1. Upload a chest X‑ray image (JPG or PNG).  
2. The app applies the appropriate preprocessing pipeline.  
3. The model outputs:
   - predicted class (Normal / Pneumonia)  
   - probability score  
   - confidence level  
4. A structured PDF report can be generated and downloaded.

---

## Tech Stack

- Python  
- PyTorch  
- Streamlit  
- PIL  
- ReportLab  

The training code, model weights, and internal architecture are intentionally excluded from the repository.

---

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Disclaimer

This tool is for **research and educational purposes only**.  
It is not approved for medical use and must not replace professional interpretation.

