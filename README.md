
# Multimodal Deepfake Detection Dashboard

> A research-aligned tool for detecting deepfakes using **video**, **audio**, and **textual** analysis — all in one intuitive Streamlit dashboard.

## Overview

This project integrates **computer vision**, **speech forensics**, and **NLP-based semantic analysis** to assess whether a given input is a deepfake. It is inspired by current research in **multimodal deepfake detection** and aligns with ongoing academic work on **neurosymbolic AI frameworks**.

## Features

✅ Face-based deepfake detection (frame-by-frame analysis)  
✅ Modular architecture for voice and text detection  
✅ Confidence scoring + final decision fusion  
✅ Streamlit-based UI — ready for demo, research, or education  
✅ Extensible: plug in your own models for each modality

## Project Structure

multimodal-deepfake-detector/
│
├── app.py # Streamlit UI
├── requirements.txt # Dependencies
├── README.md # Project overview
│
├── models/
│ ├── face_detector.py # Frame-by-frame CNN detection
│ ├── audio_detector.py # (Coming soon)
│ └── text_detector.py # (Coming soon)
│
├── utils/
│ ├── fusion.py # Aggregates model scores
│ └── preprocess.py # Preprocessing functions
│
├── samples/ # Test videos/audio/text
├── assets/ # Banners, icons
