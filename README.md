#  Automatic License Plate Recognition (ALPR) Pipeline

This project processes video frames to detect cars, localize plates, recognize characters using template matching, and stabilize results.

---

##  Script Overview

| Script | Role / Function |
|---------|-----------------|
| **main_pipeline.py** | Entry point of the system. Defines paths and calls `run_pipeline()` to start the complete ALPR process. |
| **car_detection.py** | Core controller. Handles video reading, motion detection (Optical Flow), car and plate localization, and manages CSV logging and visualization. |
| **template_loader.py** | Loads and preprocesses A–Z and 0–9 character templates into a dictionary for later matching. |
| **plate_recognition.py** | Processes the plate ROI — enhances contrast, segments characters (via projection), and performs template matching to extract plate text. |
| **stabilization.py** | Stabilizes plate recognition results across frames using majority voting to ensure consistent text output. |
| **normalize_utils.py** | Normalizes character images — centers and scales them to a fixed size for accurate template matching. |
| **output_cars/** | Folder where detected car crops are saved (optional). |
| **detected_plates_all.csv** | Output CSV containing all detections with frame number, coordinates of car ROI,coordinates of plate ROI and recognized plate text. |

---

 **Result:** A modular ALPR pipeline with detection, recognition, and stabilization.
