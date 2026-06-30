# 🛡️ DeepScan Enterprise: Unbiased Physics Forensics Engine

**DeepScan Enterprise** is an industrial-caliber image forensics tool designed to mathematically verify the physical authenticity of digital assets. Deployed via a Streamlit interface, this engine utilizes spatial interrogation, sub-metric Z-score distributions, and Fast Fourier Transform (FFT) analysis to detect deepfakes, Generative AI origins, and structural manipulations.

---

## 📑 Table of Contents
* [Core Architecture & Capabilities](#-core-architecture--capabilities)
* [System Prerequisites](#-system-prerequisites)
* [Installation Guide](#-installation-guide)
* [Usage & Execution](#-usage--execution)
* [Understanding the Telemetry](#-understanding-the-telemetry)
* [Troubleshooting & Hardware Notes](#-troubleshooting--hardware-notes)

---

## 🚀 Core Architecture & Capabilities

The DeepScan Engine is built on a modular architecture designed to interrogate the physical properties of light and digital encoding, rather than relying solely on easily fooled semantic AI models. 

* **Spatial Interrogation (Digital Fabric Analysis):** The engine fractures the image into localized patches to analyze Error Level Analysis (ELA) and Photo Response Non-Uniformity (PRNU). This detects localized manipulation, edge artifacts, and pasted objects by measuring microscopic inconsistencies in compression and sensor dust.
* **Spectral Forensics (FFT):** Calculates spatial frequency against signal energy using a Fast Fourier Transform. It determines if an image's underlying pixel structure adheres to natural camera lens light diffusion (The Safe Zone) or if it violently breaches physical bounds, which is a mathematical signature of Generative AI (The Threat Zone).
* **RGB Matrix Cross-Correlation:** Validates the physical overlap of the Red, Green, and Blue color channels. This catches intense chroma anomalies and forced artificial channel detachment that cannot occur in natural physics.
* **Dynamic VRAM Tracker:** Built-in hardware management using `GPUtil` and `psutil`. The system continuously monitors hardware load and will automatically hot-swap execution between CUDA (GPU) and CPU to prevent Out-Of-Memory (OOM) crashes during heavy tensor operations.
* **Distributed PySpark Integration:** Scales instantly from a single image to massive folder batches. When processing high volumes of data, the engine spins up local Spark worker nodes to parallelize the forensic pipeline.
* **Automated PDF Reporting:** Generates downloadable, executive-ready forensic telemetry reports via the `fpdf2` library, compiling heatmaps, spectral graphs, and final kinematic verdicts.

---

## ⚙️ System Prerequisites

To ensure maximum performance and avoid execution bottlenecks, your deployment environment must meet the following baseline requirements:

* **Python Version:** 3.9, 3.10, or 3.11 (Highly recommended for optimal PyTorch and PySpark stability).
* **Java Runtime Environment (JRE):** Java 8, 11, or 17+. This is **strictly required** for the PySpark distributed backend to function. *Note: The engine includes explicit JVM compatibility flags (`--add-opens`) to manage module accessibility for Java 17+ automatically.*
* **Hardware:** * *Minimum:* 8GB RAM, modern multi-core CPU.
  * *Recommended:* 16GB+ RAM, NVIDIA GPU with CUDA Toolkit installed (for accelerated PyTorch tensor calculations).

---

## 🛠️ Installation Guide

**Method 1: Python Library (Recommended)**
The project is available as a Python library named `phorensics`.

    pip install phorensics

**Method 2: Source Code**
For the full Streamlit dashboard, clone the repository and isolate the environment:

    git clone https://github.com/yourusername/DeepScan-Enterprise.git
    cd DeepScan-Enterprise
    python -m venv venv
    
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    
    pip install -r requirements.txt

---

## 💻 Usage & Execution

**Option A: CLI Library Usage**
After installing via pip, you can scan images directly from the terminal:

    python -m phorensics "path-to-image"

**Option B: Streamlit Dashboard**
Launch the Application Dashboard from the cloned repository:

    streamlit run phorensics.py
    
Upon launching, the web interface will provide two execution architectures:

* **Single Image Pipeline:** Upload a single .png, .jpg, .jpeg, .webp, .tif, or .tiff asset. The engine will instantly execute the full suite of spatial and spectral diagnostics, rendering the telemetry and allowing for immediate PDF report generation.
* **Batch/Folder Distributed (PySpark):** Upload multiple assets (dozens or hundreds) simultaneously. This triggers the Spark cluster. The UI will render real-time telemetry, allocating batches to worker nodes, tracking ETA, and providing a final Global Security Audit of all processed files.

---

## 📊 Understanding the Telemetry

The dashboard outputs complex mathematical variables that define the ultimate Kinematic Verdict. Here is how to interpret the core metrics:

* **Robust Z-Scores (Threshold = 3.5):** The engine enforces a 2.5 Z-score safe-baseline to prevent standard ISO sensor noise from triggering false positives. A Z-score crossing the 3.5 threat line indicates violent failure of standard physical constraints (e.g., missing sensor dust or highly abnormal compression clusters).
* **Alpha Value (α) & Spectral Decay:** Modern smartphone Image Signal Processors (ISPs) apply artificial sharpening. DeepScan applies an Optical Calibration Coefficient (0.70) to raw signals to normalize these real-time images against theoretical physics bounds.
  * **Safe Zone:** α < 2.0 (Natural noise) or α > 3.5 (Heavy natural blur).
  * **Threat Zone:** 2.0 ≤ α ≤ 3.5. If the calibrated signal is trapped here, it is definitively classified as a Deepfake/AI Generation.
* **RGB Matrix Violation:** Real light bleeds across channels. If the internal cross-correlation drops below 0.25 naturally, or if a severe Chroma anomaly forces the matrix to mathematically detach, the system flags a severe RGB violation.

---

## ⚠️ Troubleshooting & Hardware Notes

* **PySpark Startup Errors (Java Issues):** If the application crashes immediately upon selecting the Batch upload mode, it is almost certainly a Java configuration issue. Ensure your `JAVA_HOME` environment variable is correctly set and pointing to a valid JDK/JRE installation.
* **CUDA Out of Memory (OOM):** If you are processing massive 4K or .tiff files on a GPU with limited VRAM (e.g., < 4GB), the engine will attempt to catch the OOM error and hot-swap to the CPU. You will see a VRAM_Swaps warning in the UI if this occurs. Processing will be slower, but it will not crash.
* **Dependency Graceful Degradation:** The engine utilizes dynamic, fail-safe imports. If non-critical libraries like `GPUtil` (GPU tracking) or `fpdf2` (PDF generation) fail to install or load on your specific OS, the system will not crash. It will simply disable those specific UI elements and continue to perform its core forensic duties.
