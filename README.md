# 🛡️ DeepScan Enterprise: Unbiased Physics Forensics Engine

**DeepScan Enterprise** is an industrial-caliber image forensics tool designed to mathematically verify the physical authenticity of digital assets. Deployed via a Streamlit interface, this engine utilizes spatial interrogation, sub-metric Z-score distributions, and Fast Fourier Transform (FFT) analysis to detect deepfakes, Generative AI origins, and structural manipulations.

The architecture is built for both single-asset deep dives and high-velocity big data analysis using distributed PySpark worker nodes.

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

*   **Spatial Interrogation (Digital Fabric Analysis):** Analyzes Error Level Analysis (ELA) and Photo Response Non-Uniformity (PRNU/Sensor Noise) to detect localized manipulation and pasted artifacts.
*   **Spectral Forensics (FFT):** Calculates spatial frequency against signal energy to determine if an image adheres to natural lens light diffusion (Safe Zone) or breaches physical bounds (Generative AI Threat Zone).
*   **RGB Matrix Cross-Correlation:** Validates the physical overlap of color channels to catch intense chroma anomalies and forced artificial channel detachment.
*   **Dynamic VRAM Tracker:** Built-in hardware management using `GPUtil` and `psutil` that automatically hot-swaps execution between CUDA (GPU) and CPU to prevent Out-Of-Memory (OOM) crashes during heavy tensor operations.
*   **Distributed PySpark Integration:** Scales instantly from a single image to massive folder batches by spinning up local Spark worker nodes for parallel processing.
*   **Automated PDF Reporting:** Generates downloadable, executive-ready forensic telemetry reports via FPDF2.

---

## ⚙️ System Prerequisites

Ensure your environment meets the following requirements before deployment:

*   **Python:** 3.9+ 
*   **Java (Required for PySpark Backend):** Java 8, 11, or 17+ (The engine includes explicit JVM compatibility flags for Java 17+).
*   **Hardware (Optional but Recommended):** An NVIDIA GPU with CUDA support for accelerated PyTorch tensor calculations.

---

## 🛠️ Installation Guide

**1. Clone the Repository**
```bash
git clone [https://github.com/yourusername/DeepScan-Enterprise.git](https://github.com/yourusername/DeepScan-Enterprise.git)
cd DeepScan-Enterprise
