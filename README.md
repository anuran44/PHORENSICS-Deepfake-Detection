# PHORENSICS: A Physics-Inspired Methodology for Image Manipulation Detection

> *"Pixels may be hallucinated, but photons never lie."*

**PHORENSICS (DeepScan Enterprise)** is an industrial-grade, physics-based image forensics engine designed to detect image manipulation, deepfakes, and generative AI artifacts. 

While traditional Machine Learning (ML) and Deep Learning (DL) models suffer from cross-dataset vulnerabilities and adversarial attacks due to their reliance on training weights, PHORENSICS evaluates the immutable, physical laws of light and hardware. By analyzing spatial compression, Fast Fourier Transform (FFT) power decay, and CMOS sensor DNA, this methodology provides deterministic validation of an image's computational veracity.

---

## 🔬 Core Forensic Engines

The platform utilizes a multi-engine architecture to evaluate the optical and structural integrity of digital assets:

### 1. Global Spectral Forensics (FFT Power Decay)
Analyzes the physical diffusion of light using a 2D Fast Fourier Transform. The system extracts the mathematical decay of signal energy (Alpha slope) to verify natural lens optics.
* **Power Decay Law:** P(f) = 1/f^α
* **Human Sensitivity Tuning:** Detects manipulations that fall into the "Uncanny Valley" threat zone (α = 2.0 to 3.5).

### 2. The CMOS DNA (Sensor Fingerprinting / PRNU)
Every digital sensor has microscopic silicon defects that create a unique multiplicative noise fingerprint (K). 
* **Physical Sensor Noise Model:** I = I_0 + (I_0 * K) + Θ
* **Detection:** Copy-spliced or AI-generated regions lack this hardware signature, causing a localized fracture in the fingerprint's continuity.

### 3. Local Spatial Interrogation (ELA & Chroma Variance)
* **Error Level Analysis (ELA):** Exposes invisible quantization cycles in spliced regions via localized JPEG re-compression differentials: E(x,y) = |I_orig(x,y) - I_recomp(x,y)|.
* **Chroma Subsampling:** Evaluates the mathematical alignment of the 4:2:0 subsampling grid to detect color channel fractures: V_chroma = [Var(Cr) + Var(Cb)] / [Var(Y) + ε].

---

## ⚙️ Native Architecture & Hardware Management

PHORENSICS operates on a fully native Python environment, stripping away heavy external JVM dependencies for maximum portability and ease of deployment.

* **Sequential Batch Processing:** Built-in dynamic looping allows seamless ingestion and batch processing of complete image directories natively via the CPU/GPU.
* **Adaptive Compute Manager:** A dynamic VRAM tracking module that automatically hot-swaps tensor computations between CUDA and CPU to prevent Out-Of-Memory (OOM) errors during heavy matrix operations.

### Repository Structure

```text
deepscan-enterprise/
├── core/                # Core logic, physics engines, and VRAM management
│   ├── hardware.py      
│   └── forensics.py     
├── ui/                  # Dashboard components and telemetry visualizations
│   └── visualizations.py
├── utils/               # Configurations and PDF reporting 
│   ├── config.py        
│   └── reporting.py     
├── sample_data/         # Directory for evaluation assets
├── analysis_data/       # Output logs and generated security PDFs
├── app.py               # Streamlit application entry point
└── requirements.txt     # Environment dependencies
```

---

## 📊 Benchmarking Performance

PHORENSICS outperforms probabilistic Vision Large Language Models (VLLMs) such as LLaVA, Gemma 4, and Qwen 3.5 on zero-day generative deepfakes. It has been rigorously benchmarked against the following datasets:

| Dataset | Organization | Real Assets | Fake Assets | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **AI-GUARD** | Cranfield University | 204,317 | 247,123 | 78.0% |
| **Intel Scenes** | Intel | 59,000 | 61,500 | 79.0% |
| **FFHQ** | Nvidia | 146,000 | 164,200 | 78.5% |
| **SHURA** | Sheffield University | 220 | 220 | 80.0% |
| **Authenti Face** | UC Berkeley | 82,000 | 108,851 | 77.6% |

*Data derived from PHORENSICS testing against standard datasets.*

---

## 🚀 Quickstart Guide

### Prerequisites
* Python 3.8+

### Installation
1. Clone the repository:
   `git clone https://github.com/your-username/deepscan-enterprise.git`
   `cd deepscan-enterprise`
2. Install dependencies:
   `pip install -r requirements.txt`

### Execution
Launch the DeepScan OS dashboard:
`streamlit run app.py`

---

## 🎓 About

This repository represents the codebase for the **PHORENSICS** methodology, developed as part of an M.Tech Dissertation focusing on physics-based image manipulation detection.
