# PHORENSICS: A Physics-Inspired Methodology for Image Manipulation Detection

> *"Pixels may be hallucinated, but photons never lie."*

**PHORENSICS (DeepScan Enterprise)** is a physics-based image forensics engine designed to detect image manipulation, deepfakes, and generative AI artifacts. 

While traditional Machine Learning (ML) and Deep Learning (DL) models suffer from cross-dataset vulnerabilities and adversarial attacks due to their reliance on training data, PHORENSICS evaluates the immutable, physical laws of light and hardware[cite: 2]. By analyzing spatial compression, Fast Fourier Transform (FFT) power decay, and CMOS sensor DNA, this methodology provides deterministic validation of an image's computational veracity[cite: 2].

---

## 🔬 Core Forensic Engines

The platform utilizes a multi-engine architecture to evaluate the optical and structural integrity of digital assets:

### 1. Global Spectral Forensics (FFT Power Decay)
Analyzes the physical diffusion of light using a 2D Fast Fourier Transform[cite: 2]. The system extracts the mathematical decay of signal energy (Alpha slope) to verify natural lens optics.
*   **Power Decay Law:** P(f) = 1/f^a[cite: 2].
*   **Human Sensitivity Tuning:** Detects manipulations that fall into the "Uncanny Valley" threat zone (a = 2.0 to 3.5)[cite: 2].

### 2. The CMOS DNA (Sensor Fingerprinting / PRNU)
Every digital sensor has microscopic silicon defects that create a unique multiplicative noise fingerprint (K)[cite: 2]. 
*   **Physical Sensor Noise Model:** I = I_0 + (I_0 * K) + Theta[cite: 2].
*   **Detection:** Copy-spliced or AI-generated regions lack this hardware signature, causing a localized fracture in the fingerprint's continuity[cite: 2].

### 3. Local Spatial Interrogation (ELA & Chroma Variance)
*   **Error Level Analysis (ELA):** Exposes invisible quantization cycles in spliced regions via localized JPEG re-compression differentials: E(x,y) = |I_orig(x,y) - I_recomp(x,y)|[cite: 2].
*   **Chroma Subsampling:** Evaluates the mathematical alignment of the 4:2:0 subsampling grid to detect color channel fractures: V_chroma = [Var(Cr) + Var(Cb)] / [Var(Y) + epsilon][cite: 2].

---

## ⚙️ Native CPU Architecture (GPU-Free)

PHORENSICS has been aggressively streamlined to operate on a fully native, pure-CPU Python environment. By stripping away heavy GPU/CUDA overhead and external JVM dependencies (like PySpark), this tool achieves maximum portability and ease of deployment on any standard machine.

*   **Universal Compatibility:** Runs natively on any CPU without requiring dedicated graphics cards, CUDA installations, or complex driver configurations.
*   **Sequential Batch Processing:** Built-in dynamic looping allows seamless ingestion and batch processing of complete image directories natively via standard CPU execution.
*   **Ultra-Lightweight Footprint:** Eliminates gigabytes of PyTorch CUDA binaries, making containerization, Hugging Face deployment, and local setup virtually effortless.

### Repository Structure

```text
deepscan-enterprise/
├── core/                # Core physics logic (CPU-optimized)
│   └── forensics.py     
├── ui/                  # Dashboard components and telemetry visualizations
│   └── visualizations.py
├── utils/               # Configurations and PDF reporting 
│   ├── config.py        
│   └── reporting.py     
├── sample_data/         # Directory for evaluation assets
├── analysis_data/       # Output logs and generated security PDFs
├── app.py               # Streamlit application entry point
└── requirements.txt     # Environment dependencies (CPU-only)
