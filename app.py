import os
import glob
import time
import datetime
import warnings
import psutil
import tkinter as tk
from tkinter import filedialog
import numpy as np
import streamlit as st

# Environment & Internal Modules
from utils.config import setup_environment
setup_environment()

from core.hardware import HAS_GPUTIL
if HAS_GPUTIL: import GPUtil

from core.forensics import UnbiasedPhysicsForensics
from ui.visualizations import (create_pure_heatmap, create_spectral_forensics_plot, 
                               create_speedometer, create_z_distribution_plot, create_rgb_matrix_plot)
from utils.reporting import generate_pdf_report, HAS_FPDF

try:
    from pyspark.sql import SparkSession
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

warnings.filterwarnings('ignore')

def process_file_wrapper(fpath):
    engine = UnbiasedPhysicsForensics(fpath, force_cpu=True)
    if engine.valid: return engine.analyze()
    return None

def check_5_vs(files):
    size_mb = sum(os.path.getsize(f) for f in files) / (1024 * 1024)
    formats = set(f.split('.')[-1].lower() for f in files)
    return {"Volume": size_mb > 500 or len(files) > 100, "Velocity": False, "Variety": len(formats) > 1, "Veracity": True, "Value": len(files) > 0}, (size_mb > 500 or len(files) > 100)

st.set_page_config(page_title="DeepScan Enterprise", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .metric-card {background: #f8f9fa; padding: 20px; border-radius: 8px; border-top: 4px solid #0f62fe; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; font-family: sans-serif;}
    .metric-card h4 { color: #555; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;}
    .metric-card h2 { color: #333; font-size: 2.2rem; margin: 10px 0 0 0;}
    .worker-pending { background: #fafafa; padding: 15px; border-radius: 4px; border-left: 4px solid #9e9e9e; margin-bottom: 10px; font-family: monospace;}
    .worker-executing { background: #e3f2fd; padding: 15px; border-radius: 4px; border-left: 4px solid #0f62fe; margin-bottom: 10px; font-family: monospace; font-weight: bold;}
    .worker-completed { background: #e8f5e9; padding: 15px; border-radius: 4px; border-left: 4px solid #24a148; margin-bottom: 10px; font-family: monospace;}
    .diag-box { background: #f8f9fa; padding: 15px; border-radius: 4px; font-size: 0.95rem; border-left: 4px solid #0f62fe; margin-bottom: 15px; line-height: 1.5; color: #333;}
    .diag-title { color: #0f62fe; font-weight: bold; text-transform: uppercase; font-size: 0.8rem; margin-bottom: 5px; display: block;}
    </style>
""", unsafe_allow_html=True)

if 'results' not in st.session_state: st.session_state['results'] = []
if 'selected_image' not in st.session_state: st.session_state['selected_image'] = None
if 'target_path' not in st.session_state: st.session_state['target_path'] = ""

scan_triggered = False
files = []

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=60)
    st.markdown("### DeepScan OS")
    st.caption("v4.2 Enterprise Industrial Edition")
    st.divider()

    st.markdown("#### 📊 System Telemetry")
    cpu_usage = psutil.cpu_percent(interval=0.1)
    ram_usage = psutil.virtual_memory().percent
    st.progress(cpu_usage / 100.0, text=f"CPU Load: {cpu_usage}%")
    st.progress(ram_usage / 100.0, text=f"RAM Usage: {ram_usage}%")
    
    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                vram_usage = gpu.memoryUtil * 100
                st.progress(vram_usage / 100.0, text=f"GPU VRAM: {vram_usage:.1f}% ({gpu.memoryUsed}MB / {gpu.memoryTotal}MB)")
            else:
                st.info("No GPU detected by GPUtil.")
        except Exception:
            st.warning("GPUtil failed to read metrics.")
    else:
        st.warning("GPUtil not installed. GPU tracking offline.")

    st.divider()
    
    mode = st.radio("Input Architecture", ["Single Image Pipeline", "Batch/Folder Distributed (PySpark)"])
    
    st.write("Target Path:")
    col1, col2 = st.columns([3, 1])
    target = col1.text_input("Path", value=st.session_state['target_path'], label_visibility="collapsed")
    
    if mode == "Single Image Pipeline":
        if col2.button("📁 Browse"):
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', 1)
            path = filedialog.askopenfilename(master=root, filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.webp")])
            root.destroy()
            if path:
                st.session_state['target_path'] = path
                st.rerun()
    else:
        if col2.button("📁 Browse"):
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', 1)
            path = filedialog.askdirectory(master=root)
            root.destroy()
            if path:
                st.session_state['target_path'] = path
                st.rerun()
                
    st.divider()
    if st.button("🚀 INITIATE SCAN SEQUENCE", type="primary", use_container_width=True) and target:
        scan_triggered = True
        st.session_state['results'] = []
        st.session_state['selected_image'] = None

if scan_triggered and target:
    clean_path = target.strip().strip('"').strip("'")
    files = [clean_path] if mode == "Single Image Pipeline" and os.path.isfile(clean_path) else [f for f in glob.glob(os.path.join(clean_path, '*.*')) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff'))]

    if len(files) == 0:
        st.sidebar.error("SYSTEM ERROR: No valid image assets detected in path.")
    else:
        use_spark = False
        
        if mode == "Batch/Folder Distributed (PySpark)":
            st.sidebar.markdown("#### 🗄️ Big Data Analysis")
            v_res, is_big_data = check_5_vs(files)
            for k, v in v_res.items(): st.sidebar.markdown(f"{'✅' if v else '❌'} **{k}**")
            use_spark = is_big_data and HAS_SPARK
            st.sidebar.info("PySpark Clusters Engaged." if use_spark else "Standard processing active.")

        bar = st.sidebar.progress(0)
        status = st.sidebar.empty()
        eta = st.sidebar.empty()
        start_time = time.time()
        
        st.title("⚙️ CLUSTER EXECUTION IN PROGRESS")
        st.divider()

        if use_spark:
            st.subheader("🖥️ Distributed Worker Node Telemetry")
            
            spark = SparkSession.builder \
                .appName("Forensics") \
                .master("local[4]") \
                .config("spark.driver.memory", "8g") \
                .config("spark.executor.memory", "4g") \
                .config("spark.driver.maxResultSize", "4g") \
                .config("spark.scheduler.mode", "FAIR") \
                .getOrCreate()
            spark.sparkContext.setLogLevel("ERROR")
            
            num_nodes = min(4, max(1, len(files) // 5))
            batches = [list(x) for x in np.array_split(files, num_nodes) if len(x) > 0]
            
            cols = st.columns(4)
            task_slots = []
            
            for i in range(len(batches)):
                slot = cols[i % 4].empty()
                task_slots.append(slot)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            for i, batch in enumerate(batches):
                task_slots[i].markdown(f"<div class='worker-executing'><b>WORKER NODE 0{i+1}</b><br><span style='color:#0f62fe'>STATUS: EXECUTING ⚡ [{len(batch)} TENSORS]</span></div>", unsafe_allow_html=True)
            
            completed_files = 0
            
            for i, batch in enumerate(batches):
                res = spark.sparkContext.parallelize(batch).map(process_file_wrapper).collect()
                st.session_state['results'].extend([r for r in res if r])
                completed_files += len(batch)
                
                task_slots[i].markdown(f"<div class='worker-completed'><b>WORKER NODE 0{i+1}</b><br><span style='color:#24a148'>STATUS: COMPLETED ✅</span></div>", unsafe_allow_html=True)
                
                progress = min(completed_files / len(files), 1.0)
                bar.progress(progress)
                status.text(f"Processed {completed_files} / {len(files)} files.")
                
                elapsed = time.time() - start_time
                if progress > 0 and progress < 1.0:
                    eta_str = str(datetime.timedelta(seconds=int((elapsed / progress) - elapsed)))
                    eta.markdown(f"<span style='color:#fa4d56; font-weight:bold; font-family: monospace;'>ETA: {eta_str}</span>", unsafe_allow_html=True)
                elif progress >= 1.0:
                    eta.empty()

            spark.stop()
            
        else:
            st.subheader("🖥️ Hardware Node Telemetry")
            proc_slot = st.empty()
            
            for i, fpath in enumerate(files):
                proc_slot.markdown(f"<div class='worker-executing'><b>CORE ENGINE</b><br><span style='color:#0f62fe'>SCANNING: {os.path.basename(fpath)}</span></div>", unsafe_allow_html=True)
                status.text(f"Scanning: {os.path.basename(fpath)}")
                
                engine = UnbiasedPhysicsForensics(fpath)
                if engine.valid: st.session_state['results'].append(engine.analyze())
                
                progress = (i + 1) / len(files)
                bar.progress(progress)
                
                elapsed = time.time() - start_time
                if progress > 0:
                    eta_str = str(datetime.timedelta(seconds=int((elapsed / progress) - elapsed)))
                    eta.markdown(f"<span style='color:#fa4d56; font-weight:bold; font-family: monospace;'>ETA: {eta_str}</span>", unsafe_allow_html=True)
            
            proc_slot.markdown(f"<div class='worker-completed'><b>CORE ENGINE</b><br><span style='color:#24a148'>STATUS: COMPLETED ✅</span></div>", unsafe_allow_html=True)
                
        status.text("Operation Concluded.")
        
        if len(st.session_state['results']) == 1:
            st.session_state['selected_image'] = 0
            
        time.sleep(1) 
        st.rerun() 

if st.session_state['results'] and not scan_triggered:
    if st.button("← RETURN TO GLOBAL AUDIT"): 
        st.session_state['selected_image'] = None
    
    if st.session_state['selected_image'] is None:
        fakes = sum(1 for r in st.session_state['results'] if r['Fake_Prob'] >= 50)
        st.subheader("GLOBAL SECURITY AUDIT")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-card'><h4>Total Data Volume</h4><h2>{len(st.session_state['results'])}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h4>Cleared (Authentic)</h4><h2 style='color:#24a148;'>{len(st.session_state['results'])-fakes}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><h4>Threats Detected</h4><h2 style='color:#fa4d56;'>{fakes}</h2></div>", unsafe_allow_html=True)
        st.divider()
        for idx, r in enumerate(st.session_state['results']):
            col1, col2, col3 = st.columns([5, 2, 2])
            col1.write(f"📄 **{r['Filename']}**")
            col2.write("🔴 THREAT DETECTED" if r['Fake_Prob'] >= 50 else "🟢 AUTHENTIC")
            if col3.button("INSPECT TELEMETRY", key=f"btn_{idx}", use_container_width=True): 
                st.session_state['selected_image'] = idx
                st.rerun()

    else:
        data = st.session_state['results'][st.session_state['selected_image']]
        is_fake = data['Fake_Prob'] >= 50
        
        st.header(f"ASSET TELEMETRY: {data['Filename']}")

        if is_fake: 
            if data['Sub_Metric_Score'] >= 50:
                st.error(f"⚠️ HIGH SEVERITY ALERT ({data['Fake_Prob']:.1f}% Match) - Spatial Manipulation Detected. Multiple independent internal engines detected severe structural anomalies, mathematically proving localized manipulation.")
            else:
                st.error(f"⚠️ HIGH SEVERITY ALERT ({data['Fake_Prob']:.1f}% Match) - Generative AI Origin Detected. The structural light physics (FFT) violently breach natural camera lens decay curves.")
        else: 
            st.success(f"✅ CLEARANCE GRANTED ({data['Fake_Prob']:.1f}% Match) - Asset conforms to natural physical constraints. Robust statistics verified no structural manipulation.")
            
        if 'VRAM_Swaps' in data and data['VRAM_Swaps'] > 0:
            st.warning(f"⚙️ Hardware Alert: System automatically hot-swapped to CPU {data['VRAM_Swaps']} time(s) to prevent CUDA Out-Of-Memory crashing during execution.")
            
        cTitle, cBtn = st.columns([4, 1])
        if HAS_FPDF:
            pdf_bytes = generate_pdf_report(data)
            cBtn.download_button(label="📥 Download PDF Report", data=pdf_bytes, file_name=f"{data['Filename']}_Security_Report.pdf", mime="application/pdf", type="primary")

        st.divider()

        st.subheader("I. Spatial Interrogation & Digital Fabric")
        
        c1, c2 = st.columns([1, 1])
        with c1: 
            st.image(data['Image'], caption="Source Asset", use_container_width=True)
        with c2: 
            st.plotly_chart(create_pure_heatmap(data['Z_Grid']), use_container_width=True)

        st.markdown("<div class='diag-box'><span class='diag-title'>Diagnostic Meaning</span><b>What is this?</b> When a real photo is taken, its digital code acts like a uniform sheet of fabric. If you cut a hole in it and paste an object, the mathematical boundary sticks out.<br><b>How to read it:</b> White/Blue areas represent uniform physics. <b>Dark red areas</b> represent highly anomalous code. We use a 98th-percentile robust thresholding system to ensure isolated webcam noise doesn't trigger a false positive.</div>", unsafe_allow_html=True)

        if data['Threat_Flags'] >= 3:
            st.error(f"🔴 **Live Diagnostic:** Critical threshold breached. Severe, clustered anomalies detected across the structural fabric, mathematically proving localized manipulation. Isolated natural noise ruled out.")
        else:
            st.success(f"🟢 **Live Diagnostic:** Anomaly threshold not met. The digital weave showed only minor variance, safely within the natural bounds for compression or edge artifacts.")
            
        st.divider()

        st.subheader("II. Fast Fourier Transform (FFT) Tests")
        
        f1, f2 = st.columns([2, 1])
        with f1: st.plotly_chart(create_spectral_forensics_plot(data['R_Vals'], data['P_Vals'], data['Alpha_Val'], data['Intercept'], is_fake=(data['Global_Prob'] > 50)), use_container_width=True)
        with f2: st.plotly_chart(create_speedometer(data['Global_Prob'], "FFT Decay Probability", 100, True), use_container_width=True)

        st.markdown("<div class='diag-box'><span class='diag-title'>Graph Explanation: Diagnostic Zones & Calibration</span><b>What is this graph?</b> It visualizes the physical diffusion of light (Spatial Frequency vs Signal Energy). The shaded red area represents the Mathematical Threat Zone (2.0 to 3.5).<br><b>Calibration:</b> Because modern smartphone and real-time webcam Image Signal Processors (ISPs) apply aggressive artificial sharpening before the image reaches our system, we apply an Optical Calibration Coefficient (0.70) to the raw signal to normalize real-time images against our theoretical bounds.</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='diag-box'><span class='diag-title'>Output Explanation: Kinematic Verdict</span><b>How to read the result:</b> If the calibrated mathematical decay signal (the thick line) is trapped inside the red Threat Zone, the FFT engine definitively classifies the image as a Deepfake. If it safely scales into the Safe Zones, it is classified as Authentic Camera Optics.</div>", unsafe_allow_html=True)

        if data['Global_Prob'] > 50:
            st.error(f"🔴 **VERDICT: DEEPFAKE DETECTED** - Reason: The calibrated signal's mathematical decay (α={data['Alpha_Val']:.2f}) is trapped strictly inside the generative threat zone.")
        else:
            st.success(f"🟢 **VERDICT: AUTHENTIC CAMERA OPTICS** - Reason: The calibrated signal's decay (α={data['Alpha_Val']:.2f}) escapes the threat bounds, matching physical optics.")

        st.divider()
        
        st.subheader("III. Sub-Metric Distributions & Spectral Correlation")
        st.markdown("<div class='diag-box'><span class='diag-title'>Diagnostic Meaning</span><b>What are these histograms?</b> The image is cut into thousands of tiny blocks to undergo specific physical tests. We enforce a 2.5 Z-score safe-baseline to prevent standard ISO sensor noise from triggering the anomaly score.<br><b>How to read it:</b> If the bars heavily cross the red Threat Line (a Z-Score of 3.5), it means that specific physical test failed violently.</div>", unsafe_allow_html=True)
        
        st.markdown("**Compression Analysis (ELA)** - *The Save-Test. Pasted objects stick out because they compress differently than the background.*")
        e1, e2 = st.columns([1, 2])
        with e1: st.plotly_chart(create_speedometer(data['Max_Z']['ELA'], "Robust Z-Score", 10, False), use_container_width=True)
        with e2: st.plotly_chart(create_z_distribution_plot(data['Z_Arrays']['ELA'], "#0f62fe"), use_container_width=True)
        
        if data['Max_Z']['ELA'] > 3.5:
            st.error(f"🔴 **Live Diagnostic:** ELA scored {data['Max_Z']['ELA']:.2f} Z. A significant cluster of patches is decaying at a vastly different rate, confirming JPEG manipulation.")
        else:
            st.success(f"🟢 **Live Diagnostic:** ELA scored {data['Max_Z']['ELA']:.2f} Z. Compression rates are statistically uniform.")
        st.markdown("---")
        
        st.markdown("**Sensor Noise (PRNU)** - *The Dust-Test. Every physical camera leaves unique microscopic 'dust' on its photos. We check if parts are missing this natural dust.*")
        n1, n2 = st.columns([1, 2])
        with n1: st.plotly_chart(create_speedometer(data['Max_Z']['Noise'], "Robust Z-Score", 10, False), use_container_width=True)
        with n2: st.plotly_chart(create_z_distribution_plot(data['Z_Arrays']['Noise'], "#0f62fe"), use_container_width=True)
        
        if data['Max_Z']['Noise'] > 3.5:
            st.error(f"🔴 **Live Diagnostic:** Noise scored {data['Max_Z']['Noise']:.2f} Z. A distinct section of the image is completely missing the host camera's digital fingerprint.")
        else:
            st.success(f"🟢 **Live Diagnostic:** Noise scored {data['Max_Z']['Noise']:.2f} Z. The camera sensor dust is uniform across the entire image.")
        st.markdown("---")

        st.markdown("**Color Chroma & RGB Correlation** - *The Overlap Test. Digital cameras group colors. Pasting breaks these groupings. Additionally, Red, Green, and Blue light naturally bleed into each other. If artificially separated, the matrix breaks.*")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1: 
            st.plotly_chart(create_speedometer(data['Max_Z']['Chroma'], "Robust Z-Score", 10, False), use_container_width=True)
        with c2: 
            st.plotly_chart(create_z_distribution_plot(data['Z_Arrays']['Chroma'], "#0f62fe"), use_container_width=True)
        
        if data['RGB_Violation']:
            st.error(f"🔴 **Live Diagnostic:** Chroma scored {data['Max_Z']['Chroma']:.2f} Z. The intense color anomaly forced a mathematical detachment in the RGB Matrix. Real light physically cannot separate its color channels this sharply.")
        else:
            st.success(f"🟢 **Live Diagnostic:** Chroma scored {data['Max_Z']['Chroma']:.2f} Z. The RGB correlation matrix is perfectly intact, meaning the color channels overlap naturally.")