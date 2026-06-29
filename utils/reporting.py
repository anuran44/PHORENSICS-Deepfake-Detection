import os
import cv2
import tempfile
import datetime
import numpy as np
import matplotlib.pyplot as plt

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

def generate_pdf_report(data):
    if not HAS_FPDF: return None
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 10, f"DeepScan Telemetry Report: {data['Filename']}", ln=True, align='C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "1. Executive Summary & Security Clearance", ln=True)
    pdf.set_font('Arial', '', 12)
    status = "THREAT DETECTED (Artificial or Manipulated)" if data['Fake_Prob'] >= 50 else "SECURE (Natural Physics Verified)"
    pdf.cell(0, 10, f"System Status: {status}", ln=True)
    pdf.cell(0, 10, f"Aggregated Threat Probability: {data['Fake_Prob']:.1f}%", ln=True)
    pdf.ln(5)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "base.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(data['Image'], cv2.COLOR_RGB2BGR))
        pdf.image(img_path, x=10, w=90)
        
        hm_path = os.path.join(tmpdir, "heatmap.jpg")
        plt.figure(figsize=(4,3))
        plt.imshow(data['Z_Grid'], cmap='Reds', vmin=0, vmax=4.5)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(hm_path)
        plt.close()
        pdf.image(hm_path, x=110, y=pdf.get_y()-65, w=90)
        pdf.ln(70)
        
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "2. Fast Fourier Transform (FFT) Tests", ln=True)
        pdf.set_font('Arial', '', 12)
        fft_path = os.path.join(tmpdir, "fft.png")
        
        plt.figure(figsize=(8,4))
        limit = min(200, len(data['R_Vals']))
        x_freq = data['R_Vals'][1:limit]
        y_actual_log = np.log10(data['P_Vals'][1:limit] + 1e-10)
        y_upper = -2.0 * np.log10(x_freq) + data['Intercept']
        y_lower = -3.5 * np.log10(x_freq) + data['Intercept']
        
        plt.fill_between(x_freq, y_lower, y_upper, color='#d32f2f', alpha=0.15, label='Threat Zone')
        plt.plot(x_freq, y_upper, 'r--', alpha=0.5)
        plt.plot(x_freq, y_lower, 'r--', alpha=0.5)
        
        line_color = '#d32f2f' if data['Global_Prob'] > 50 else '#1565c0'
        plt.plot(x_freq, y_actual_log, color=line_color, linewidth=2, label=f"Calibrated Signal (α={data['Alpha_Val']:.2f})")
        plt.title(f"Power Spectrum Decay Score: {data['Global_Prob']:.1f}%")
        plt.xscale('log')
        plt.legend()
        plt.savefig(fft_path)
        plt.close()
        
        pdf.image(fft_path, x=15, w=180)
        pdf.ln(90)

        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "3. Sub-Metric Z-Score Breakdown & RGB Correlation", ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"- Compression Anomaly (ELA): {data['Max_Z']['ELA']:.2f} Z", ln=True)
        pdf.cell(0, 10, f"- Sensor Noise Anomaly (PRNU): {data['Max_Z']['Noise']:.2f} Z", ln=True)
        pdf.cell(0, 10, f"- Color Chroma Anomaly: {data['Max_Z']['Chroma']:.2f} Z", ln=True)
        rgb_status = "VIOLATION" if data['RGB_Violation'] else "Passed (Channels Intact)"
        pdf.cell(0, 10, f"- RGB Cross-Correlation Status: {rgb_status}", ln=True)

    return pdf.output(dest='S').encode('latin-1')