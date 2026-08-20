import os
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import io
import datetime
import tempfile
import numpy as np
from PIL import Image
from fpdf import FPDF
from src.config import CLASS_NAMES

# =====================================
# CLINICAL KNOWLEDGE BASE & DEFINITIONS
# =====================================

CLINICAL_DEFINITIONS = {
    "C1": {
        "title": "Category 1 (C1) - Normal Mandibular Cortex",
        "severity": "Low Risk / Healthy",
        "color_rgb": (16, 185, 129),  # Emerald Green
        "summary": "The mandibular inferior cortex is normal with sharp, continuous inner margins on both sides.",
        "findings": [
            "Mandibular cortical margin is smooth, dense, and fully continuous.",
            "No endosteal resorptive defects or semilunar scalloping observed.",
            "Trabecular bone architecture exhibits normal density and structural alignment.",
            "Low overall risk for systemic osteoporosis based on mandibular cortical index (MCI)."
        ],
        "recommendations": [
            "Routine oral health evaluation and preventive care.",
            "Maintain standard dietary calcium and Vitamin D supplementation.",
            "Re-evaluate radiographically during standard dental recall (2-3 years)."
        ]
    },
    "C2": {
        "title": "Category 2 (C2) - Mild to Moderate Osteopenia",
        "severity": "Moderate Risk / Scalloped",
        "color_rgb": (245, 158, 11),  # Amber Yellow
        "summary": "The mandibular inferior cortex shows mild thinning or semilunar defects with clear inner cortical endosteal resorption.",
        "findings": [
            "Inner cortical margin demonstrates mild scalloping and endosteal porosity.",
            "Semilunar resorptive defects detected along the inferior mandibular border.",
            "Early reduction in trabecular bone mass observed near the mandibular angle.",
            "Mandibular cortical index indicates borderline or mild systemic osteopenia."
        ],
        "recommendations": [
            "Clinical correlation with patient metabolic history and FRAX risk score.",
            "Recommend baseline Dual-Energy X-ray Absorptiometry (DEXA) T-score assessment.",
            "Evaluate serum 25-hydroxyvitamin D and calcium levels.",
            "Schedule follow-up panoramic radiograph in 12 months."
        ]
    },
    "C3": {
        "title": "Category 3 (C3) - Severe Osteoporosis",
        "severity": "High Risk / Porous & Fragmented",
        "color_rgb": (239, 68, 68),  # Rose Red
        "summary": "Severe cortical thinning with extensive endosteal resorption, high porosity, and discontinuous cortical residue.",
        "findings": [
            "Mandibular inferior cortical margin is severely porous and discontinuous.",
            "Extensive endosteal erosion with cortical residue fragments.",
            "Marked reduction in trabecular radiodensity along the mandibular ramus and body.",
            "Strong radiographic indicator of systemic skeletal osteoporosis and elevated fracture risk."
        ],
        "recommendations": [
            "Urgent referral to an endocrinologist or rheumatologist for formal DEXA evaluation.",
            "Consider medical therapy (e.g., antiresorptive or bone-forming agents).",
            "Exercise caution prior to invasive oral surgeries (e.g., tooth extractions, implants).",
            "Implement fall prevention protocols and comprehensive metabolic workup."
        ]
    }
}


class MedicalReportPDF(FPDF):
    def header(self):
        # Header Banner background
        self.set_fill_color(15, 118, 110)  # Teal header
        self.rect(0, 0, 210, 24, 'F')
        
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 6)
        self.cell(0, 8, 'OSTEOPOROSIS AI DIAGNOSTIC REPORT', 0, 1, 'L')
        
        self.set_font('Helvetica', 'I', 9)
        self.set_xy(10, 14)
        self.cell(0, 5, 'Mandibular Panorex Radiograph Analysis & Explainable AI (XAI) Saliency', 0, 1, 'L')
        self.ln(8)

    def footer(self):
        self.set_y(-18)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, 'CONFIDENTIAL MEDICAL DECISION SUPPORT REPORT | NOT A STANDALONE DIAGNOSIS', 0, 1, 'C')
        self.cell(0, 5, f'Page {self.page_no()} of {{nb}}', 0, 0, 'C')


def generate_pdf_report(
    original_image,
    gradcam_image,
    predicted_class,
    confidence,
    probabilities,
    model_predictions,
    best_model_name,
    patient_info=None
):
    """
    Generates a standardized, professional PDF report as bytes.
    """
    if patient_info is None:
        patient_info = {
            "patient_id": "PT-89421",
            "age": "58",
            "gender": "Female",
            "scan_type": "Mandibular Panoramic Radiograph (OPG)",
            "physician": "Dr. A. Sharma, MD / DDS"
        }
        
    info = CLINICAL_DEFINITIONS.get(predicted_class, CLINICAL_DEFINITIONS["C1"])
    r, g, b = info["color_rgb"]
    
    pdf = MedicalReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # ----------------------------------------------------
    # SECTION 1: REPORT METADATA & PATIENT INFORMATION
    # ----------------------------------------------------
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(15, 118, 110)
    pdf.cell(0, 6, 'PATIENT & EXAMINATION METADATA', 0, 1, 'L')
    pdf.set_draw_color(15, 118, 110)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(30, 41, 59)
    
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    col_w = 47
    pdf.set_fill_color(241, 245, 249)
    pdf.rect(10, pdf.get_y(), 190, 20, 'F')
    
    y_start = pdf.get_y() + 2
    pdf.set_xy(12, y_start)
    pdf.cell(col_w, 5, f"Patient ID: {patient_info.get('patient_id', 'N/A')}", 0, 0)
    pdf.cell(col_w, 5, f"Age: {patient_info.get('age', 'N/A')} yrs", 0, 0)
    pdf.cell(col_w, 5, f"Gender: {patient_info.get('gender', 'N/A')}", 0, 0)
    pdf.cell(col_w, 5, f"Date: {now_str[:10]}", 0, 1)
    
    pdf.set_xy(12, y_start + 8)
    pdf.cell(col_w * 2, 5, f"Referring Clinician: {patient_info.get('physician', 'N/A')}", 0, 0)
    pdf.cell(col_w * 2, 5, f"Modality: {patient_info.get('scan_type', 'OPG Panorex')}", 0, 1)
    
    pdf.set_y(y_start + 22)
    
    # ----------------------------------------------------
    # SECTION 2: PRIMARY DIAGNOSTIC IMPRESSION
    # ----------------------------------------------------
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(15, 118, 110)
    pdf.cell(0, 6, 'PRIMARY DIAGNOSTIC IMPRESSION', 0, 1, 'L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    # Badge Box for Diagnostic Class
    pdf.set_fill_color(r, g, b)
    pdf.rect(10, pdf.get_y(), 190, 14, 'F')
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(14, pdf.get_y() + 4)
    pdf.cell(0, 6, f"DIAGNOSIS: {info['title']} ({confidence*100:.1f}% Consensus Confidence)", 0, 1, 'L')
    
    pdf.ln(6)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5, f"Clinical Summary: {info['summary']}")
    pdf.ln(3)
    
    # ----------------------------------------------------
    # SECTION 3: EXPLAINABLE AI (XAI) SALIENCY MAPS
    # ----------------------------------------------------
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(15, 118, 110)
    pdf.cell(0, 6, 'EXPLAINABLE AI (XAI) GRAD-CAM SALIENCY ANALYSIS', 0, 1, 'L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(
        0, 4, 
        "Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the visual cortical zones "
        "prioritized by the deep learning neural network during inference. Warm red/yellow zones represent "
        "maximum model focus concentrated along the inferior cortical margin."
    )
    pdf.ln(2)
    
    # Save images to temp files to embed into PDF
    temp_dir = tempfile.mkdtemp()
    orig_path = os.path.join(temp_dir, "orig.jpg")
    cam_path = os.path.join(temp_dir, "cam.jpg")
    
    if isinstance(original_image, Image.Image):
        original_image.save(orig_path, "JPEG")
    else:
        Image.fromarray(original_image).save(orig_path, "JPEG")
        
    if gradcam_image is not None:
        if isinstance(gradcam_image, Image.Image):
            gradcam_image.save(cam_path, "JPEG")
        else:
            Image.fromarray(gradcam_image).save(cam_path, "JPEG")
    else:
        # Fallback if gradcam unavailable
        original_image.save(cam_path, "JPEG")
        
    img_y = pdf.get_y()
    pdf.image(orig_path, x=15, y=img_y, w=85, h=45)
    pdf.image(cam_path, x=110, y=img_y, w=85, h=45)
    
    pdf.set_y(img_y + 46)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(90, 4, 'Figure 1: Original Panorex Radiograph', 0, 0, 'C')
    pdf.cell(90, 4, 'Figure 2: SmoothGradCAM++ Cortical Saliency Heatmap', 0, 1, 'C')
    pdf.ln(4)
    
    # ----------------------------------------------------
    # SECTION 4: MORPHOMETRIC FINDINGS & RECOMMENDATIONS
    # ----------------------------------------------------
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(15, 118, 110)
    pdf.cell(0, 6, 'MORPHOMETRIC FINDINGS & CLINICAL RECOMMENDATIONS', 0, 1, 'L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 5, 'Radiographic Morphometric Observations:', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    for f in info["findings"]:
        pdf.cell(5, 4, '-', 0, 0)
        pdf.cell(0, 4, f, 0, 1)
    pdf.ln(2)
    
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(0, 5, 'Suggested Clinical Action Plan:', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    for r_item in info["recommendations"]:
        pdf.cell(5, 4, '>', 0, 0)
        pdf.cell(0, 4, r_item, 0, 1)
    pdf.ln(4)
    
    # ----------------------------------------------------
    # SECTION 5: MODEL ENSEMBLE CONSENSUS & PERFORMANCE
    # ----------------------------------------------------
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(15, 118, 110)
    pdf.cell(0, 6, 'MULTI-MODEL ARCHITECTURE CONSENSUS', 0, 1, 'L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    # Table Header
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(226, 232, 240)
    pdf.cell(50, 5, 'Model Architecture', 1, 0, 'L', fill=True)
    pdf.cell(45, 5, 'Predicted Category', 1, 0, 'C', fill=True)
    pdf.cell(45, 5, 'Confidence Level', 1, 0, 'C', fill=True)
    pdf.cell(50, 5, 'Status', 1, 1, 'C', fill=True)
    
    pdf.set_font('Helvetica', '', 8)
    for name in ["ResNet18", "ResNet50", "EfficientNetB0"]:
        p_data = model_predictions.get(name, {"class": "N/A", "confidence": 0.0})
        p_cls = p_data["class"]
        p_cnf = p_data["confidence"] * 100
        is_best = "[Primary]" if name == best_model_name else "[Active]"
        
        pdf.cell(50, 5, name, 1, 0, 'L')
        pdf.cell(45, 5, p_cls, 1, 0, 'C')
        pdf.cell(45, 5, f"{p_cnf:.2f}%", 1, 0, 'C')
        pdf.cell(50, 5, is_best, 1, 1, 'C')
        
    pdf.ln(4)
    
    # ----------------------------------------------------
    # SECTION 6: FORMAL CLINICAL & REGULATORY DISCLAIMER
    # ----------------------------------------------------
    pdf.set_fill_color(254, 242, 242)
    pdf.set_draw_color(239, 68, 68)
    pdf.rect(10, pdf.get_y(), 190, 18, 'DF')
    
    pdf.set_xy(12, pdf.get_y() + 2)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(185, 28, 28)
    pdf.cell(0, 4, 'CLINICAL AI DECISION SUPPORT DISCLAIMER:', 0, 1, 'L')
    
    pdf.set_x(12)
    pdf.set_font('Helvetica', '', 7.5)
    pdf.set_text_color(127, 29, 29)
    pdf.multi_cell(
        186, 3.5,
        "This diagnostic report was generated automatically by an artificial intelligence software system. "
        "It is strictly intended as a clinical decision-support tool to assist healthcare professionals "
        "and MUST NOT replace professional clinical judgment, DEXA scans, or radiologic evaluation by a licensed physician."
    )
    
    # Return raw PDF byte stream
    pdf_bytes = pdf.output(dest='S')
    
    # Cleanup temp files
    try:
        os.remove(orig_path)
        os.remove(cam_path)
        os.rmdir(temp_dir)
    except Exception:
        pass
        
    return bytes(pdf_bytes)
