from fpdf import FPDF
import datetime

class CropReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.cell(0, 10, "CropMind AI - Advisory Report", 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

def generate_crop_report(crop_name: str, viability: float, n: float, p: float, k: float, temp: float, hum: float, ph: float, rain: float, summary: str):
    pdf = CropReportPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, f"Recommendation: {crop_name.upper()}", 0, 1)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.cell(0, 8, f"Viability Score: {viability*100:.1f}%", 0, 1)
    pdf.ln(5)
    
    # Soil Profile
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Soil & Climate Profile:", 0, 1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Nitrogen (N): {n} mg/kg", 0, 1)
    pdf.cell(0, 8, f"Phosphorus (P): {p} mg/kg", 0, 1)
    pdf.cell(0, 8, f"Potassium (K): {k} mg/kg", 0, 1)
    pdf.cell(0, 8, f"pH Level: {ph}", 0, 1)
    pdf.cell(0, 8, f"Temperature: {temp} C", 0, 1)
    pdf.cell(0, 8, f"Humidity: {hum} %", 0, 1)
    pdf.cell(0, 8, f"Rainfall: {rain} mm", 0, 1)
    pdf.ln(5)
    
    # Explanations
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "AI Advisory Summary:", 0, 1)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 8, summary)
    
    return pdf.output(dest="S").encode("latin-1")
