"""
Unit tests for CropMind AI PDF Advisory Generator.
"""

import pytest
from src.pdf_generator import generate_crop_report, sanitize_text


def test_sanitize_text():
    raw_text = "Temperature: 28°C ≥ 25°C • Yield: 2.5 t/ha"
    cleaned = sanitize_text(raw_text)
    assert "deg" in cleaned
    assert ">=" in cleaned
    assert "*" in cleaned


def test_generate_crop_report_bytes():
    pdf_bytes = generate_crop_report(
        crop_name="Rice",
        viability=0.96,
        n=90.0,
        p=42.0,
        k=43.0,
        temp=26.5,
        hum=75.0,
        ph=6.5,
        rain=110.0,
        summary="Optimal hydrothermal and edaphic conditions for high paddy yield.",
        location_name="Coimbatore Experimental Farm",
        positive_factors=["High seasonal rainfall (+110 mm)", "Optimal soil pH (6.5)"],
        negative_factors=["Slight nitrogen deficiency (-10 kg/ha)"],
        fertilizer_advisory="Apply 2 bags of Urea and 1 bag of DAP in 3 split doses."
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
