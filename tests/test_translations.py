"""
Unit tests for CropMind AI Multi-Language Translation & Localization Matrix.
"""

import pytest
from src.translations import TRANSLATIONS, get_translation


def test_translation_languages_present():
    expected_langs = ["English", "Hindi", "Tamil", "Telugu", "Spanish"]
    for lang in expected_langs:
        assert lang in TRANSLATIONS, f"Language {lang} missing from TRANSLATIONS"


def test_core_keys_present_in_all_languages():
    required_keys = [
        "title", "tagline", "login", "register", "username", "password",
        "logout", "recommendations", "simulation", "fertilizer_advisor",
        "disease_risk", "economics", "soil_health", "generate_report_pdf"
    ]
    for lang, table in TRANSLATIONS.items():
        for key in required_keys:
            assert key in table, f"Key '{key}' missing from {lang} translation matrix"
            assert len(table[key].strip()) > 0, f"Key '{key}' has empty translation in {lang}"


def test_get_translation_fallback():
    # Valid key
    val = get_translation("Tamil", "login")
    assert val == "உள்நுழைக"
    
    # Missing key fallback to English
    fallback_val = get_translation("Spanish", "non_existent_key_xyz", default="Default Val")
    assert fallback_val == "Default Val"
