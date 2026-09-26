# ExplainCrop-AI / CropMind AI - Comprehensive Changelog

## 🚀 Version 2.0.0 - Advanced Agronomic Decision Support & Multi-Modal Intelligence

### 🌟 New Agronomic Microservices & AI Features
1. **Precision Fertilizer Prescription Engine (`src/fertilizer_advisor.py`)**:
   - Computes crop-specific elemental nutrient deficits for Nitrogen, Phosphorus, and Potassium against baseline benchmarks across 22 crop classes.
   - Calculates commercial fertilizer bag quantities (Urea 46% N, DAP 18:46:0, MOP 60% K2O, SSP).
   - Provides split application scheduling (Basal, Vegetative, Flowering) and soil pH remediation recommendations.
   - Rest API Endpoint: `POST /api/v1/advisory/fertilizer`.

2. **FAO-56 Irrigation Scheduler & ET0 Engine (`src/irrigation_scheduler.py`)**:
   - Hargreaves-Samani / Penman-Monteith reference evapotranspiration ($ET_0$) calculation.
   - Dynamic crop coefficient ($K_c$) phenological modeling across 4 growth stages.
   - Net irrigation requirement ($NIR$) and daily water volume ($L/\text{acre}/\text{day}$) for Drip, Sprinkler, and Flood systems.
   - Rest API Endpoint: `POST /api/v1/advisory/irrigation`.

3. **Crop Disease & Pest Risk Forecasting Matrix (`src/disease_risk.py`)**:
   - Computes microclimate Disease Severity Index ($DSI$) based on temperature, relative humidity, and rainfall.
   - Evaluates pathogen vulnerability for high-impact crop diseases (Rice Blast, Bacterial Blight, Fall Armyworm, Pink Bollworm, Sigatoka, etc.).
   - Provides Integrated Pest Management (IPM) guidance, biological control agents (*Trichoderma*, *Pseudomonas*), and chemical cures.
   - Rest API Endpoint: `POST /api/v1/advisory/disease-risk`.

4. **Agricultural Economics & Yield Profitability Calculator (`src/economics_engine.py`)**:
   - Yield estimation per acre using TreeSHAP confidence and nutrient multi-modal multipliers.
   - Government Minimum Support Price (MSP) & regional mandi spot rate integration.
   - Complete input cost breakdown (seed, machinery, labor, fertigation, plant protection) and Return on Investment (ROI %) estimation.
   - Rest API Endpoint: `POST /api/v1/advisory/economics`.

5. **Soil Health Index (SHI) & Carbon Footprint Scorecard (`src/soil_health.py`)**:
   - Multidimensional Soil Health Index (0-100) assessing pH balance, NPK stoichiometry, organic carbon proxy, and climate resilience.
   - Nitrogen leaching and runoff risk quantification.
   - IPCC Tier 1 greenhouse gas ($CO_2e$) emissions from fertilizer application, machinery diesel, and irrigation pumping.
   - Rest API Endpoint: `POST /api/v1/advisory/soil-health`.

6. **Expanded Multi-Language Matrix (`src/translations.py`)**:
   - Full dictionary localization across **English, Hindi, Tamil, Telugu, and Spanish**.
   - Dynamic UI localization across all 14 Streamlit tabs, headers, inputs, and advisory panels.

7. **Automated PDF Advisory Generator (`src/pdf_generator.py`)**:
   - Enterprise-grade advisory report generation with character sanitization and clean layouts.
   - Rest API Endpoint: `POST /api/v1/reports/pdf`.

---

### 🛠️ Bug Fixes & Compatibility Enhancements
1. **Dual-Authentication Bug Fix**:
   - Fixed `OAuth2PasswordBearer` auto-error rejection preventing valid `X-API-Key` requests from passing when Bearer tokens are absent.
2. **Pydantic V2 Migration**:
   - Modernized schema declarations using `json_schema_extra` to eliminate deprecation warnings.
3. **Flexible Schema Ingestion**:
   - Added automatic model validator supporting both flat JSON (`nitrogen`, `phosphorus`, `potassium`, `ph`) and nested `soil_profile` objects for developer API calls.
4. **FPDF2 Byte Stream Return**:
   - Updated PDF output stream handling to ensure cross-version compatibility between `fpdf2` and `pyfpdf`.

---

### 🧪 Unit & Integration Test Suite (`pytest`)
- `tests/test_api.py`: Full REST API integration tests including dual-auth, advisory endpoints, metadata, and PDF output.
- `tests/test_db.py`: SQLite schema and table creation integrity tests.
- `tests/test_fertilizer.py`: NPK deficit, commercial bag counts, and FAO-56 ET0 calculations.
- `tests/test_disease_risk.py`: Temperature-humidity pathogen severity scoring.
- `tests/test_economics.py`: Financial yield, cost distribution, and ROI modeling.
- `tests/test_soil_health.py`: Soil health index, nitrogen leaching, and GHG emissions.
- `tests/test_translations.py`: Multi-language dictionary verification across all 5 languages.
- `tests/test_pdf.py`: PDF byte generation and formatting validation.
- `tests/test_security.py`: Password hashing and JWT encoding/decoding.
