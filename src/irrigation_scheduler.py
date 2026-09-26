"""
CropMind AI: Precision Irrigation Scheduling & Water Balance Engine
Implements FAO-56 Evapotranspiration (ET0 / ETc) modeling and daily irrigation scheduling.
"""

from typing import Dict, Any, List, Optional
import math

# FAO Crop Coefficients (Kc) across phenological growth stages
CROP_KC_VALUES = {
    "Rice": {"initial": 1.05, "development": 1.15, "mid": 1.20, "late": 0.90, "total_days": 120},
    "Maize": {"initial": 0.40, "development": 0.80, "mid": 1.15, "late": 0.70, "total_days": 110},
    "Chickpea": {"initial": 0.40, "development": 0.75, "mid": 1.00, "late": 0.35, "total_days": 100},
    "Kidneybeans": {"initial": 0.40, "development": 0.70, "mid": 1.05, "late": 0.65, "total_days": 90},
    "Pigeonpeas": {"initial": 0.35, "development": 0.70, "mid": 1.00, "late": 0.45, "total_days": 150},
    "Mothbeans": {"initial": 0.30, "development": 0.65, "mid": 0.95, "late": 0.40, "total_days": 80},
    "Mungbean": {"initial": 0.35, "development": 0.70, "mid": 1.05, "late": 0.45, "total_days": 75},
    "Blackgram": {"initial": 0.35, "development": 0.70, "mid": 1.05, "late": 0.45, "total_days": 80},
    "Lentil": {"initial": 0.40, "development": 0.70, "mid": 1.00, "late": 0.35, "total_days": 100},
    "Pomegranate": {"initial": 0.50, "development": 0.65, "mid": 0.75, "late": 0.60, "total_days": 365},
    "Banana": {"initial": 0.70, "development": 1.00, "mid": 1.20, "late": 1.10, "total_days": 300},
    "Mango": {"initial": 0.60, "development": 0.75, "mid": 0.85, "late": 0.70, "total_days": 365},
    "Grapes": {"initial": 0.30, "development": 0.60, "mid": 0.85, "late": 0.45, "total_days": 210},
    "Watermelon": {"initial": 0.40, "development": 0.75, "mid": 1.00, "late": 0.75, "total_days": 85},
    "Muskmelon": {"initial": 0.40, "development": 0.70, "mid": 0.95, "late": 0.70, "total_days": 80},
    "Apple": {"initial": 0.50, "development": 0.75, "mid": 0.95, "late": 0.75, "total_days": 240},
    "Orange": {"initial": 0.70, "development": 0.70, "mid": 0.70, "late": 0.70, "total_days": 365},
    "Papaya": {"initial": 0.65, "development": 0.90, "mid": 1.10, "late": 0.90, "total_days": 300},
    "Coconut": {"initial": 0.80, "development": 0.80, "mid": 0.80, "late": 0.80, "total_days": 365},
    "Cotton": {"initial": 0.45, "development": 0.80, "mid": 1.15, "late": 0.65, "total_days": 160},
    "Jute": {"initial": 0.50, "development": 0.90, "mid": 1.15, "late": 0.80, "total_days": 120},
    "Coffee": {"initial": 0.90, "development": 0.95, "mid": 1.05, "late": 0.95, "total_days": 365},
}

DEFAULT_KC = {"initial": 0.45, "development": 0.75, "mid": 1.05, "late": 0.65, "total_days": 110}


def estimate_et0(temperature_c: float, humidity_pct: float, latitude: float = 13.0) -> float:
    """
    Estimates Reference Evapotranspiration (ET0 in mm/day) using radiation-temperature approximation.
    """
    # Extraterrestrial radiation approximation Ra (MJ/m2/day)
    lat_rad = math.radians(latitude)
    ra_approx = 35.0 - 0.2 * abs(latitude)
    
    # Vapor pressure deficit factor
    es = 0.6108 * math.exp((17.27 * temperature_c) / (temperature_c + 237.3))
    ea = es * (humidity_pct / 100.0)
    vpd = max(0.1, es - ea)
    
    # Hargreaves-Samani / Turc empirical ET0 formulation
    temp_factor = max(5.0, temperature_c)
    et0 = 0.0135 * (temp_factor / (temp_factor + 15.0)) * (ra_approx * 23.8846 + 50.0) * (1.0 + (50.0 - min(95.0, humidity_pct)) / 100.0)
    return max(1.5, min(9.5, round(et0, 2)))


def calculate_irrigation_schedule(
    crop_name: str,
    temperature_c: float,
    humidity_pct: float,
    rainfall_14d_mm: float,
    growth_stage: str = "mid",
    field_area_acres: float = 1.0,
    soil_type: str = "Loamy",
    irrigation_method: str = "Drip Irrigation"
) -> Dict[str, Any]:
    """
    Computes daily water demand, effective rainfall contribution, and actionable irrigation schedule.
    """
    crop_key = crop_name.strip().capitalize()
    kc_info = CROP_KC_VALUES.get(crop_key, DEFAULT_KC)
    kc = kc_info.get(growth_stage.lower(), kc_info["mid"])
    
    et0_mm_day = estimate_et0(temperature_c, humidity_pct)
    etc_mm_day = round(et0_mm_day * kc, 2)
    
    # Effective rainfall (USDA SCS method approximation)
    daily_precip = max(0.0, rainfall_14d_mm / 14.0)
    effective_rain_daily = max(0.0, (daily_precip * 0.8) - 1.0) if daily_precip > 2.0 else 0.0
    
    # Net Irrigation Requirement (NIR in mm/day)
    nir_mm_day = max(0.0, etc_mm_day - effective_rain_daily)
    
    # Volume in Litres/Acre/Day (1 mm over 1 acre = 4,046.86 Litres)
    litres_per_acre_daily = round(nir_mm_day * 4046.86, 0)
    total_litres_daily = round(litres_per_acre_daily * field_area_acres, 0)
    
    # Irrigation efficiency based on method
    eff_map = {
        "Drip Irrigation": 0.90,
        "Sprinkler": 0.75,
        "Flood / Furrow": 0.55
    }
    efficiency = eff_map.get(irrigation_method, 0.85)
    gross_litres_daily = round(total_litres_daily / efficiency, 0)
    
    # Soil available water capacity (AWC in mm/meter)
    awc_map = {"Sandy": 70, "Sandy Loam": 110, "Loamy": 150, "Clay Loam": 180, "Heavy Clay": 200}
    awc = awc_map.get(soil_type, 150)
    
    # Recommended frequency & runtime
    if nir_mm_day <= 0.5:
        watering_status = "No Irrigation Required"
        runtime_hours = 0.0
        interval_days = 0
        advisory = "Recent and forecasted precipitation covers crop evapotranspiration demand. Ensure proper field drainage to prevent waterlogging."
    elif irrigation_method == "Drip Irrigation":
        watering_status = "Daily / Alternate Day Fertigation"
        # Assuming typical drip emitter flow of 2000 Litres/hour/acre
        runtime_hours = round(gross_litres_daily / (2000.0 * field_area_acres), 1)
        interval_days = 1
        advisory = f"Operate drip system for ~{runtime_hours} hours every {interval_days} day(s) during early morning (6:00 AM - 9:00 AM) to minimize evaporation."
    elif irrigation_method == "Sprinkler":
        watering_status = "Cycle Every 3-4 Days"
        runtime_hours = round((gross_litres_daily * 3) / (4000.0 * field_area_acres), 1)
        interval_days = 3
        advisory = f"Run sprinkler system for ~{runtime_hours} hours every {interval_days} days during calm wind hours."
    else: # Flood
        watering_status = "Flood Every 7-10 Days"
        runtime_hours = round((nir_mm_day * 7) / 10.0, 1) # depth in cm
        interval_days = 7
        advisory = f"Apply ~5 cm depth of irrigation water every {interval_days} days."

    return {
        "crop": crop_key,
        "growth_stage": growth_stage,
        "crop_kc": kc,
        "reference_et0_mm_day": et0_mm_day,
        "crop_etc_mm_day": etc_mm_day,
        "effective_rainfall_daily_mm": round(effective_rain_daily, 2),
        "net_irrigation_req_mm_day": round(nir_mm_day, 2),
        "water_volume_litres_acre_day": litres_per_acre_daily,
        "total_water_volume_litres_day": total_litres_daily,
        "gross_water_req_litres_day": gross_litres_daily,
        "irrigation_method": irrigation_method,
        "system_efficiency_pct": int(efficiency * 100),
        "recommended_interval_days": interval_days,
        "recommended_runtime_hours": runtime_hours,
        "watering_status": watering_status,
        "actionable_advisory": advisory
    }
