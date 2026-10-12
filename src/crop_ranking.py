"""
CropMind AI: Multi-Criteria Decision Analysis (MCDA / TOPSIS) Crop Ranking Engine
Combines Machine Learning Viability, Economic Net Profit, Water Efficiency, and Pathogen Resilience
into an optimal composite ranking matrix with customizable weights.
"""

from typing import Dict, Any, List, Optional
import math
import numpy as np

from src.fertilizer_advisor import CROP_NUTRIENT_BENCHMARKS
from src.economics_engine import calculate_crop_profitability
from src.disease_risk import calculate_disease_pest_risk


def calculate_topsis_crop_ranking(
    candidate_crops_with_scores: List[Dict[str, Any]], # [{"crop": "Rice", "viability_score": 0.85}, ...]
    temperature_c: float,
    humidity_pct: float,
    rainfall_mm: float,
    weight_viability: float = 0.35,
    weight_profit: float = 0.25,
    weight_water_efficiency: float = 0.20,
    weight_resilience: float = 0.20,
    field_area_acres: float = 1.0
) -> Dict[str, Any]:
    """
    Ranks alternative candidate crops using TOPSIS vector normalization and Euclidean distance to ideal solution.
    """
    if not candidate_crops_with_scores:
        return {"ranked_crops": [], "message": "No candidate crops provided."}
        
    crops_data = []
    
    for item in candidate_crops_with_scores:
        crop_name = item["crop"].strip().capitalize()
        viab = float(item.get("viability_score", 0.70))
        
        # 1. Economic Profit Projection
        econ = calculate_crop_profitability(crop_name, viability_score=viab, field_area_acres=field_area_acres)
        profit_inr = max(1000.0, float(econ["financial_summary"]["net_profit_inr"]))
        
        # 2. Water Requirement & Efficiency
        bench = CROP_NUTRIENT_BENCHMARKS.get(crop_name, {"water_req_mm": 600})
        water_req = float(bench.get("water_req_mm", 600))
        water_efficiency_score = 1000.0 / max(100.0, water_req) # Higher is better
        
        # 3. Disease & Climate Resilience (100 - max risk score)
        disease = calculate_disease_pest_risk(crop_name, temperature_c, humidity_pct, rainfall_mm)
        resilience_score = max(10.0, 100.0 - float(disease["max_risk_score"]))
        
        crops_data.append({
            "crop": crop_name,
            "viability_score": viab,
            "net_profit_inr": profit_inr,
            "water_req_mm": water_req,
            "water_efficiency": water_efficiency_score,
            "resilience_score": resilience_score,
            "financial_roi_pct": econ["financial_summary"]["roi_pct"]
        })

    # Total weight normalization
    total_w = weight_viability + weight_profit + weight_water_efficiency + weight_resilience
    w1 = weight_viability / total_w
    w2 = weight_profit / total_w
    w3 = weight_water_efficiency / total_w
    w4 = weight_resilience / total_w
    
    matrix = np.array([
        [c["viability_score"], c["net_profit_inr"], c["water_efficiency"], c["resilience_score"]]
        for c in crops_data
    ])
    
    # Vector Normalization: r_ij = x_ij / sqrt(sum(x_kj^2))
    norm_denom = np.sqrt(np.sum(matrix ** 2, axis=0))
    norm_denom[norm_denom == 0] = 1e-6
    norm_matrix = matrix / norm_denom
    
    # Weighted Normalized Matrix
    weights = np.array([w1, w2, w3, w4])
    weighted_matrix = norm_matrix * weights
    
    # Ideal Best (A+) and Ideal Worst (A-) - all criteria are benefit criteria
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)
    
    # Euclidean Distances S+ and S-
    dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))
    
    # Relative Closeness to Ideal Solution (C*)
    denom = dist_best + dist_worst
    denom[denom == 0] = 1e-6
    closeness = dist_worst / denom
    
    # Assemble ranked outputs
    ranked_results = []
    for idx, c in enumerate(crops_data):
        score = float(closeness[idx])
        ranked_results.append({
            "crop": c["crop"],
            "topsis_score": round(score, 4),
            "topsis_score_pct": round(score * 100.0, 1),
            "ml_viability_pct": round(c["viability_score"] * 100.0, 1),
            "net_profit_inr": round(c["net_profit_inr"], 2),
            "roi_pct": round(c["financial_roi_pct"], 1),
            "water_req_mm": c["water_req_mm"],
            "climate_resilience_score": round(c["resilience_score"], 1)
        })
        
    ranked_results.sort(key=lambda x: x["topsis_score"], reverse=True)
    for rank_idx, r in enumerate(ranked_results, start=1):
        r["topsis_rank"] = rank_idx

    return {
        "ranked_crops": ranked_results,
        "optimal_crop_decision": ranked_results[0]["crop"] if ranked_results else "None",
        "evaluation_criteria_weights": {
            "ml_viability": round(w1, 2),
            "net_profit": round(w2, 2),
            "water_efficiency": round(w3, 2),
            "climate_resilience": round(w4, 2)
        },
        "mcda_methodology": "Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS)"
    }
