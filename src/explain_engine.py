"""
OpticCrop: Multi-Modal Inference & Explainability Engine
Generates Top-K crop rankings with calibrated viability scores,
computes exact local TreeSHAP factor attributions (sub-150ms),
and delivers natural-language agronomic causal explanations.
"""

import os
import sys

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

MODELS_DIR = os.path.join(BASE_DIR, "models")
EPSILON = 1e-5


class OpticCropEngine:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.explainer = None
        self.background_centroids = None
        self.metadata = {}
        self.crop_profiles = {}

        self.feature_names = [
            "nitrogen",
            "phosphorus",
            "potassium",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
            "r_np",
            "r_nk",
            "r_pk",
            "thi",
            "mai",
        ]

        self.feature_labels = {
            "nitrogen": "Soil Nitrogen (N)",
            "phosphorus": "Soil Phosphorus (P)",
            "potassium": "Soil Potassium (K)",
            "temperature": "Ambient Temperature (°C)",
            "humidity": "Relative Humidity (%)",
            "ph": "Soil pH",
            "rainfall": "Rainfall / Forecast (mm)",
            "r_np": "N:P Nutrient Ratio",
            "r_nk": "N:K Nutrient Ratio",
            "r_pk": "P:K Nutrient Ratio",
            "thi": "Temperature-Humidity Index (THI)",
            "mai": "Moisture Availability Index (MAI)",
        }

        self.mu_rain = 103.46
        self.sigma_rain = 54.96

        self.load_artifacts()

    def load_artifacts(self):
        """Loads serialized model, encoder, explainer, background, and metadata."""
        model_path = os.path.join(MODELS_DIR, "xgboost_crop_model.joblib")
        encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
        explainer_path = os.path.join(MODELS_DIR, "shap_explainer.joblib")
        bg_path = os.path.join(MODELS_DIR, "shap_background.joblib")
        meta_path = os.path.join(MODELS_DIR, "metadata.json")
        profiles_path = os.path.join(MODELS_DIR, "crop_profiles.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact missing at {model_path}. Please run 'src/model_train.py' first."
            )

        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.explainer = joblib.load(explainer_path)

        if os.path.exists(bg_path):
            self.background_centroids = joblib.load(bg_path)

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                self.mu_rain = self.metadata.get("mu_rain", 103.46)
                self.sigma_rain = self.metadata.get("sigma_rain", 54.96)

        if os.path.exists(profiles_path):
            with open(profiles_path, "r", encoding="utf-8") as f:
                self.crop_profiles = json.load(f)

    def synthesize_input_vector(
        self,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float,
    ) -> pd.DataFrame:
        """
        Synthesizes a 12-dimensional multi-modal vector X in R^12 per TRD Section 3.2:
          - R_NP = N / (P + epsilon)
          - R_NK = N / (K + epsilon)
          - R_PK = P / (K + epsilon)
          - THI = 0.8 * T + (RH / 100) * (T - 14.4) + 46.4
          - MAI = (Rainfall - mu_hist) / sigma_hist
        """
        r_np = nitrogen / (phosphorus + EPSILON)
        r_nk = nitrogen / (potassium + EPSILON)
        r_pk = phosphorus / (potassium + EPSILON)
        thi = 0.8 * temperature + (humidity / 100.0) * (temperature - 14.4) + 46.4
        mai = (rainfall - self.mu_rain) / (self.sigma_rain + EPSILON)

        row = [
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall,
            r_np,
            r_nk,
            r_pk,
            thi,
            mai,
        ]
        return pd.DataFrame([row], columns=self.feature_names)

    def predict_and_explain(
        self,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float,
        top_k: int = 3,
        include_all_crops: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs XGBoost multi-class inference, computes exact TreeSHAP attribution,
        and generates causal agronomic advice.
        """
        input_df = self.synthesize_input_vector(
            nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall
        )

        # 1. Calibrated Softmax Probabilities
        probabilities = self.model.predict_proba(input_df)[0]
        top_indices = np.argsort(probabilities)[::-1]
        selected_indices = top_indices if include_all_crops else top_indices[:top_k]

        # 2. Compute TreeSHAP values for the input
        shap_values = self.explainer.shap_values(input_df)
        expected_val = self.explainer.expected_value

        recommendations = []
        for rank, idx in enumerate(selected_indices, start=1):
            crop_name = self.encoder.inverse_transform([idx])[0]
            viability_score = float(probabilities[idx])

            # Extract SHAP vector for this class
            if isinstance(shap_values, list):
                crop_shap = shap_values[idx][0]
                base_v = float(expected_val[idx]) if isinstance(expected_val, (list, np.ndarray)) else float(expected_val)
            elif len(shap_values.shape) == 3:
                if shap_values.shape[2] == len(self.encoder.classes_):
                    crop_shap = shap_values[0, :, idx]
                else:
                    crop_shap = shap_values[0, idx, :]
                base_v = float(expected_val[idx]) if isinstance(expected_val, (list, np.ndarray)) else float(expected_val)
            else:
                crop_shap = shap_values[0]
                base_v = float(expected_val) if not isinstance(expected_val, (list, np.ndarray)) else float(expected_val[0])

            # Decompose positive and negative factor contributions
            feature_contributions = []
            pos_factors = []
            neg_factors = []

            for feat, val, s_val in zip(self.feature_names, input_df.iloc[0], crop_shap):
                delta = float(s_val)
                contrib = {
                    "feature": feat,
                    "label": self.feature_labels.get(feat, feat),
                    "value": round(float(val), 2),
                    "shap_delta": round(delta, 4),
                    "impact": "positive" if delta >= 0 else "negative",
                    "abs_impact": abs(delta),
                }
                feature_contributions.append(contrib)

                if delta > 0.01:
                    pos_factors.append({"feature": feat, "value": round(float(val), 2), "shap_delta": round(delta, 4)})
                elif delta < -0.01:
                    neg_factors.append({"feature": feat, "value": round(float(val), 2), "shap_delta": round(delta, 4)})

            # Sort factors by impact
            pos_factors.sort(key=lambda x: x["shap_delta"], reverse=True)
            neg_factors.sort(key=lambda x: x["shap_delta"])
            feature_contributions.sort(key=lambda x: x["abs_impact"], reverse=True)

            # Generate narrative summary
            summary = self._build_human_readable_summary(
                crop_name, input_df.iloc[0], pos_factors, neg_factors, viability_score
            )

            recommendations.append(
                {
                    "rank": rank,
                    "crop": crop_name,
                    "viability_score": round(viability_score, 4),
                    "confidence_percent": round(viability_score * 100, 2),
                    "class_index": int(idx),
                    "explanations": {
                        "base_value": round(base_v, 4),
                        "top_positive_factors": pos_factors[:3],
                        "top_negative_factors": neg_factors[:2],
                        "all_contributions": feature_contributions,
                        "human_readable_summary": summary,
                    },
                }
            )

        top_crop = recommendations[0]["crop"]
        advisory_tips = self._generate_agronomic_advisory(top_crop, input_df.iloc[0])

        return {
            "top_crop": top_crop,
            "top_viability_score": recommendations[0]["viability_score"],
            "recommendations": recommendations,
            "feature_contributions": recommendations[0]["explanations"]["all_contributions"],
            "human_readable_summary": recommendations[0]["explanations"]["human_readable_summary"],
            "advisory": advisory_tips,
            "raw_inputs": {
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            },
            "derived_metrics": {
                "r_np": round(float(input_df["r_np"].iloc[0]), 2),
                "r_nk": round(float(input_df["r_nk"].iloc[0]), 2),
                "r_pk": round(float(input_df["r_pk"].iloc[0]), 2),
                "thi": round(float(input_df["thi"].iloc[0]), 1),
                "mai": round(float(input_df["mai"].iloc[0]), 2),
            },
        }

    def _build_human_readable_summary(
        self,
        crop: str,
        inputs: pd.Series,
        pos_factors: List[Dict[str, Any]],
        neg_factors: List[Dict[str, Any]],
        viability: float,
    ) -> str:
        """Constructs intuitive natural-language explanation per PRD Section 2.2."""
        if viability >= 0.70:
            rating = "Highly recommended"
        elif viability >= 0.40:
            rating = "Moderately recommended"
        else:
            rating = "Viable alternative"

        pos_descriptions = []
        for p in pos_factors[:2]:
            feat = p["feature"]
            val = p["value"]
            if feat == "temperature":
                pos_descriptions.append(f"forecasted temperature ({val:.1f}°C)")
            elif feat == "rainfall":
                pos_descriptions.append(f"moisture/precipitation level ({val:.0f} mm)")
            elif feat == "nitrogen":
                pos_descriptions.append(f"favorable soil Nitrogen ({val:.0f} mg/kg)")
            elif feat == "potassium":
                pos_descriptions.append(f"soil Potassium ({val:.0f} mg/kg)")
            elif feat == "ph":
                pos_descriptions.append(f"balanced pH ({val:.1f})")
            elif feat == "thi":
                pos_descriptions.append(f"climate comfort index THI ({val:.1f})")
            elif feat == "mai":
                pos_descriptions.append(f"moisture availability index ({val:+.2f})")
            else:
                pos_descriptions.append(f"{self.feature_labels.get(feat, feat)}")

        if pos_descriptions:
            support_clause = " due to superior alignment with " + " and ".join(pos_descriptions)
        else:
            support_clause = " based on overall multi-factor soil-climate compatibility"

        neg_descriptions = []
        for n in neg_factors[:1]:
            feat = n["feature"]
            val = n["value"]
            neg_descriptions.append(f"{self.feature_labels.get(feat, feat)} ({val})")

        limitation_clause = ""
        if neg_descriptions:
            limitation_clause = f", while managing minor variation in {neg_descriptions[0]}."
        else:
            limitation_clause = "."

        return f"{rating}{support_clause}{limitation_clause}"

    def _generate_agronomic_advisory(self, crop: str, inputs: pd.Series) -> List[Dict[str, str]]:
        """Generates actionable soil and fertilizer recommendations."""
        profile = self.crop_profiles.get(crop, {})
        advisories = []

        # Nitrogen Management
        n_val = inputs["nitrogen"]
        if profile and "nitrogen" in profile:
            opt_n = profile["nitrogen"]["mean"]
            if n_val < opt_n - 25:
                advisories.append(
                    {
                        "category": "Fertilizer (Nitrogen)",
                        "status": "Deficient",
                        "recommendation": f"Soil Nitrogen ({n_val:.0f}) is below {crop} benchmark ({opt_n:.0f}). Apply Urea (46% N) or organic farmyard manure (FYM) during basal sowing.",
                    }
                )
            elif n_val > opt_n + 35:
                advisories.append(
                    {
                        "category": "Fertilizer (Nitrogen)",
                        "status": "Excess",
                        "recommendation": f"Excess soil Nitrogen ({n_val:.0f}). Limit top-dressing of nitrogenous fertilizer to avoid excessive vegetative growth.",
                    }
                )
            else:
                advisories.append(
                    {
                        "category": "Fertilizer (Nitrogen)",
                        "status": "Optimal",
                        "recommendation": f"Nitrogen level ({n_val:.0f}) is optimal for {crop}.",
                    }
                )

        # Phosphorus Management
        p_val = inputs["phosphorus"]
        if profile and "phosphorus" in profile:
            opt_p = profile["phosphorus"]["mean"]
            if p_val < opt_p - 20:
                advisories.append(
                    {
                        "category": "Fertilizer (Phosphorus)",
                        "status": "Deficient",
                        "recommendation": f"Phosphorus ({p_val:.0f}) is low. Apply Diammonium Phosphate (DAP) or Single Superphosphate (SSP) to boost root elongation.",
                    }
                )

        # Potassium Management
        k_val = inputs["potassium"]
        if profile and "potassium" in profile:
            opt_k = profile["potassium"]["mean"]
            if k_val < opt_k - 20:
                advisories.append(
                    {
                        "category": "Fertilizer (Potassium)",
                        "status": "Deficient",
                        "recommendation": f"Potassium ({k_val:.0f}) is low. Apply Muriate of Potash (MOP, 60% K2O) to enhance pest resilience and grain filling.",
                    }
                )

        # Soil pH Correction
        ph_val = inputs["ph"]
        if ph_val < 5.5:
            advisories.append(
                {
                    "category": "Soil pH Correction",
                    "status": "Acidic",
                    "recommendation": f"Soil pH ({ph_val:.1f}) is acidic. Incorporate agricultural lime (CaCO3) or dolomite at 250-400 kg/acre to buffer soil acidity.",
                }
            )
        elif ph_val > 7.8:
            advisories.append(
                {
                    "category": "Soil pH Correction",
                    "status": "Alkaline",
                    "recommendation": f"Soil pH ({ph_val:.1f}) is alkaline. Apply agricultural gypsum (CaSO4) and incorporate green manure / sulfur to lower alkalinity.",
                }
            )
        else:
            advisories.append(
                {
                    "category": "Soil pH",
                    "status": "Optimal",
                    "recommendation": f"Soil pH ({ph_val:.1f}) is within the optimal agronomic range.",
                }
            )

        # Water & Moisture Management
        rain_val = inputs["rainfall"]
        if profile and "rainfall" in profile:
            opt_rain = profile["rainfall"]["mean"]
            if rain_val < opt_rain - 45:
                advisories.append(
                    {
                        "category": "Irrigation Management",
                        "status": "Supplemental Required",
                        "recommendation": f"Forecast/Rainfall ({rain_val:.0f} mm) is below optimal requirement ({opt_rain:.0f} mm). Plan supplementary drip/sprinkler irrigation.",
                    }
                )

        return advisories

    def predict_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Runs bulk predictions on a dataframe.
        Expects columns: nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall.
        """
        results = []
        for idx, row in df.iterrows():
            nitrogen = row.get("nitrogen", 90.0)
            phosphorus = row.get("phosphorus", 42.0)
            potassium = row.get("potassium", 43.0)
            temperature = row.get("temperature", 25.0)
            humidity = row.get("humidity", 75.0)
            ph = row.get("ph", 6.5)
            rainfall = row.get("rainfall", 100.0)
            
            res = self.predict_and_explain(
                nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, top_k=1
            )
            
            results.append({
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
                "predicted_crop": res["top_crop"],
                "confidence": round(res["top_viability_score"] * 100, 2),
                "summary": res["human_readable_summary"]
            })
            
        return results

# Singleton helper
_engine_instance = None


def get_engine() -> OpticCropEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = OpticCropEngine()
    return _engine_instance


def predict_and_explain(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
    top_k: int = 3,
) -> Dict[str, Any]:
    engine = get_engine()
    return engine.predict_and_explain(
        nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, top_k=top_k
    )


if __name__ == "__main__":
    eng = get_engine()
    res = eng.predict_and_explain(
        nitrogen=90, phosphorus=42, potassium=43, temperature=20.8, humidity=82.0, ph=6.5, rainfall=202.9
    )
    print(f"Top Crop: {res['top_crop']} ({res['top_viability_score'] * 100:.1f}%)")
    print(f"Summary: {res['human_readable_summary']}")
    print(f"Derived: {res['derived_metrics']}")
