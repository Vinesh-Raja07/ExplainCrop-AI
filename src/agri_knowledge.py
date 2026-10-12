"""
CropMind AI: ICAR & FAO Agronomic Knowledge Base & Semantic Search Engine
Provides expert agronomic guidance, Package of Practices (POP), IPM protocols,
and ICAR/FAO certified management practices for Indian and tropical agriculture.
"""

from typing import Dict, Any, List, Optional
import re


# Curated ICAR & FAO Knowledge Repository
AGRI_KNOWLEDGE_BASE = [
    {
        "id": "KB-RICE-01",
        "crop": "Rice",
        "category": "Seed Treatment",
        "title": "ICAR Recommended Seed Treatment for Paddy",
        "keywords": ["seed", "fungus", "blast", "blight", "trichoderma", "pseudomonas"],
        "summary": "Treat paddy seeds with Pseudomonas fluorescens @ 10g/kg or Carbendazim 50% WP @ 2g/kg in 1 litre water to protect against seed-borne blast, brown spot, and bacterial leaf blight.",
        "protocol": [
            "Soak seeds in bio-control slurry (Pseudomonas fluorescens @ 10g/kg of seeds) for 12-24 hours.",
            "Shade dry treated seeds for 30 minutes before broadcasting or nursery sowing.",
            "For wet nursery, incubate treated seeds under moist gunny bags for 24 hours to initiate sprouting."
        ],
        "source": "ICAR-NRRI (National Rice Research Institute) Package of Practices"
    },
    {
        "id": "KB-RICE-02",
        "crop": "Rice",
        "category": "Water Management",
        "title": "Alternate Wetting and Drying (AWD) Water Conservation",
        "keywords": ["water", "awd", "irrigation", "methane", "conservation", "flooding"],
        "summary": "Implement Alternate Wetting and Drying (AWD) using field water tubes (pani pipe) to save 25-30% water and reduce methane emissions by 30-50% without penalty on paddy yield.",
        "protocol": [
            "Install perforated PVC tube (30cm length, 15cm diameter) 20cm into the soil after transplanting.",
            "Re-irrigate up to 5cm standing depth only when water level inside the pipe drops to 15cm below the soil surface.",
            "Maintain continuous shallow submergence (2-4 cm) during flowering and grain-filling stages."
        ],
        "source": "IRRI & ICAR Water Management Advisory"
    },
    {
        "id": "KB-WHEAT-01",
        "crop": "Wheat",
        "category": "Irrigation Stages",
        "title": "Critical Growth Stages for Wheat Irrigation",
        "keywords": ["wheat", "irrigation", "cri", "crown root", "booting", "flowering"],
        "summary": "Crown Root Initiation (CRI at 20-25 DAS) is the most critical irrigation stage. Delaying irrigation at CRI reduces tillering and final yield by up to 25%.",
        "protocol": [
            "1st Irrigation: Crown Root Initiation (21-25 days after sowing) - mandatory.",
            "2nd Irrigation: Tillering stage (40-45 DAS).",
            "3rd Irrigation: Late jointing stage (60-65 DAS).",
            "4th Irrigation: Flowering / Heading stage (80-85 DAS).",
            "5th Irrigation: Milking stage (100-105 DAS) - avoid heavy irrigation during windy days to prevent lodging."
        ],
        "source": "ICAR-IIWBR (Indian Institute of Wheat & Barley Research)"
    },
    {
        "id": "KB-COTTON-01",
        "crop": "Cotton",
        "category": "Integrated Pest Management",
        "title": "Pink Bollworm (PBW) & Sucking Pest IPM Strategy",
        "keywords": ["cotton", "bollworm", "pink bollworm", "whitefly", "neem", "traps", "pheromones"],
        "summary": "Install Pheromone traps @ 5/acre for monitoring and 20/acre for mass trapping of Pink Bollworm. Spray Neem oil (1500 ppm) @ 5ml/litre at initial infestation.",
        "protocol": [
            "Erect yellow sticky traps @ 10/acre for whiteflies and aphids.",
            "Install Gossyplure pheromone traps @ 5/acre at 45 DAS to monitor male moth catches.",
            "Spray 5% NSKE (Neem Seed Kernel Extract) or Azadirachtin 0.03% EC @ 5ml/L at ETL (8 moths/trap/night for 3 consecutive days).",
            "Release egg parasitoid Trichogramma bactrae @ 60,000/acre at weekly intervals."
        ],
        "source": "ICAR-CICR (Central Institute for Cotton Research)"
    },
    {
        "id": "KB-MAIZE-01",
        "crop": "Maize",
        "category": "Pest Management",
        "title": "Fall Armyworm (Spodoptera frugiperda) Management",
        "keywords": ["maize", "corn", "fall armyworm", "faw", "whorl", "emamectin"],
        "summary": "Early detection in leaf whorls is crucial. Apply Metarhizium anisopliae or Bacillus thuringiensis (Bt) in early stages, or Emamectin benzoate 5% SG @ 0.4g/L directly into leaf whorls.",
        "protocol": [
            "Deep summer ploughing to expose pupae to predatory birds and sun.",
            "Intercrop maize with cowpea or pigeonpea (4:1 ratio) to suppress FAW oviposition.",
            "Apply sand + neem cake mixture (9:1 ratio) or ash into whorls of young maize plants.",
            "If infestation exceeds 10% damaged whorls, apply Emamectin benzoate 5% SG @ 0.4 g/L targeting the central whorl."
        ],
        "source": "ICAR-IIMR (Indian Institute of Maize Research)"
    },
    {
        "id": "KB-SOIL-01",
        "crop": "General",
        "category": "Soil Health & Biofertilizers",
        "title": "Biofertilizer Inoculation & Mycorrhizal Application",
        "keywords": ["biofertilizer", "rhizobium", "azospirillum", "vam", "mycorrhiza", "psb"],
        "summary": "Inoculate seeds/soil with Azospirillum/Rhizobium and Phosphate Solubilizing Bacteria (PSB) to mobilize 20-25% fixed phosphorus and fix 20-30 kg biological N/ha.",
        "protocol": [
            "For cereals & millets: Use Azospirillum + PSB (Phosphate Solubilizing Bacteria) @ 2 kg/ha mixed with 200 kg FYM.",
            "For pulses & legumes: Treat seeds with Rhizobium culture (200g/10kg seed) using rice gruel as adhesive.",
            "Apply Vesicular-Arbuscular Mycorrhiza (VAM) @ 5 kg/acre near the root zone to enhance root branching and drought tolerance."
        ],
        "source": "ICAR-IISS (Indian Institute of Soil Science)"
    },
    {
        "id": "KB-POSTHARVEST-01",
        "crop": "General",
        "category": "Post-Harvest Management",
        "title": "Safe Grain Storage & Moisture Management",
        "keywords": ["storage", "moisture", "grain", "weevil", "hermetic", "fungus"],
        "summary": "Dry grains to recommended safe moisture content (Paddy 13-14%, Wheat 10-12%, Maize 12%, Pulses 9-10%) before bagging to prevent Aspergillus flavus aflatoxin and grain borers.",
        "protocol": [
            "Dry harvest on clean tarpaulins under full sunlight for 2-3 days until moisture drops below 12%.",
            "Store in Purdue Improved Crop Storage (PICS) multi-layer hermetic bags or galvanized iron silos to suffocate stored grain pests.",
            "Clean and fumigate storage godowns with malathion 50% EC spray (1:100 dilution) before loading new stock."
        ],
        "source": "FAO Post-Harvest Compendium & ICAR-CIPHET"
    }
]


def search_agronomic_knowledge(
    query: str,
    crop: Optional[str] = None,
    category: Optional[str] = None,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Performs keyword & semantic token matching across ICAR and FAO agronomic knowledge guides.
    """
    if not query and not crop and not category:
        return {
            "status": "success",
            "query": "",
            "total_matches": len(AGRI_KNOWLEDGE_BASE),
            "results": AGRI_KNOWLEDGE_BASE[:max_results]
        }

    tokens = [t.lower() for t in re.findall(r'\w+', query)] if query else []
    
    scored_results = []
    
    for item in AGRI_KNOWLEDGE_BASE:
        score = 0.0
        
        # Crop matching
        if crop:
            if item["crop"].lower() == crop.lower():
                score += 3.0
            elif item["crop"].lower() == "general":
                score += 1.0
                
        # Category matching
        if category and item["category"].lower() == category.lower():
            score += 2.5
            
        # Keyword & text token scoring
        item_keywords = [k.lower() for k in item["keywords"]]
        item_text = (item["title"] + " " + item["summary"] + " " + " ".join(item["protocol"])).lower()
        
        for token in tokens:
            if token in item_keywords:
                score += 2.0
            if token in item["crop"].lower():
                score += 2.5
            if token in item["title"].lower():
                score += 1.5
            if token in item_text:
                score += 0.8
                
        if score > 0:
            scored_results.append({
                "relevance_score": round(score, 2),
                **item
            })
            
    # Sort descending by relevance score
    scored_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    selected = scored_results[:max_results]
    
    return {
        "status": "success",
        "query": query,
        "filters": {"crop": crop, "category": category},
        "total_matches": len(scored_results),
        "results": selected
    }


def get_all_categories() -> List[str]:
    """Returns unique knowledge categories."""
    cats = list(set([item["category"] for item in AGRI_KNOWLEDGE_BASE]))
    cats.sort()
    return cats
