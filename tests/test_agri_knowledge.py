"""
Unit tests for ICAR & FAO agronomic knowledge search engine.
"""
from src.agri_knowledge import search_agronomic_knowledge, get_all_categories, AGRI_KNOWLEDGE_BASE


def test_knowledge_base_not_empty():
    assert len(AGRI_KNOWLEDGE_BASE) >= 5
    categories = get_all_categories()
    assert len(categories) >= 3
    assert "Water Management" in categories or "Seed Treatment" in categories


def test_search_by_crop():
    res = search_agronomic_knowledge(query="", crop="Rice")
    assert res["status"] == "success"
    assert res["total_matches"] >= 2
    for item in res["results"]:
        assert item["crop"].lower() in ["rice", "general"]


def test_search_by_keyword():
    res = search_agronomic_knowledge(query="bollworm")
    assert res["status"] == "success"
    assert res["total_matches"] >= 1
    top_match = res["results"][0]
    assert top_match["crop"] == "Cotton"
    assert "Pink Bollworm" in top_match["title"] or "bollworm" in str(top_match["keywords"])


def test_search_empty_returns_defaults():
    res = search_agronomic_knowledge(query="", max_results=3)
    assert res["status"] == "success"
    assert len(res["results"]) == 3
