"""
Unit tests for spatial farm parcel geometry, geodesic polygon area calculation,
and database persistence.
"""
import pytest
from src.spatial_parcels import calculate_polygon_geodesic_area, polygon_to_geojson
from src.db import init_database, create_farm_parcel, get_user_farm_parcels, delete_farm_parcel


def test_polygon_geodesic_area_square():
    # Roughly 100m x 100m square in Coimbatore, India (~1 hectare or 2.47 acres)
    # Approx 0.001 deg lat is ~111m, 0.001 deg lon is ~109m
    coords = [
        [11.0000, 76.9500],
        [11.0010, 76.9500],
        [11.0010, 76.9510],
        [11.0000, 76.9510],
        [11.0000, 76.9500],
    ]
    res = calculate_polygon_geodesic_area(coords)
    assert res["status"] == "success"
    assert res["area_sq_meters"] > 10000
    assert res["area_sq_meters"] < 15000
    assert res["area_acres"] > 2.0
    assert res["area_hectares"] > 1.0
    assert res["perimeter_meters"] > 300
    assert "centroid" in res
    assert 11.0000 <= res["centroid"]["lat"] <= 11.0010
    assert 76.9500 <= res["centroid"]["lon"] <= 76.9510


def test_polygon_to_geojson_format():
    coords = [
        [11.0000, 76.9500],
        [11.0010, 76.9500],
        [11.0010, 76.9510],
        [11.0000, 76.9510],
        [11.0000, 76.9500],
    ]
    import json
    geojson_str = polygon_to_geojson(coords, properties={"parcel": "Test Block"})
    data = json.loads(geojson_str)
    assert data["type"] == "Feature"
    assert data["geometry"]["type"] == "Polygon"
    assert len(data["geometry"]["coordinates"][0]) == 5
    # GeoJSON coordinates should be [lon, lat]
    assert data["geometry"]["coordinates"][0][0][0] == 76.9500
    assert data["geometry"]["coordinates"][0][0][1] == 11.0000


def test_polygon_validation_failure():
    # Less than 3 points
    coords = [
        [11.0000, 76.9500],
        [11.0010, 76.9500],
    ]
    with pytest.raises(ValueError):
        calculate_polygon_geodesic_area(coords)


def test_parcel_database_crud():
    init_database()
    # Insert test parcel for user_id = 99999
    coords = [
        [12.000, 77.000],
        [12.002, 77.000],
        [12.002, 77.002],
        [12.000, 77.002],
        [12.000, 77.000],
    ]
    geo_res = calculate_polygon_geodesic_area(coords)
    geojson_str = polygon_to_geojson(coords, properties={"name": "Paddy Field A"})
    
    # We first ensure user 1 exists or use user_id = 1
    parcel_id = create_farm_parcel(
        user_id=1,
        parcel_name="Paddy Field A",
        polygon_geojson=geojson_str,
        area_acres=geo_res["area_acres"],
        centroid_lat=geo_res["centroid"]["lat"],
        centroid_lon=geo_res["centroid"]["lon"],
        primary_crop="Rice",
        soil_type="Clay Loam"
    )
    assert parcel_id > 0
    
    parcels = get_user_farm_parcels(user_id=1)
    matching = [p for p in parcels if p["id"] == parcel_id]
    assert len(matching) == 1
    assert matching[0]["parcel_name"] == "Paddy Field A"
    assert matching[0]["primary_crop"] == "Rice"
    
    # Clean up
    deleted = delete_farm_parcel(parcel_id=parcel_id, user_id=1)
    assert deleted is True
