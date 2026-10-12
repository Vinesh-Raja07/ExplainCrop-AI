"""
CropMind AI: Farm Spatial Parcel Geometry & Geodesic Field Area Engine
Computes polygon acreage, perimeter, centroid coordinates, GeoJSON encoding, and spatial field compactness from GPS boundaries.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import math
import json


def calculate_polygon_geodesic_area(coordinates: List[Union[Tuple[float, float], List[float]]]) -> Dict[str, Any]:
    """
    Computes geodesic polygon area in acres, hectares, and sq meters from a list of (lat, lon) coordinates
    using spherical excess projection (WGS84 ellipsoid approximation).
    """
    if len(coordinates) < 3:
        raise ValueError("A valid polygon boundary requires at least 3 coordinates.")
        
    # Radius of earth in meters
    R = 6378137.0
    
    # Standardize coordinate format to (lat, lon)
    clean_coords = [(float(c[0]), float(c[1])) for c in coordinates]
    
    # Convert lat/lon to radians
    coords_rad = [(math.radians(lat), math.radians(lon)) for lat, lon in clean_coords]
    
    # Close polygon if not already closed
    if coords_rad[0] != coords_rad[-1]:
        coords_rad.append(coords_rad[0])
        clean_coords.append(clean_coords[0])
        
    area_sq_meters = 0.0
    num_pts = len(coords_rad)
    
    for i in range(num_pts - 1):
        lat1, lon1 = coords_rad[i]
        lat2, lon2 = coords_rad[i + 1]
        area_sq_meters += (lon2 - lon1) * (2.0 + math.sin(lat1) + math.sin(lat2))
        
    area_sq_meters = abs(area_sq_meters * (R ** 2) / 2.0)
    
    # Conversions
    area_acres = area_sq_meters / 4046.8564224
    area_hectares = area_sq_meters / 10000.0
    
    # Perimeter
    total_dist = 0.0
    for i in range(len(clean_coords) - 1):
        lat1, lon1 = math.radians(clean_coords[i][0]), math.radians(clean_coords[i][1])
        lat2, lon2 = math.radians(clean_coords[i+1][0]), math.radians(clean_coords[i+1][1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        total_dist += R * c
        
    # Centroid
    lats = [c[0] for c in clean_coords[:-1]]
    lons = [c[1] for c in clean_coords[:-1]]
    centroid_lat = round(sum(lats) / len(lats), 6)
    centroid_lon = round(sum(lons) / len(lons), 6)
    
    # Shape compactness index
    compactness = (4.0 * math.pi * area_sq_meters) / (total_dist ** 2) if total_dist > 0 else 0.0
    compactness = round(min(1.0, max(0.01, compactness)), 3)
    
    return {
        "status": "success",
        "area_acres": round(area_acres, 3),
        "area_hectares": round(area_hectares, 3),
        "area_sq_meters": round(area_sq_meters, 1),
        "perimeter_meters": round(total_dist, 1),
        "centroid": {"lat": centroid_lat, "lon": centroid_lon},
        "compactness_index": compactness,
        "is_workable": compactness >= 0.45
    }


def calculate_polygon_area_acres(coordinates: List[Union[Tuple[float, float], List[float]]]) -> float:
    """Convenience function returning just the acreage."""
    if len(coordinates) < 3:
        return 0.0
    res = calculate_polygon_geodesic_area(coordinates)
    return res["area_acres"]


def calculate_polygon_perimeter_meters(coordinates: List[Union[Tuple[float, float], List[float]]]) -> float:
    """Convenience function returning perimeter in meters."""
    if len(coordinates) < 2:
        return 0.0
    res = calculate_polygon_geodesic_area(coordinates)
    return res["perimeter_meters"]


def calculate_polygon_centroid(coordinates: List[Union[Tuple[float, float], List[float]]]) -> Tuple[float, float]:
    """Convenience function returning centroid (lat, lon)."""
    if not coordinates:
        return (0.0, 0.0)
    lats = [float(c[0]) for c in coordinates]
    lons = [float(c[1]) for c in coordinates]
    return (round(sum(lats) / len(lats), 6), round(sum(lons) / len(lons), 6))


def polygon_to_geojson(coordinates: List[Union[Tuple[float, float], List[float]]], properties: Optional[Dict[str, Any]] = None) -> str:
    """Encodes coordinates into standard GeoJSON Feature (Polygon format with [lon, lat])."""
    # GeoJSON requires [lon, lat] coordinates
    poly_coords = [[float(c[1]), float(c[0])] for c in coordinates]
    if poly_coords[0] != poly_coords[-1]:
        poly_coords.append(poly_coords[0])
        
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [poly_coords]
        },
        "properties": properties or {}
    }
    return json.dumps(feature)


def analyze_farm_parcel(
    parcel_name: str,
    boundary_coordinates: List[Tuple[float, float]],
    soil_type: str = "Clay Loam",
    primary_crop: Optional[str] = "Rice"
) -> Dict[str, Any]:
    """
    Performs complete spatial geometry evaluation for a farm boundary.
    """
    geo_res = calculate_polygon_geodesic_area(boundary_coordinates)
    
    return {
        "parcel_name": parcel_name,
        "primary_crop": primary_crop,
        "soil_type": soil_type,
        "vertices_count": len(boundary_coordinates),
        "centroid_lat_lon": {"latitude": geo_res["centroid"]["lat"], "longitude": geo_res["centroid"]["lon"]},
        "geometry": {
            "area_acres": geo_res["area_acres"],
            "area_hectares": geo_res["area_hectares"],
            "perimeter_meters": geo_res["perimeter_meters"],
            "isoperimetric_compactness_score": geo_res["compactness_index"],
            "field_shape_classification": "Regular / High Workability" if geo_res["compactness_index"] >= 0.45 else "Irregular / Narrow Strips"
        },
        "boundary_coordinates": [{"latitude": c[0], "longitude": c[1]} for c in boundary_coordinates]
    }
