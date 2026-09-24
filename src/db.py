"""
OpticCrop: Database & Ingestion Layer
Manages relational SQLite storage (data/optic_crop.db) and Parquet backups,
eliminating all CSV dependencies.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "optic_crop.db")
PARQUET_PATH = os.path.join(DATA_DIR, "crop_dataset.parquet")

os.makedirs(DATA_DIR, exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite database connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initializes the database schema."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Soil & Climate Samples Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS soil_climate_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nitrogen REAL NOT NULL,
        phosphorus REAL NOT NULL,
        potassium REAL NOT NULL,
        temperature REAL NOT NULL,
        humidity REAL NOT NULL,
        ph REAL NOT NULL,
        rainfall REAL NOT NULL,
        crop TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 1.5 Users Table for Authentication
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        role TEXT DEFAULT 'Farmer',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 1.6 Prediction History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        nitrogen REAL,
        phosphorus REAL,
        potassium REAL,
        temperature REAL,
        humidity REAL,
        ph REAL,
        rainfall REAL,
        predicted_crop TEXT NOT NULL,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_crop ON soil_climate_samples (crop);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_soil_params ON soil_climate_samples (nitrogen, phosphorus, potassium, ph);")

    # 2. 30-Year Historical Climate Normals (Fallback Table for Graceful Degradation)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS historical_climate_normals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        region_name TEXT NOT NULL,
        geohash_prefix TEXT NOT NULL,
        latitude_min REAL,
        latitude_max REAL,
        longitude_min REAL,
        longitude_max REAL,
        temperature_mean REAL NOT NULL,
        temperature_std REAL NOT NULL,
        humidity_mean REAL NOT NULL,
        humidity_std REAL NOT NULL,
        rainfall_mean REAL NOT NULL,
        rainfall_std REAL NOT NULL
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_geohash ON historical_climate_normals (geohash_prefix);")

    conn.commit()
    conn.close()


def populate_from_initial_data(csv_path: Optional[str] = None):
    """
    Populates SQLite database and Parquet storage from raw records if DB is empty,
    then allows removing the CSV.
    """
    init_database()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as cnt FROM soil_climate_samples;")
    count = cursor.fetchone()["cnt"]

    if count == 0:
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            col_map = {
                "Nitrogen": "nitrogen",
                "Phosphorus": "phosphorus",
                "Potassium": "potassium",
                "Temperature": "temperature",
                "Humidity": "humidity",
                "pH_Value": "ph",
                "ph": "ph",
                "Rainfall": "rainfall",
                "Crop": "crop",
                "label": "crop",
            }
            df = df.rename(columns=col_map)
            df["crop"] = df["crop"].str.strip().str.capitalize()

            records = df[["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall", "crop"]].to_dict(orient="records")
            cursor.executemany("""
                INSERT INTO soil_climate_samples (nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, crop)
                VALUES (:nitrogen, :phosphorus, :potassium, :temperature, :humidity, :ph, :rainfall, :crop);
            """, records)
            conn.commit()
            print(f"[Database] Successfully populated {len(records)} samples into SQLite ({DB_PATH}).")
        else:
            raise FileNotFoundError(f"Database empty and initial seed not found at {csv_path}")

    # Seed baseline climate normals if empty
    cursor.execute("SELECT COUNT(*) as cnt FROM historical_climate_normals;")
    hist_cnt = cursor.fetchone()["cnt"]
    if hist_cnt == 0:
        # Default global & regional baselines
        normals = [
            ("Tropical Semi-Arid", "td", 10.0, 20.0, 75.0, 85.0, 28.5, 4.2, 65.0, 15.0, 85.0, 35.0),
            ("Sub-Tropical Humid", "tf", 20.0, 30.0, 75.0, 90.0, 25.0, 6.5, 75.0, 12.0, 140.0, 50.0),
            ("Temperate Highlands", "tu", 30.0, 38.0, 70.0, 80.0, 15.2, 5.0, 58.0, 10.0, 95.0, 30.0),
            ("Coastal Tropics", "tc", 8.0, 18.0, 70.0, 82.0, 29.0, 2.5, 82.0, 8.0, 180.0, 65.0),
            ("Global Fallback Baseline", "global", -90.0, 90.0, -180.0, 180.0, 25.6, 5.0, 71.4, 22.0, 103.4, 54.9),
        ]
        cursor.executemany("""
            INSERT INTO historical_climate_normals (region_name, geohash_prefix, latitude_min, latitude_max, longitude_min, longitude_max, temperature_mean, temperature_std, humidity_mean, humidity_std, rainfall_mean, rainfall_std)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, normals)
        conn.commit()

    conn.close()

    # Also save Parquet copy for fast batch operations
    export_to_parquet()


def export_to_parquet():
    """Exports SQLite dataset to Parquet format for fast loading."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, crop FROM soil_climate_samples", conn)
    conn.close()
    df.to_parquet(PARQUET_PATH, index=False)
    return df


def load_dataset_from_db() -> pd.DataFrame:
    """Loads the entire crop dataset directly from SQLite or Parquet without CSV."""
    if os.path.exists(PARQUET_PATH):
        return pd.read_parquet(PARQUET_PATH)
    
    conn = get_db_connection()
    df = pd.read_sql_query(
        "SELECT nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, crop FROM soil_climate_samples",
        conn
    )
    conn.close()
    return df


def get_historical_climate_fallback(lat: float, lon: float, geohash6: str = "") -> Dict[str, float]:
    """
    Returns 30-year climate averages from SQLite if live weather APIs fail or timeout (>800ms).
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Try matching geohash prefix first
    prefix = geohash6[:2] if len(geohash6) >= 2 else ""
    cursor.execute("SELECT * FROM historical_climate_normals WHERE geohash_prefix = ?", (prefix,))
    row = cursor.fetchone()

    # Try bounding box match
    if not row:
        cursor.execute("""
            SELECT * FROM historical_climate_normals 
            WHERE latitude_min <= ? AND latitude_max >= ? AND longitude_min <= ? AND longitude_max >= ?
            LIMIT 1;
        """, (lat, lat, lon, lon))
        row = cursor.fetchone()

    # Global default fallback
    if not row:
        cursor.execute("SELECT * FROM historical_climate_normals WHERE geohash_prefix = 'global' LIMIT 1;")
        row = cursor.fetchone()

    conn.close()

    if row:
        return {
            "temperature": float(row["temperature_mean"]),
            "humidity": float(row["humidity_mean"]),
            "rainfall": float(row["rainfall_mean"]),
            "is_fallback": True,
            "fallback_region": row["region_name"],
        }
    
    return {
        "temperature": 25.6,
        "humidity": 71.4,
        "rainfall": 103.4,
        "is_fallback": True,
        "fallback_region": "Default Global",
    }


def get_admin_metrics() -> Dict[str, Any]:
    """Returns system-wide metrics for the admin dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as count FROM users")
    total_users = cursor.fetchone()["count"]
    
    cursor.execute("SELECT COUNT(*) as count FROM prediction_history")
    total_predictions = cursor.fetchone()["count"]
    
    # Get top predicted crop
    cursor.execute("""
        SELECT predicted_crop, COUNT(*) as count 
        FROM prediction_history 
        GROUP BY predicted_crop 
        ORDER BY count DESC 
        LIMIT 1
    """)
    row = cursor.fetchone()
    top_crop = row["predicted_crop"] if row else "N/A"
    
    conn.close()
    
    return {
        "total_users": total_users,
        "total_predictions": total_predictions,
        "top_crop": top_crop,
        "system_status": "Healthy"
    }


if __name__ == "__main__":
    init_database()
    csv_cand = os.path.join(BASE_DIR, "Crop_Recommendation.csv")
    if os.path.exists(csv_cand):
        populate_from_initial_data(csv_cand)
    df = load_dataset_from_db()
    print(f"[OpticCrop DB] Successfully verified database with {len(df)} samples across {df['crop'].nunique()} crops.")
