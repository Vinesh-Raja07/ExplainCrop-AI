import pytest
import sqlite3
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db import init_database, get_db_connection

def test_database_initialization():
    init_database()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if key tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
    assert cursor.fetchone() is not None
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_history';")
    assert cursor.fetchone() is not None
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='api_keys';")
    assert cursor.fetchone() is not None
    
    conn.close()
