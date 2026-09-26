"""
Unit tests for CropMind AI Security & Authentication Module.
"""

import pytest
from src.security import get_password_hash, verify_password, create_access_token, SECRET_KEY, ALGORITHM
from jose import jwt


def test_password_hashing_and_verification():
    raw_pwd = "SuperSecretFarmerPassword2026!"
    hashed = get_password_hash(raw_pwd)
    
    assert hashed != raw_pwd
    assert verify_password(raw_pwd, hashed) is True
    assert verify_password("WrongPassword123", hashed) is False


def test_jwt_token_generation_and_decode():
    username = "agronomist_alice"
    token = create_access_token(data={"sub": username})
    assert isinstance(token, str)
    assert len(token) > 20
    
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload.get("sub") == username
    assert "exp" in payload
