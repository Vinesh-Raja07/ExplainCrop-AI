"""
CropMind AI: Enterprise Security & Pure-Python Resilient Authentication Layer
Provides PBKDF2-HMAC-SHA256 password hashing and URL-safe JWT access token generation
with zero hard external C-extension dependencies, ensuring 100% compatibility across
Streamlit Community Cloud, Linux, Windows, macOS, and Docker.
"""

import hashlib
import hmac
import base64
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

SECRET_KEY = "optic-crop-super-secret-key-do-not-share-in-prod"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Optional passlib / jose acceleration if present, otherwise pure-Python fallbacks
try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except Exception:
    pwd_context = None

try:
    from jose import jwt, JWTError
except Exception:
    jwt = None
    JWTError = Exception


def get_password_hash(password: str) -> str:
    """Generates a secure password hash using PBKDF2-HMAC-SHA256 or bcrypt."""
    if pwd_context is not None:
        try:
            return pwd_context.hash(password)
        except Exception:
            pass
    # Pure Python PBKDF2-HMAC-SHA256 with 100,000 iterations
    salt = secrets.token_hex(16)
    kdf = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"pbkdf2_sha256${salt}${kdf.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against stored PBKDF2 or bcrypt hash."""
    if not hashed_password:
        return False
    if hashed_password.startswith("pbkdf2_sha256$"):
        parts = hashed_password.split("$")
        if len(parts) == 3:
            salt = parts[1]
            target_hash = parts[2]
            kdf = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt.encode("utf-8"), 100_000)
            return hmac.compare_digest(kdf.hex(), target_hash)
    if pwd_context is not None:
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            pass
    # Check if direct match fallback
    return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generates a URL-safe signed JWT authentication token."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=1)
    to_encode.update({"exp": int(expire.timestamp())})
    
    if jwt is not None:
        try:
            return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        except Exception:
            pass
            
    # Pure Python HMAC-SHA256 JWT encoding
    header = {"alg": "HS256", "typ": "JWT"}
    b64_header = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    b64_payload = base64.urlsafe_b64encode(json.dumps(to_encode).encode()).decode().rstrip("=")
    signing_input = f"{b64_header}.{b64_payload}".encode("utf-8")
    signature = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    b64_sig = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    return f"{b64_header}.{b64_payload}.{b64_sig}"
