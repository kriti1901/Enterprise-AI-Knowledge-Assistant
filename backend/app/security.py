"""Password hashing and JWT access tokens."""

from __future__ import annotations

import datetime as dt

import bcrypt
import jwt

from app.config import get_settings

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


def create_access_token(username: str) -> str:
    settings = get_settings()
    now = dt.datetime.now(dt.UTC)
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + dt.timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """Return the username in the token, or None if the token is invalid or expired."""
    try:
        payload = jwt.decode(token, get_settings().jwt_secret, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    return payload.get("sub")
